import heapq
from typing import List, Iterator, Optional, Tuple
import os
import sys
import json
from datetime import datetime
import struct

# Try to import tqdm, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False



class VarByteCodec:
    """Varbyte encoding/decoding for integers"""
    
    @staticmethod
    def encode(number: int) -> bytes:
        """
            Encode single integer to varbyte format
            Uses 7 bits for data, 1 bit for continuation
        """
        if number < 0:
            raise ValueError("VarByte encoding only supports non-negative integers")
        
        # Special case: 0
        if number == 0:
            return bytes([0])
        
        # Collect bytes in reverse order
        byte_list = []
        
        while number > 0:
            # Extract lowest 7 bits
            byte_list.append(number & 0x7F)  # 0x7F = 0b01111111
            number >>= 7
        
        # Reverse to get proper order (most significant first)
        byte_list.reverse()
        
        # Set continuation bit (MSB = 1) for all bytes except the last
        for i in range(len(byte_list) - 1):
            byte_list[i] |= 0x80  # Set MSB to 1
        
        return bytes(byte_list)
    
    @staticmethod
    def encode_list(numbers: List[int]) -> bytes:
        """
            Encode list of integers (for docIDs or frequencies)
            Returns: compressed bytes
        """
        result = bytearray()
        for num in numbers:
            result.extend(VarByteCodec.encode(num))
        return bytes(result)
    
    @staticmethod
    def decode(data: bytes, offset: int = 0) -> tuple:
        """
            Decode a single integer from varbyte format
        """
        number = 0
        bytes_read = 0
        
        while offset + bytes_read < len(data):
            byte = data[offset + bytes_read]
            bytes_read += 1
            
            # Extract the 7 data bits
            number = (number << 7) | (byte & 0x7F)
            
            # Check continuation bit (MSB)
            if (byte & 0x80) == 0:  # MSB is 0, this is the last byte
                break
        
        return number, bytes_read
    
    @staticmethod
    def decode_all(data: bytes) -> list:
        """Decode all integers from varbyte encoded data"""
        numbers = []
        offset = 0
        
        while offset < len(data):
            number, bytes_read = VarByteCodec.decode(data, offset)
            numbers.append(number)
            offset += bytes_read
        
        return numbers
    
    @staticmethod
    def decode_list(data: bytes, count: int = None) -> list:
        """
            Decode a list of integers from varbyte format
        """
        numbers = []
        offset = 0
        decoded_count = 0
        
        while offset < len(data):
            if count is not None and decoded_count >= count:
                break
            
            number, bytes_consumed = VarByteCodec.decode(data, offset)
            numbers.append(number)
            offset += bytes_consumed
            decoded_count += 1
        
        return numbers
    
    @staticmethod
    def decode_delta_list(data: bytes, count: int = None) -> list:
        """
            Decode a delta-encoded list (converts deltas back to original values)
        """
        deltas = VarByteCodec.decode_list(data, count)
        
        if not deltas:
            return []
        
        # Convert deltas back to original values
        original = [deltas[0]]
        for i in range(1, len(deltas)):
            original.append(original[-1] + deltas[i])
        
        return original


class BlockReader:
    """
    Reads a single sorted block file
    Yields (term, postings_list) tuples
    """
    
    def __init__(self, block_path: str):
        """
            Initialize reader for a block file
        """
        self.block_path = block_path
        self.file = open(block_path, 'r', encoding='utf-8')
        self.current_term = None
        self.current_postings = None
        self.exhausted = False
        
        # Read first line to initialize
        self._read_next_line()

    def _read_next_line(self):
        """
            Read next line from file and parse it
            Updates self.current_term and self.current_postings
        """
        line = self.file.readline()
        
        if not line:
            # End of file
            self.exhausted = True
            self.current_term = None
            self.current_postings = None
            return
        
        line = line.strip()
        if not line:
            # Empty line, try next
            self._read_next_line()
            return
        
        # Parse line: term\tdoc_id:freq,doc_id:freq,...
        try:
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Malformed line in {self.block_path}: {line}")
                self._read_next_line()
                return
            
            term = parts[0]
            postings_str = parts[1]
            
            # Parse postings: "0:2,1:1,2:1,5:3" -> [(0,2), (1,1), (2,1), (5,3)]
            postings = []
            for posting in postings_str.split(','):
                doc_id_str, freq_str = posting.split(':')
                doc_id = int(doc_id_str)
                freq = int(freq_str)
                postings.append((doc_id, freq))
            
            self.current_term = term
            self.current_postings = postings
            
        except Exception as e:
            print(f"Error parsing line in {self.block_path}: {line}")
            print(f"Exception: {e}")
            self._read_next_line()
    
    def peek(self) -> Optional[str]:
        """Return current term without advancing"""
        return self.current_term
    
    def next(self) -> Tuple[str, List[Tuple[int, int]]]:
        """
            Return current (term, postings) and advance to next
        """
        if self.exhausted:
            return None, []
        
        # Save current values
        term = self.current_term
        postings = self.current_postings
        
        # Advance to next line
        self._read_next_line()
        
        return term, postings
    
    def is_exhausted(self) -> bool:
        """Check if reader has no more data"""
        return self.exhausted
    
    def close(self):
        """Close file handle"""
        if self.file:
            self.file.close()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()


class KWayMerger:

    """
        Merges K sorted block files into single sorted stream
        Uses min-heap for efficient merging
    """

    def __init__(self, block_paths: List[str]):

        self.readers = [BlockReader(path) for path in block_paths]
        self.heap = []
        self._initialize_heap()

    def _initialize_heap(self):
        '''
            Push first term from each reader onto heap.
        '''
        for idx, reader in enumerate(self.readers):
            term = reader.peek()
            if term:
                heapq.heappush(self.heap, (term, idx))

    def merge(self) -> Iterator[Tuple[str, List[Tuple[int, int]]]]:
        '''
            Yields: (term, merged_postings_list)
            merged_postings_list = [(doc_id, freq), ...] sorted by doc_id
        '''

        while self.heap:
            # Get terms with smallest lexicographical value.
            current_term, _ = self.heap[0]

            # Collect all posting for this term from all blocks.
            merged_postings = []

            while self.heap and self.heap[0][0] == current_term:
                _, reader_idx = heapq.heappop(self.heap)
                reader = self.readers[reader_idx]
                
                term, postings = reader.next()
                merged_postings.extend(postings)
                
                # Push next term from this reader
                next_term = reader.peek()
                if next_term:
                    heapq.heappush(self.heap, (next_term, reader_idx))
            
            # Sort merged postings by doc_id (required by assignment)
            merged_postings.sort(key=lambda x: x[0])
            
            yield current_term, merged_postings
    
    def close(self):
        """Close all readers"""
        for reader in self.readers:
            reader.close()

class InvertedListReader:
    """
    Reader for compressed inverted lists with DAAT traversal support.
    Implements the API framework from the assignment.
    """
    
    def __init__(self, inverted_lists_path: str):
        self.inverted_lists_path = inverted_lists_path
        self.file = None
        
        # Current list state
        self.doc_ids = []
        self.frequencies = []
        self.current_index = -1
        self.list_length = 0
    
    def open(self, term: str, lexicon: 'Lexicon') -> bool:
        """
        Open inverted list for a term.
        
        Args:
            term: The term to open
            lexicon: Lexicon containing term information
            
        Returns:
            True if successful, False if term not found
        """
        # Check if term exists in lexicon
        if term not in lexicon.entries:
            return False

        term_info = lexicon.entries[term]
        offset = term_info['offset']
        
        # Open file and seek to position
        self.file = open(self.inverted_lists_path, 'rb')
        self.file.seek(offset)
        
        # Read header
        header = self.file.read(12)
        doc_count, docid_length, freq_length = struct.unpack('III', header)
        
        # Read compressed docIDs
        docid_data = self.file.read(docid_length)
        delta_docids = VarByteCodec.decode_all(docid_data)
        
        # Decode delta-encoded docIDs
        self.doc_ids = []
        current_doc_id = 0
        for delta in delta_docids:
            current_doc_id += delta
            self.doc_ids.append(current_doc_id)
        
        # Read compressed frequencies
        freq_data = self.file.read(freq_length)
        self.frequencies = VarByteCodec.decode_all(freq_data)

        # Verify decoded data matches expected count
        if len(self.doc_ids) != doc_count or len(self.frequencies) != doc_count:
            print(f"WARNING: term={term}, expected={doc_count}, got docids={len(self.doc_ids)}, freqs={len(self.frequencies)}")
            return False
        
        self.list_length = doc_count
        self.current_index = -1
        
        return True
    
    def next_doc(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Move to next document in the list.
        
        Returns:
            Tuple of (doc_id, frequency) or (None, None) if at end
        """
        self.current_index += 1
        
        if self.current_index >= self.list_length:
            return None, None
        
        return self.doc_ids[self.current_index], self.frequencies[self.current_index]
    
    def seek(self, target_doc_id: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Seek to the first document with doc_id >= target_doc_id.
        
        Args:
            target_doc_id: Target document ID
            
        Returns:
            Tuple of (doc_id, frequency) or (None, None) if not found
        """
        # Binary search for target_doc_id
        left = self.current_index + 1
        right = self.list_length - 1
        result_index = -1
        
        while left <= right:
            mid = (left + right) // 2
            if self.doc_ids[mid] >= target_doc_id:
                result_index = mid
                right = mid - 1
            else:
                left = mid + 1
        
        if result_index == -1:
            # No document >= target_doc_id found
            self.current_index = self.list_length
            return None, None
        
        self.current_index = result_index
        return self.doc_ids[self.current_index], self.frequencies[self.current_index]
    
    def get_frequency(self) -> int:
        """
        Get frequency for current document.
        
        Returns:
            Frequency of current posting
        """
        if 0 <= self.current_index < self.list_length:
            return self.frequencies[self.current_index]
        return 0
    
    def get_doc_id(self) -> Optional[int]:
        """
        Get current document ID.
        
        Returns:
            Current doc_id or None
        """
        if 0 <= self.current_index < self.list_length:
            return self.doc_ids[self.current_index]
        return None
    
    def close(self):
        """Close the inverted list and file."""
        if self.file:
            self.file.close()
            self.file = None
        self.doc_ids = []
        self.frequencies = []
        self.current_index = -1
        self.list_length = 0

class InvertedListWriter:
    """
    Writes compressed inverted lists to binary file
    Maintains separate blocks for docIDs and frequencies
    """
    
    def __init__(self, output_path: str, block_size: int = 128):
        """
        block_size: number of postings per block (for compression)
        """
        self.file = open(output_path, 'wb')
        self.current_offset = 0
        self.block_size = block_size
        self.codec = VarByteCodec()
    
    def write_inverted_list(self, postings: List[Tuple[int, int]]) -> dict:
        """
        Write compressed inverted list for one term
        
        Returns: metadata dict with:
            - offset: byte offset where list starts
            - length: number of bytes written
            - doc_count: number of documents in list
        """
        start_offset = self.current_offset
        
        # Separate docIDs and frequencies
        doc_ids = [doc_id for doc_id, freq in postings]
        freqs = [freq for doc_id, freq in postings]
        
        # Delta-encode docIDs (improves compression)
        delta_doc_ids = self._delta_encode(doc_ids)
        
        # Compress both lists
        compressed_doc_ids = self.codec.encode_list(delta_doc_ids)
        compressed_freqs = self.codec.encode_list(freqs)
        
        # Write header: [doc_count][docID_len][freq_len]
        header = self._encode_header(len(postings), 
                                     len(compressed_doc_ids), 
                                     len(compressed_freqs))
        self.file.write(header)
        
        # Write compressed data
        self.file.write(compressed_doc_ids)
        self.file.write(compressed_freqs)
        
        bytes_written = len(header) + len(compressed_doc_ids) + len(compressed_freqs)
        self.current_offset += bytes_written
        
        return {
            'offset': start_offset,
            'length': bytes_written,
            'doc_count': len(postings)
        }
    
    @staticmethod
    def _delta_encode(doc_ids: List[int]) -> List[int]:
        """Convert [5, 8, 12, 15] → [5, 3, 4, 3]"""
        if not doc_ids:
            return []
        deltas = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            deltas.append(doc_ids[i] - doc_ids[i-1])
        return deltas
    
    def _encode_header(self, doc_count: int, docid_len: int, freq_len: int) -> bytes:
        """
        Encode header as fixed-size integers
        Format: [4 bytes doc_count][4 bytes docid_len][4 bytes freq_len]
        """
        import struct
        return struct.pack('III', doc_count, docid_len, freq_len)
    
    def close(self):
        self.file.close()


class Lexicon:
    """
    In-memory dictionary: term → inverted list metadata
    Saved to disk as binary file
    """
    
    def __init__(self):
        self.entries = {}  # term → {'offset': int, 'length': int, 'doc_count': int}
        self.num_terms = 0
    
    def add_term(self, term: str, metadata: dict):
        """Add term with its inverted list metadata"""
        if term not in self.entries:
            self.num_terms += 1
        self.entries[term] = metadata


    def save(self, output_path: str):
        """
        Save lexicon to binary file
        Format: [num_terms][term_len|term|offset|length|doc_count]...
        """
        import struct
        
        with open(output_path, 'wb') as f:
            # Write number of terms
            f.write(struct.pack('I', len(self.entries)))
            
            # Write each entry
            for term, meta in sorted(self.entries.items()):
                term_bytes = term.encode('utf-8')
                
                # Format: [term_len][term][offset][length][doc_count]
                entry = struct.pack('I', len(term_bytes))  # term length
                entry += term_bytes                         # term string
                entry += struct.pack('QQI',                 # 3 metadata fields
                                     meta['offset'],
                                     meta['length'],
                                     meta['doc_count'])
                f.write(entry)
    
    @staticmethod
    def load(input_path: str) -> 'Lexicon':
        """Load lexicon from binary file."""
        lex = Lexicon()
        
        with open(input_path, 'rb') as f:
            # Read number of terms
            num_terms = struct.unpack('I', f.read(4))[0]
            
            # Read each term entry
            for _ in range(num_terms):
                # Read term
                term_length = struct.unpack('I', f.read(4))[0]
                term = f.read(term_length).decode('utf-8')
                
                # Read metadata
                offset = struct.unpack('Q', f.read(8))[0]
                length = struct.unpack('Q', f.read(8))[0]
                doc_count = struct.unpack('I', f.read(4))[0]
                
                lex.entries[term] = {
                    'offset': offset,
                    'length': length,
                    'doc_count': doc_count
                }
        
        return lex


class DocumentTable:
    """
    Stores document metadata: doc_id → passage_id, length
    """
    
    def __init__(self):
        self.documents = {}  # doc_id → {'passage_id': str, 'length': int}
        self.doc_id_to_passage = {}  # doc_id → passage_id
        self.doc_id_to_length = {}   # doc_id → document length
        self.num_docs = 0
        self.avg_doc_length = 0.0
    
    def load_from_json(self, metadata_path: str):
        """Load from Part 1's doc_metadata.json"""
        import json
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
    
    def save(self, output_path: str):
        """
        Save as binary file for efficiency
        Format: [num_docs][passage_id_len|passage_id|length]...
        """
        import struct
         
        with open(output_path, 'wb') as f:
            f.write(struct.pack('I', len(self.documents)))
            
            for doc_id_str in sorted(self.documents.keys(), key=int):
                doc_id = int(doc_id_str)
                meta = self.documents[doc_id_str]
                
                passage_id_bytes = meta['passage_id'].encode('utf-8')
                
                entry = struct.pack('II', doc_id, len(passage_id_bytes))
                entry += passage_id_bytes
                entry += struct.pack('I', meta['length'])
                
                f.write(entry)

    def get_passage_id(self, doc_id: int) -> str:
        """Get passage_id for a given doc_id."""
        return self.doc_id_to_passage.get(doc_id, f"doc_{doc_id}")
    
    def get_doc_length(self, doc_id: int) -> int:
        """Get document length for a given doc_id."""
        return self.doc_id_to_length.get(doc_id, 0)
        
    @staticmethod
    def load(filepath: str) -> 'DocumentTable':
        """
        Load document table from binary file.
        
        File format:
        [num_docs: 4 bytes]
        For each document:
          [doc_id: 4 bytes]
          [passage_id_length: 4 bytes]
          [passage_id: UTF-8 bytes]
          [doc_length: 4 bytes]
        """
        doc_table = DocumentTable()
        
        with open(filepath, 'rb') as f:
            # Read number of documents
            num_docs_bytes = f.read(4)
            doc_table.num_docs = struct.unpack('I', num_docs_bytes)[0]
            
            total_length = 0
            
            # Read each document
            for _ in range(doc_table.num_docs):
                # Read doc_id
                doc_id = struct.unpack('I', f.read(4))[0]
                
                # Read passage_id
                passage_id_length = struct.unpack('I', f.read(4))[0]
                passage_id = f.read(passage_id_length).decode('utf-8')
                
                # Read document length
                doc_length = struct.unpack('I', f.read(4))[0]
                
                # Store mappings
                doc_table.doc_id_to_passage[doc_id] = passage_id
                doc_table.doc_id_to_length[doc_id] = doc_length
                total_length += doc_length
            
            # Calculate average document length from loaded data
            if doc_table.num_docs > 0:
                doc_table.avg_doc_length = total_length / doc_table.num_docs
            else:
                doc_table.avg_doc_length = 0.0
        
        return doc_table


class IndexMerger:
    """
    Main class that orchestrates the entire merging process
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def merge(self):
        """Execute the full merge pipeline"""
        import time
        start_time = time.time()

        print("="*70)
        print("Starting index merge...")
        print("="*70)

        # 1. Find all block files
        block_files = self._find_block_files()
        print(f"\n✓ Found {len(block_files)} block files to merge")

        # 2. Initialize components
        print("\nInitializing merger components...")
        merger = KWayMerger(block_files)
        list_writer = InvertedListWriter(
            os.path.join(self.output_dir, 'inverted_lists.bin')
        )
        lexicon = Lexicon()

        # 3. Merge and write inverted lists
        print("\nMerging inverted lists...")
        print("-"*70)
        term_count = 0
        total_postings = 0

        # Use tqdm if available
        if HAS_TQDM:
            # We don't know total terms upfront, so use no total (shows count only)
            merge_iter = merger.merge()
            pbar = tqdm(merge_iter, desc="Merging", unit=" terms", unit_scale=True)

            for term, postings in pbar:
                # Write compressed inverted list
                metadata = list_writer.write_inverted_list(postings)

                # Add to lexicon
                lexicon.add_term(term, metadata)

                term_count += 1
                total_postings += len(postings)

                # Update progress bar with stats
                if term_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    terms_per_sec = term_count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'terms/sec': f'{terms_per_sec:.0f}',
                        'postings': f'{total_postings:,}'
                    })

            pbar.close()
        else:
            # Fallback to simple progress
            for term, postings in merger.merge():
                # Write compressed inverted list
                metadata = list_writer.write_inverted_list(postings)

                # Add to lexicon
                lexicon.add_term(term, metadata)

                term_count += 1
                total_postings += len(postings)

                if term_count % 10000 == 0:
                    elapsed = time.time() - start_time
                    terms_per_sec = term_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {term_count:,} terms ({terms_per_sec:.0f} terms/sec, "
                          f"{total_postings:,} postings)", end='\r')

        print(f"\n✓ Processed {term_count:,} terms, {total_postings:,} total postings")

        list_writer.close()
        merger.close()

        # 4. Save lexicon
        print("\nSaving lexicon...")
        lexicon_path = os.path.join(self.output_dir, 'lexicon.bin')
        lexicon.save(lexicon_path)
        lexicon_size_mb = os.path.getsize(lexicon_path) / (1024 * 1024)
        print(f"✓ Lexicon saved ({lexicon_size_mb:.2f} MB)")

        # 5. Process document table
        print("\nProcessing document table...")
        doc_table = DocumentTable()
        doc_table.load_from_json(
            os.path.join(self.input_dir, 'doc_metadata.json')
        )
        doc_table_path = os.path.join(self.output_dir, 'doc_table.bin')
        doc_table.save(doc_table_path)
        doc_table_size_mb = os.path.getsize(doc_table_path) / (1024 * 1024)
        print(f"✓ Document table saved ({doc_table_size_mb:.2f} MB)")

        # 6. Save index metadata
        self._save_metadata(term_count, len(doc_table.documents))

        # Final statistics
        elapsed_time = time.time() - start_time
        inverted_lists_size = os.path.getsize(os.path.join(self.output_dir, 'inverted_lists.bin')) / (1024 * 1024)
        total_size = inverted_lists_size + lexicon_size_mb + doc_table_size_mb

        print("\n" + "="*70)
        print("MERGE COMPLETE!")
        print("="*70)
        print(f"Terms indexed:        {term_count:,}")
        print(f"Total postings:       {total_postings:,}")
        print(f"Documents:            {len(doc_table.documents):,}")
        print(f"Time elapsed:         {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"\nIndex size breakdown:")
        print(f"  Inverted lists:     {inverted_lists_size:.2f} MB")
        print(f"  Lexicon:            {lexicon_size_mb:.2f} MB")
        print(f"  Document table:     {doc_table_size_mb:.2f} MB")
        print(f"  Total:              {total_size:.2f} MB")
        print(f"\nOutput directory:     {self.output_dir}")
        print("="*70)
    
    def _find_block_files(self) -> List[str]:
        """Find all block_*.txt files"""
        import glob
        pattern = os.path.join(self.input_dir, 'block_*.txt')
        return sorted(glob.glob(pattern))
    
    def _save_metadata(self, term_count: int, doc_count: int):
        """Save index statistics"""
        import json
        metadata = {
            'term_count': term_count,
            'doc_count': doc_count,
            'compression': 'varbyte',
            'created': str(datetime.now())
        }
        with open(os.path.join(self.output_dir, 'index_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge intermediate index blocks into final compressed index (Part 2)"
    )
    parser.add_argument('input_dir',
                       help='Input directory containing block_*.txt files from indexer')
    parser.add_argument('output_dir',
                       help='Output directory for final compressed index')

    args = parser.parse_args()

    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Check for block files
    import glob
    block_files = glob.glob(os.path.join(args.input_dir, 'block_*.txt'))
    if not block_files:
        print(f"Error: No block_*.txt files found in {args.input_dir}", file=sys.stderr)
        print("Make sure you've run indexer.py first!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(block_files)} block files in {args.input_dir}")

    merger = IndexMerger(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    merger.merge()