import sys
import os
import re
import json
import struct
from collections import defaultdict
from pathlib import Path
import argparse

# Try to import tqdm, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False



class PassageParser:
    '''
        Parse MSMARCO Passage colection.
    '''

    def __init__(self, input_file):
        self.input_file = input_file
        self.doc_id = 0
    

    def parser(self):
        '''
            Generator that yields (doc_id, text) tuples.
        '''

        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
            
                
                try:
                    # MS MARCO format: passage_id\tpassage_txt
                    parts = line.split('\t', 1)
                    if len(parts)==2:
                        passage_id, text = parts
                        yield(self.doc_id, passage_id, text)
                        self.doc_id += 1
                except Exception as e:
                    # Skip malformed lines
                    print(f"Warning: Skipping malformed lines: {e}", file=sys.stderr)
                    continue

class Tokenizer:
    '''
        Toekizer for processing words.
    '''

    @staticmethod
    def tokenize(text):

        '''
            Tokenize text into terms
            - Lowercase
            - Split on non-alphanumeric
            - Remove very short and very long tokens
        '''

        # Lowercase and split
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)

        # Filters tokens between 2-20
        tokens = [token for token in tokens if 2<=len(token)<=20]

        return tokens

class BlockedIndexer:

    '''
        Builds inverted index using blocked sort-based indexing (BSBI)
        Memory efficient for large collection
    '''

    def __init__(self, output_dir, max_postings_per_block=5_000_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preallocate fixed-size array for postings
        # Each posting: (term, doc_id, frequency)
        self.max_postings = max_postings_per_block
        self.postings_array = [None] * self.max_postings
        self.array_position = 0  # Current fill position
        self.block_num = 0

        # Document metadata: maps internal doc_id to (passage_id, length)
        self.doc_metadata = {}

        # Statistics
        self.total_docs = 0
        self.total_terms = 0


    def add_documents(self, doc_id, passage_id, text):
        '''
            Add a document to the current block.
        '''

        tokens = Tokenizer.tokenize(text)

        if  not tokens:
            return
        
        # Term frequencies in this document
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
        
        # Add to inverted index
        for term, freq in term_freqs.items():
            # Check if block is full
            if self.array_position >= self.max_postings:
                # Block full, write to disk
                self.write_block()
            
            self.postings_array[self.array_position] = (term, doc_id, freq)
            self.array_position += 1
        
        # Store document metadata
        self.doc_metadata[doc_id] = {
            'passage_id': passage_id,
            'length': len(tokens)
        }

        self.total_docs += 1
        
    def write_block(self):

        '''
           Write the current block to disk as a sorted itermediate file
        '''
        if self.array_position == 0:
            return  # Nothing to write
        
        block_file = self.output_dir / f"block_{self.block_num:04d}.txt"

        print(f" Writing block {self.block_num} with {self.array_position} postings....")

        # Get only the filled portion of the array
        filled_postings = self.postings_array[:self.array_position]
        
        # Sort by term, then by doc_id
        filled_postings.sort(key=lambda x: (x[0], x[1]))
        
        # Group postings by term and write to file
        with open(block_file, 'w', encoding='utf-8') as f:
            current_term = None
            postings_for_term = []
            
            for term, doc_id, freq in filled_postings:
                if term != current_term:
                    # Write previous term's postings
                    if current_term is not None:
                        postings_str = ','.join(f"{d}:{f}" for d, f in postings_for_term)
                        f.write(f"{current_term}\t{postings_str}\n")
                    
                    # Start new term
                    current_term = term
                    postings_for_term = [(doc_id, freq)]
                else:
                    # Same term, accumulate postings
                    postings_for_term.append((doc_id, freq))
            
            # Write last term
            if current_term is not None:
                postings_str = ','.join(f"{d}:{f}" for d, f in postings_for_term)
                f.write(f"{current_term}\t{postings_str}\n")
        
        # Count unique terms
        unique_terms = len(set(posting[0] for posting in filled_postings))
        self.total_terms += unique_terms
        
        # Reset for next block
        self.block_num += 1
        self.array_position = 0

    
    def write_metadata(self):
        """Write document metadata to disk"""
        metadata_file = self.output_dir / "doc_metadata.json"
        
        print(f"Writing metadata for {len(self.doc_metadata)} documents...")
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'doc_count': self.total_docs,
                'documents': self.doc_metadata
            }, f)


    def finalize(self):
        """Flush remaining data and write metadata"""
        # Write last block if not empty
        if self.array_position > 0:
            self.write_block()
        
        # Write document metadata (same as before)
        self.write_metadata()
        
        # Write indexing statistics
        stats_file = self.output_dir / "index_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Total documents: {self.total_docs}\n")
            f.write(f"Total unique terms: {self.total_terms}\n")
            f.write(f"Number of blocks: {self.block_num}\n")
            f.write(f"Max postings per block: {self.max_postings:,}\n")
        
        print(f"\nIndexing complete!")
        print(f"  Documents: {self.total_docs:,}")
        print(f"  Unique terms: {self.total_terms:,}")
        print(f"  Intermediate blocks: {self.block_num}")



def main():

    '''
        Main function for Indexer
    '''

    parser = argparse.ArgumentParser(
        description="Build inverted index from MS MARCO passages (Part 1: Indexer)"
    )

    parser.add_argument('input_file', help='Input MS MARCO TSV file')
    parser.add_argument('output_dir', help='Output directory for intermediate files')
    parser.add_argument('--max-postings', type=int, default=5_000_000,
                   help='Max postings per block (default: 5 million)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Starting indexer...")
    print(f"  Input: {args.input_file}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max postings per block: {args.max_postings:,}\n")

    # Initialize components
    passage_parser = PassageParser(args.input_file)
    indexer = BlockedIndexer(args.output_dir, max_postings_per_block=args.max_postings)

    # Process documents
    doc_count = 0
    parsed_passages = passage_parser.parser()

    if HAS_TQDM:
        # Use tqdm progress bar
        pbar = tqdm(parsed_passages, desc="Indexing", unit=" docs", unit_scale=True)
        for doc_id, passage_id, text in pbar:
            indexer.add_documents(doc_id, passage_id, text)
            doc_count += 1

            if doc_count % 10000 == 0:
                pbar.set_postfix({'blocks': indexer.block_num})
        pbar.close()
    else:
        # Fallback to simple progress
        for doc_id, passage_id, text in parsed_passages:
            indexer.add_documents(doc_id, passage_id, text)
            doc_count += 1
            if doc_count % 10000 == 0:
                print(f"Processed {doc_count:,} documents...", end='\r')

    print(f"\nProcessed {doc_count:,} documents total")

    # Finalize index
    indexer.finalize()


if __name__ == "__main__":
    main()