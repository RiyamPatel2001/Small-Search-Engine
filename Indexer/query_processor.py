import math
import heapq
import re
import time
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from merger import Lexicon, DocumentTable, InvertedListReader


class SearchResult:
    """Represents a single search result."""
    
    def __init__(self, doc_id: int, score: float, passage_id: str, passage_text: str = ""):
        self.doc_id = doc_id
        self.score = score
        self.passage_id = passage_id
        self.passage_text = passage_text
    
    def __lt__(self, other):
        # For heapq - we want max heap, so reverse comparison
        return self.score < other.score
    
    def __repr__(self):
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, passage_id={self.passage_id})"


class BM25Scorer:

    """
        BM25 scoring implementation.
        Formula: score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))
    """

    def __init__(self, total_docs, avg_doc_length, k1=1.2, b=0.75):

        """
            Initialize BM25 scorer with collection statistics.
            
            Args:
                total_docs: Total number of documents in collection (N)
                avg_doc_length: Average document length in collection
                k1: BM25 k1 parameter (default 1.2)
                b: BM25 b parameter (default 0.75)
        """
        self.total_docs = total_docs
        self.avg_doc_length = avg_doc_length
        self.k1 = k1
        self.b = b
        self.idf_cache = {}  # Cache IDF values

    
    def compute_idf(self, doc_freq):

        """
            Compute IDF score for a term.
            IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
        """

        if doc_freq not in self.idf_cache:
            numerator = self.total_docs - doc_freq + 0.5
            denominator = doc_freq + 0.5
            self.idf_cache[doc_freq] = math.log(numerator/denominator)

        return self.idf_cache[doc_freq]

    def score_document(self, term_freqs, doc_freq_map, doc_length):
        """
            Calculate BM25 score for a document given query terms.
        """

        score = 0.0

        for term, freq in term_freqs.items():
            if term not in doc_freq_map:
                continue
                
            # Compute IDF
            idf = self.compute_idf(doc_freq_map[term])
            
            # Compute term score component
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * (numerator / denominator)

        return score


class QueryProcessor:

    def __init__(self, lexicon, doc_table, inverted_lists_path, passages_path):

        """
            Initialize query processor.
        """

        self.lexicon = lexicon
        self.doc_table = doc_table
        self.inverted_lists_path = inverted_lists_path
        self.passages_path = passages_path

        self.bm25 = BM25Scorer(
            total_docs = doc_table.num_docs,
            avg_doc_length = doc_table.avg_doc_length
        )

        # Cache for passage test
        self.passage_cache = {}
    
    
    def tokenize_query(self, query):

        '''
            Tokenize query using same rules as indexing.
            Lowercase, extract alphanumeric tokens of length 2-20.
        '''

        query = query.lower()
        tokens = re.findall(r'[a-z0-9]+', query)
        tokens = [t for t in tokens if 2 <= len(t) <= 20]

        return tokens
    
    def process_disjunctive_query(self, query_terms: List[str], k: int = 10) -> Tuple[List[SearchResult], float]:
        """
        Process disjunctive (OR) query using DAAT.
        Documents containing ANY query term are scored.

        Args:
            query_terms: List of query tokens
            k: Number of top results to return

        Returns:
            Tuple of (List of top-k SearchResult objects, query time in seconds)
        """
        start_time = time.time()

        # Remove duplicate terms and filter out terms not in lexicon
        unique_terms = []
        doc_freq_map = {}

        for term in query_terms:
            if term not in unique_terms and term in self.lexicon.entries:
                unique_terms.append(term)
                doc_freq_map[term] = self.lexicon.entries[term]['doc_count']

        if not unique_terms:
            return [], time.time() - start_time

        # Open inverted lists for all query terms
        readers = {}
        for term in unique_terms:
            reader = InvertedListReader(self.inverted_lists_path)
            if reader.open(term, self.lexicon):
                readers[term] = reader

        if not readers:
            return [], time.time() - start_time

        # DAAT processing with min-heap for next doc_id
        # Each entry: (current_doc_id, term)
        heap = []
        for term, reader in readers.items():
            doc_id, freq = reader.next_doc()
            if doc_id is not None:
                heapq.heappush(heap, (doc_id, term))

        # Top-k results (max heap by score)
        top_k = []

        # Process documents in order
        while heap:
            # Get the smallest doc_id
            current_doc_id, _ = heap[0]

            # Collect all terms that have this doc_id
            term_freqs = {}
            terms_to_advance = []

            while heap and heap[0][0] == current_doc_id:
                _, term = heapq.heappop(heap)
                freq = readers[term].get_frequency()
                term_freqs[term] = freq
                terms_to_advance.append(term)

            # Score this document
            doc_length = self.doc_table.get_doc_length(current_doc_id)
            score = self.bm25.score_document(term_freqs, doc_freq_map, doc_length)

            # Add to top-k results
            passage_id = self.doc_table.get_passage_id(current_doc_id)
            result = SearchResult(current_doc_id, score, passage_id)

            if len(top_k) < k:
                heapq.heappush(top_k, result)
            elif score > top_k[0].score:
                heapq.heapreplace(top_k, result)

            # Advance readers for terms that matched this doc
            for term in terms_to_advance:
                doc_id, freq = readers[term].next_doc()
                if doc_id is not None:
                    heapq.heappush(heap, (doc_id, term))

        # Close all readers
        for reader in readers.values():
            reader.close()

        # Return results sorted by score (descending)
        results = sorted(top_k, key=lambda x: x.score, reverse=True)
        query_time = time.time() - start_time
        return results, query_time
    
    def process_conjunctive_query(self, query_terms: List[str], k: int = 10) -> Tuple[List[SearchResult], float]:
        """
        Process conjunctive (AND) query using DAAT.
        Only documents containing ALL query terms are scored.

        Args:
            query_terms: List of query tokens
            k: Number of top results to return

        Returns:
            Tuple of (List of top-k SearchResult objects, query time in seconds)
        """
        start_time = time.time()

        # Remove duplicates and filter terms not in lexicon
        unique_terms = []
        doc_freq_map = {}

        for term in query_terms:
            if term not in unique_terms and term in self.lexicon.entries:
                unique_terms.append(term)
                doc_freq_map[term] = self.lexicon.entries[term]['doc_count']

        if not unique_terms:
            return [], time.time() - start_time

        # Open inverted lists for all query terms
        readers = {}
        for term in unique_terms:
            reader = InvertedListReader(self.inverted_lists_path)
            if reader.open(term, self.lexicon):
                readers[term] = reader

        if len(readers) != len(unique_terms):
            # Some terms not found
            for reader in readers.values():
                reader.close()
            return [], time.time() - start_time

        # Sort readers by list length (shortest first for efficiency)
        sorted_terms = sorted(unique_terms, key=lambda t: self.lexicon.entries[t]['doc_count'])

        # Top-k results
        top_k = []

        # Get first doc from shortest list
        shortest_term = sorted_terms[0]
        doc_id, freq = readers[shortest_term].next_doc()

        while doc_id is not None:
            # Check if all other lists contain this doc_id
            term_freqs = {shortest_term: freq}
            all_match = True

            for term in sorted_terms[1:]:
                # Seek to current doc_id in this list
                found_doc_id, found_freq = readers[term].seek(doc_id)

                if found_doc_id == doc_id:
                    term_freqs[term] = found_freq
                else:
                    # This document doesn't contain all terms
                    all_match = False
                    break

            if all_match:
                # Score this document
                doc_length = self.doc_table.get_doc_length(doc_id)
                score = self.bm25.score_document(term_freqs, doc_freq_map, doc_length)

                passage_id = self.doc_table.get_passage_id(doc_id)
                result = SearchResult(doc_id, score, passage_id)

                if len(top_k) < k:
                    heapq.heappush(top_k, result)
                elif score > top_k[0].score:
                    heapq.heapreplace(top_k, result)

            # Advance shortest list
            doc_id, freq = readers[shortest_term].next_doc()

        # Close all readers
        for reader in readers.values():
            reader.close()

        # Return results sorted by score (descending)
        results = sorted(top_k, key=lambda x: x.score, reverse=True)
        query_time = time.time() - start_time
        return results, query_time
    

    def load_passage_text(self, passage_id):
        """
            Load passage text from passages file.

            NOTE: This is a simplified implementation that only returns the passage_id.
            Loading actual text would require either:
            1. Building an offset index during indexing (recommended for production)
            2. Loading entire file into memory (not feasible for 8.8M passages)
            3. Sequential scan (extremely slow - O(n) per query result)

            For the assignment demo, returning passage_id is acceptable.
        """
        # Just return the passage ID - actual text loading is too slow
        return f"Passage ID: {passage_id}"

    
    def format_result(self, results):

        '''
            Format search results for display.
        '''

        if not results:
            return "No Results"
        
        output = [f"\nTop {len(results)} results:\n" + "="*60]

        for i, result in enumerate(results, 1):
            # Load passage text if available
            passage_text = self.load_passage_text(result.passage_id)

            # Truncate long passages for display
            if len(passage_text) > 200:
                passage_text = passage_text[:200] + "..."

            output.append(f"\n{i}. [Doc {result.doc_id}] [Score: {result.score:.4f}]")
            output.append(f"   Passage ID: {result.passage_id}")
            output.append(f"   {passage_text}")

        return "\n".join(output)

def run_query_interface(processor: QueryProcessor):
    """
    Run interactive command-line query interface.
    
    Args:
        processor: Initialized QueryProcessor
    """
    print("="*60)
    print("Search Engine Query Processor")
    print("="*60)
    print("Commands:")
    print("  - Enter a query to search")
    print("  - Type 'exit' or 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            # Get query
            query = input("\n> ").strip()
            k = 4

            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Get number of results first (outside validation loop)
            k_input = input(f"Number of top results to return [default: {k}]: ").strip()
            if k_input:
                try:
                    k = int(k_input)
                    if k <= 0:
                        print(f"Invalid number, using default: {k}")
                        k = 4
                except ValueError:
                    print(f"Invalid number, using default: {k}")
                    k = 4

            # Get query mode
            while True:
                mode = input("Mode (conjunctive/disjunctive) [default: disjunctive]: ").strip().lower()

                if not mode:
                    mode = 'disjunctive'
                    break
                if mode in ['conjunctive', 'disjunctive', 'c', 'd']:
                    if mode == 'c':
                        mode = 'conjunctive'
                    elif mode == 'd':
                        mode = 'disjunctive'
                    break
                print("Invalid mode. Please enter 'conjunctive' or 'disjunctive'.")
            
            # Tokenize query
            query_terms = processor.tokenize_query(query)
            
            if not query_terms:
                print("No valid query terms found.")
                continue
            
            print(f"\nQuery terms: {query_terms}")
            print(f"Mode: {mode}")
            print("Searching...")

            # Process query
            if mode == 'disjunctive':
                results, query_time = processor.process_disjunctive_query(query_terms, k)
            else:
                results, query_time = processor.process_conjunctive_query(query_terms, k)

            # Display timing information (Google-style)
            print(f"\nRetrieved {len(results):,} documents from a total of {processor.doc_table.num_docs:,} in {query_time:.3f} seconds")

            # Display results
            print(processor.format_result(results))
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Query processor for MS MARCO search engine (Part 3)"
    )
    parser.add_argument('index_dir',
                       help='Directory containing the final index files (lexicon.bin, doc_table.bin, inverted_lists.bin)')
    parser.add_argument('passages_file',
                       help='Original passages TSV file (for passage text lookup - optional feature)')

    args = parser.parse_args()

    # Validate files exist
    lexicon_path = os.path.join(args.index_dir, 'lexicon.bin')
    doc_table_path = os.path.join(args.index_dir, 'doc_table.bin')
    inverted_lists_path = os.path.join(args.index_dir, 'inverted_lists.bin')

    missing_files = []
    if not os.path.exists(lexicon_path):
        missing_files.append(lexicon_path)
    if not os.path.exists(doc_table_path):
        missing_files.append(doc_table_path)
    if not os.path.exists(inverted_lists_path):
        missing_files.append(inverted_lists_path)

    if missing_files:
        print("Error: Required index files not found:", file=sys.stderr)
        for f in missing_files:
            print(f"  - {f}", file=sys.stderr)
        print("\nMake sure you've run merger.py first!", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.passages_file):
        print(f"Warning: Passages file not found: {args.passages_file}", file=sys.stderr)
        print("Passage text will not be displayed (only passage IDs)", file=sys.stderr)

    # Load index
    print("Loading index...")
    try:
        lexicon = Lexicon.load(lexicon_path)
        doc_table = DocumentTable.load(doc_table_path)
        print(f"✓ Loaded {len(lexicon.entries):,} terms")
        print(f"✓ Loaded {doc_table.num_docs:,} documents")
        print(f"✓ Average document length: {doc_table.avg_doc_length:.1f} tokens")
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        sys.exit(1)

    processor = QueryProcessor(lexicon, doc_table, inverted_lists_path, args.passages_file)

    # Run interactive interface
    run_query_interface(processor)
