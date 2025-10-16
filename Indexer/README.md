# MS MARCO Search Engine

A memory-efficient inverted index search engine built for a grad school assignment. Indexes 8.8M passages from MS MARCO dataset and returns BM25-ranked search results.

## What it does

- Builds inverted index using blocked sort-based indexing (BSBI) for memory efficiency
- Merges intermediate blocks into compressed final index using varbyte encoding
- Searches using BM25 ranking with both conjunctive (AND) and disjunctive (OR) query modes
- Traverses compressed inverted lists using DAAT (Document-At-A-Time) processing
- Provides browser-accessible Streamlit interface with snippet generation (extra credit)

## Architecture

The system consists of three main components:

1. **Indexer** ([indexer.py](indexer.py)) - Parses MS MARCO passages and creates sorted intermediate blocks
2. **Merger** ([merger.py](merger.py)) - Merges blocks into compressed final index with lexicon and document table
3. **Query Processor** ([query_processor.py](query_processor.py)) - Processes queries using BM25 scoring with DAAT traversal

## Setup

### Get the Dataset

Download and extract the MS MARCO passage collection:

```bash
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzf collection.tar.gz
mkdir -p data
mv collection.tsv data/
```

The dataset contains 8.8M passages in TSV format (passage_id\tpassage_text).

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Build Index

Run the indexer to create intermediate blocks:

```bash
python indexer.py data/collection.tsv index/intermediate
```

Options:
- `--max-postings` - Postings per block (default: 5,000,000)

### Step 2: Merge Index

Merge intermediate blocks into final compressed index:

```bash
python merger.py index/intermediate index/final_index
```

### Step 3: Search

**Command-line interface:**

```bash
python query_processor.py index/final_index data/collection.tsv
```

**Browser interface:**

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Optional: Enable Snippets

Build offset index for fast passage retrieval and snippet generation:

```bash
python build_offset_index.py data/collection.tsv --output offset_index.json
```

## Features

### Core Requirements
- Three separate executables (indexer, merger, query processor)
- Memory-efficient BSBI indexing for large datasets
- Varbyte compression for docIDs and frequencies
- Separate blocks for docIDs and frequencies (not interleaved)
- Binary compressed inverted lists stored on disk
- Lexicon and document table for metadata
- DAAT query processing with forward seeking in compressed lists
- BM25 ranking function
- Conjunctive (AND) and disjunctive (OR) query modes
- Inverted index API framework with open/close/seek operations

### Extra Credit
- Browser-accessible Streamlit web interface
- Query-dependent snippet generation with term highlighting
- Offset index for O(1) passage text retrieval

## Technical Details

### Compression
- **Varbyte encoding** for docIDs (delta-encoded) and frequencies
- **Separate storage** for docIDs and frequencies in each inverted list
- **Block format**: [header: doc_count, docid_length, freq_length][compressed_docids][compressed_freqs]

### Index Structure
- `lexicon.bin` - Maps terms to inverted list offsets and metadata
- `doc_table.bin` - Maps internal doc_ids to passage_ids and lengths
- `inverted_lists.bin` - Compressed inverted lists with varbyte encoding
- `offset_index.json` - Optional offset index for snippet generation

### Query Processing
- Lexicon loaded into memory at startup
- Inverted lists fetched on-demand from disk using seek operations
- DAAT traversal with min-heap for disjunctive queries
- Forward seeking with binary search for conjunctive queries
- Top-k results maintained using heap

## Output

**Indexing Phase:**
- Intermediate block files in text format
- Final compressed binary index files
- Metadata with statistics

**Query Results:**
- Ranked list of passages with BM25 scores
- Query-dependent snippets with highlighted terms (if offset index available)
- Passage IDs for each result

## Performance Notes

- Indexer processes ~10k-50k documents per second depending on hardware
- Block-based indexing keeps memory usage constant regardless of collection size
- Varbyte compression typically achieves 3-5x compression ratio
- Query processing returns results in under 1 second for typical queries
- Index size typically 10-15% of original collection size

## Assignment Context

Built for NYU Tandon CS 6913 (Web Search Engines) Fall 2025. Requirements included:
- Memory-efficient indexing for 8.8M passages
- Blocked, binary, compressed inverted index format
- DAAT query processing with BM25 ranking
- Inverted index API framework with seek operations
- Both conjunctive and disjunctive query modes
- Browser-accessible interface with Streamlit
- Snippet generation with term highlighting
- Offset index for fast passage retrieval

## License

Educational use only.
