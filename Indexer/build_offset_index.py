"""
Build Offset Index for Fast Passage Retrieval
Creates a mapping: passage_id -> byte offset in collection.tsv

This allows O(1) random access to any passage without scanning the entire file.
"""

import sys
import json
import argparse
from pathlib import Path


def build_offset_index(collection_file, output_file):
    """
    Build offset index for fast passage text retrieval.

    Args:
        collection_file: Path to collection.tsv
        output_file: Path to save offset_index.json

    Returns:
        Dictionary mapping passage_id -> byte_offset
    """
    print(f"Building offset index for {collection_file}...")

    offset_index = {}

    with open(collection_file, 'rb') as f:
        current_offset = 0
        line_count = 0

        while True:
            # Remember current position
            line_start = current_offset

            # Read line
            line = f.readline()
            if not line:
                break

            # Parse passage_id
            try:
                line_str = line.decode('utf-8', errors='ignore')
                passage_id = line_str.split('\t', 1)[0]

                # Store offset
                offset_index[passage_id] = line_start

                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} passages...", end='\r')

            except Exception as e:
                print(f"\nWarning: Skipping malformed line at offset {line_start}: {e}")

            # Update offset for next line
            current_offset += len(line)

    print(f"\n✓ Built offset index for {len(offset_index):,} passages")

    # Save to JSON
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(offset_index, f)

    # Calculate size
    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"✓ Offset index saved ({size_mb:.2f} MB)")

    return offset_index


def load_offset_index(index_file):
    """Load offset index from JSON file."""
    with open(index_file, 'r') as f:
        return json.load(f)


def get_passage_text(passage_id, offset_index, collection_file):
    """
    Retrieve passage text using offset index.

    Args:
        passage_id: Passage ID to retrieve
        offset_index: Dictionary from build_offset_index()
        collection_file: Path to collection.tsv

    Returns:
        Tuple of (passage_id, passage_text) or (None, None) if not found
    """
    if passage_id not in offset_index:
        return None, None

    offset = offset_index[passage_id]

    with open(collection_file, 'rb') as f:
        f.seek(offset)
        line = f.readline().decode('utf-8', errors='ignore')

        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            pid, text = parts
            return pid, text

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Build offset index for fast passage retrieval"
    )
    parser.add_argument('collection_file',
                       help='Path to collection.tsv')
    parser.add_argument('--output', default='offset_index.json',
                       help='Output file for offset index (default: offset_index.json)')

    args = parser.parse_args()

    # Validate input
    if not Path(args.collection_file).exists():
        print(f"Error: File not found: {args.collection_file}", file=sys.stderr)
        sys.exit(1)

    # Build index
    build_offset_index(args.collection_file, args.output)

    print("\n" + "="*70)
    print("Offset index ready!")
    print("="*70)
    print(f"Collection file:  {args.collection_file}")
    print(f"Offset index:     {args.output}")
    print("\nYou can now use this for fast passage text retrieval.")
    print("="*70)


if __name__ == "__main__":
    main()
