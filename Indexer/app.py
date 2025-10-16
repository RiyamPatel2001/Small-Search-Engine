"""
Streamlit Web Interface for MS MARCO Search Engine
Extra Credit: Browser-accessible interface with snippet generation
"""

import streamlit as st
import sys
import json
from pathlib import Path

from merger import Lexicon, DocumentTable, InvertedListReader
from query_processor import QueryProcessor, BM25Scorer
from build_offset_index import load_offset_index, get_passage_text
from snippet_generator import SnippetGenerator


# Page config
st.set_page_config(
    page_title="MS MARCO Search Engine",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_search_engine(index_dir, collection_file, offset_index_file):
    """Load search engine components (cached)."""
    try:
        # Load index
        lexicon_path = Path(index_dir) / 'lexicon.bin'
        doc_table_path = Path(index_dir) / 'doc_table.bin'
        inverted_lists_path = Path(index_dir) / 'inverted_lists.bin'

        lexicon = Lexicon.load(str(lexicon_path))
        doc_table = DocumentTable.load(str(doc_table_path))

        # Create query processor
        processor = QueryProcessor(
            lexicon,
            doc_table,
            str(inverted_lists_path),
            collection_file
        )

        # Load offset index if exists
        offset_index = None
        if Path(offset_index_file).exists():
            offset_index = load_offset_index(offset_index_file)

        return processor, offset_index, lexicon, doc_table

    except Exception as e:
        st.error(f"Error loading search engine: {e}")
        return None, None, None, None


def main():
    # Title
    st.title("🔍 MS MARCO Search Engine")
    st.markdown("**Assignment 2 - Web Search Engines**")
    st.markdown("---")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        index_dir = st.text_input(
            "Index Directory",
            value="index/final_index",
            help="Directory containing lexicon.bin, doc_table.bin, inverted_lists.bin"
        )

        collection_file = st.text_input(
            "Collection File",
            value="data/collection.tsv",
            help="Original MS MARCO collection.tsv file"
        )

        offset_index_file = st.text_input(
            "Offset Index File",
            value="offset_index.json",
            help="Offset index for fast passage retrieval (optional)"
        )

        st.markdown("---")

        # Load search engine
        if st.button("🔄 Load/Reload Index"):
            st.cache_resource.clear()

        processor, offset_index, lexicon, doc_table = load_search_engine(
            index_dir, collection_file, offset_index_file
        )

        # Show index stats
        if processor:
            st.success("✓ Index loaded!")
            st.metric("Terms", f"{len(lexicon.entries):,}")
            st.metric("Documents", f"{doc_table.num_docs:,}")
            st.metric("Avg Doc Length", f"{doc_table.avg_doc_length:.1f}")

            if offset_index:
                st.info(f"✓ Offset index loaded ({len(offset_index):,} passages)")
            else:
                st.warning("⚠️ No offset index (snippets disabled)")
        else:
            st.error("❌ Index not loaded")

    # Main content
    if not processor:
        st.warning("👈 Please configure and load the index from the sidebar")
        st.markdown("""
        ### Quick Start:
        1. Make sure you've run the indexer and merger
        2. Check that the paths in the sidebar are correct
        3. Click "Load/Reload Index"

        ### Optional: Build Offset Index for Snippets
        Run this command to enable snippet generation:
        ```bash
        python build_offset_index.py data/collection.tsv --output offset_index.json
        ```
        """)
        return

    # Search interface
    st.header("Search")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter your query",
            placeholder="e.g., manhattan project, machine learning, climate change",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Query options
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio(
            "Query Mode",
            ["disjunctive", "conjunctive"],
            horizontal=True,
            help="Disjunctive (OR): matches ANY query term | Conjunctive (AND): matches ALL query terms"
        )

    with col2:
        k = st.slider("Number of results", min_value=1, max_value=50, value=10)

    st.markdown("---")

    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            # Tokenize query
            query_terms = processor.tokenize_query(query)

            if not query_terms:
                st.warning("No valid query terms found. Try a different query.")
                return

            st.info(f"**Query terms:** {', '.join(query_terms)}")

            # Process query
            if mode == "disjunctive":
                results, query_time = processor.process_disjunctive_query(query_terms, k)
            else:
                results, query_time = processor.process_conjunctive_query(query_terms, k)

            # Display results
            if not results:
                st.warning("No results found.")
                return

            st.success(f"Retrieved **{len(results):,}** documents from a total of **{doc_table.num_docs:,}** in **{query_time:.3f}** seconds")

            # Initialize snippet generator if offset index available
            snippet_gen = None
            if offset_index:
                snippet_gen = SnippetGenerator(max_snippet_length=250)

            # Display each result
            for i, result in enumerate(results, 1):
                with st.container():
                    # Header
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {i}. Passage {result.passage_id}")
                    with col2:
                        st.metric("BM25 Score", f"{result.score:.4f}")

                    # Get passage text and generate snippet
                    if snippet_gen and offset_index:
                        pid, text = get_passage_text(
                            result.passage_id,
                            offset_index,
                            collection_file
                        )

                        if text:
                            # Generate snippet
                            snippet = snippet_gen.generate_snippet(
                                text,
                                query_terms,
                                highlight=True
                            )

                            # Display snippet with highlighting
                            st.markdown(snippet, unsafe_allow_html=True)

                            # Expandable full text
                            with st.expander("📄 View full passage"):
                                st.write(text)
                        else:
                            st.text(f"Passage ID: {result.passage_id}")
                    else:
                        st.text(f"Passage ID: {result.passage_id}")
                        st.caption("(Build offset index to see snippets)")

                    st.markdown("---")

    elif query and not search_button:
        st.info("👆 Press Enter or click Search button")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>MS MARCO Search Engine | NYU Tandon CS 6913 | Fall 2025</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
