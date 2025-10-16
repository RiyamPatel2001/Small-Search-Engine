"""
Snippet Generation for Search Results
Extracts and highlights relevant passages containing query terms.
"""

import re


class SnippetGenerator:
    """
    Generates query-dependent snippets from passage text.
    """

    def __init__(self, max_snippet_length=200, context_chars=80):
        """
        Args:
            max_snippet_length: Maximum snippet length in characters
            context_chars: Characters to show before/after query term
        """
        self.max_snippet_length = max_snippet_length
        self.context_chars = context_chars

    def generate_snippet(self, text, query_terms, highlight=True):
        """
        Generate snippet from text containing query terms.

        Args:
            text: Full passage text
            query_terms: List of query terms (already tokenized/lowercased)
            highlight: Whether to add HTML highlighting

        Returns:
            Snippet string with optional HTML highlighting
        """
        if not text or not query_terms:
            return self._truncate(text)

        # Find all query term positions
        term_positions = self._find_term_positions(text, query_terms)

        if not term_positions:
            # No query terms found, return beginning
            return self._truncate(text)

        # Select best snippet window
        snippet_start, snippet_end = self._select_snippet_window(
            text, term_positions
        )

        # Extract snippet
        snippet = text[snippet_start:snippet_end]

        # Add ellipsis if truncated
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(text):
            snippet = snippet + "..."

        # Highlight query terms
        if highlight:
            snippet = self._highlight_terms(snippet, query_terms)

        return snippet

    def _find_term_positions(self, text, query_terms):
        """
        Find all positions of query terms in text.

        Returns:
            List of (start, end, term) tuples
        """
        text_lower = text.lower()
        positions = []

        for term in query_terms:
            # Find all occurrences of this term
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break

                # Check if it's a word boundary (not part of larger word)
                if self._is_word_boundary(text_lower, pos, len(term)):
                    positions.append((pos, pos + len(term), term))

                start = pos + 1

        # Sort by position
        positions.sort(key=lambda x: x[0])
        return positions

    def _is_word_boundary(self, text, start, length):
        """Check if match is at word boundary."""
        # Check before
        if start > 0 and text[start - 1].isalnum():
            return False

        # Check after
        end = start + length
        if end < len(text) and text[end].isalnum():
            return False

        return True

    def _select_snippet_window(self, text, term_positions):
        """
        Select best window to show query terms.

        Strategy: Find window that contains most query terms.
        """
        if not term_positions:
            return 0, min(self.max_snippet_length, len(text))

        # Try to center around first query term
        first_pos = term_positions[0][0]

        # Calculate window
        start = max(0, first_pos - self.context_chars)
        end = min(len(text), start + self.max_snippet_length)

        # Adjust to word boundaries if possible
        start = self._find_word_boundary_left(text, start)
        end = self._find_word_boundary_right(text, end)

        return start, end

    def _find_word_boundary_left(self, text, pos):
        """Find nearest word boundary to the left."""
        if pos == 0:
            return 0

        # Look for space or punctuation
        while pos > 0 and text[pos - 1].isalnum():
            pos -= 1

        return pos

    def _find_word_boundary_right(self, text, pos):
        """Find nearest word boundary to the right."""
        if pos >= len(text):
            return len(text)

        # Look for space or punctuation
        while pos < len(text) and text[pos].isalnum():
            pos += 1

        return pos

    def _truncate(self, text):
        """Truncate text to max length."""
        if not text:
            return ""

        if len(text) <= self.max_snippet_length:
            return text

        # Truncate and add ellipsis
        truncated = text[:self.max_snippet_length]
        # Try to end at word boundary
        last_space = truncated.rfind(' ')
        if last_space > self.max_snippet_length * 0.8:
            truncated = truncated[:last_space]

        return truncated + "..."

    def _highlight_terms(self, snippet, query_terms):
        """
        Add HTML highlighting to query terms in snippet.

        Returns snippet with <b> tags around query terms.
        """
        # Build regex pattern for all query terms
        # Escape special regex characters
        escaped_terms = [re.escape(term) for term in query_terms]
        pattern = '|'.join(escaped_terms)

        # Case-insensitive replacement with word boundaries
        def replace_func(match):
            return f"<b>{match.group(0)}</b>"

        # Use word boundary regex
        highlighted = re.sub(
            f'\\b({pattern})\\b',
            replace_func,
            snippet,
            flags=re.IGNORECASE
        )

        return highlighted


# Example usage
if __name__ == "__main__":
    generator = SnippetGenerator(max_snippet_length=200)

    # Test text
    text = ("The Manhattan Project was a research and development program "
            "undertaken during World War II that produced the first nuclear weapons. "
            "It was led by the United States with the support of the United Kingdom "
            "and Canada. The project began in 1939 and involved scientists from "
            "around the world working on the atomic bomb.")

    # Test queries
    query_terms = ["manhattan", "project"]

    snippet = generator.generate_snippet(text, query_terms, highlight=True)
    print("Original text:")
    print(text)
    print("\nSnippet with query terms ['manhattan', 'project']:")
    print(snippet)
    print()

    # Test another query
    query_terms2 = ["nuclear", "weapons"]
    snippet2 = generator.generate_snippet(text, query_terms2, highlight=True)
    print("Snippet with query terms ['nuclear', 'weapons']:")
    print(snippet2)
