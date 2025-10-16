SUBMITTED FILES
---------------
1. crawler.py - Main Python source code for the multi-threaded web crawler
2. readme.txt - This file
3. explain.txt - Technical explanation of program design
4. Log Files <search_term>_<timestamp>.text
4.1 human_2025-10-01_14-06-59.txt (10000 pages)
4.2 movies_2025-10-01_13-07-34.txt (10000 pages)

REQUIREMENTS
------------
- Python 3.7 or higher
- Required libraries (install via pip):
  * requests
  * beautifulsoup4
  * ddgs (metasearch library)
  * urllib3


INSTALLATION
------------
Install required dependencies:
    pip install requests beautifulsoup4 ddgs urllib3


HOW TO RUN
----------
Basic usage:
    python crawler.py --search "your search term"

With custom parameters:
    python crawler.py --search "your search term" --max_pages 5000 --threads 10 --seed_pages 10


COMMAND LINE PARAMETERS
-----------------------
--search <term>        (REQUIRED) Search term to find initial seed pages
--max_pages <n>        Maximum number of pages to crawl (default: 500)
--threads <n>          Number of concurrent worker threads (default: 10)
--seed_pages <n>       Number of seed URLs to fetch via DDGS metasearch (default: 10)


OUTPUT
------
The crawler creates a timestamped log file in the current directory with the format:
    <search_term>_YYYY-MM-DD_HH-MM-SS.txt

The log file contains:
- Header row with column names
- One line per crawled URL with: URL, Timestamp, Size (bytes), Depth, Status Code, Content Type
- Final statistics section appended at the end


PARAMETER LIMITATIONS
---------------------
- max_pages: No hard limit. Constrained only by available time and storage. Testing showed memory limits are difficult to reach for typical crawling tasks, so max_pages can be set based on desired crawl size and available time.

- threads: Recommended range 20-30 for best performance (based on testing, 20 threads optimal). Too few threads reduce crawling speed; too many may cause rate limiting or connection issues with target servers.

- seed_pages: Minimum 10 recommended. For niche search terms, increase seed_pages (15-20) as many sites block crawling via robots.txt, which can cause the queue to become empty before reaching max_pages.

- The crawler respects robots.txt rules (except for seed URLs) as per assignment requirements.
- Only crawls HTML pages; skips binary files (PDF, images, videos, etc.) as per assignment requirements.
- Network timeout is set to 5 seconds per page.