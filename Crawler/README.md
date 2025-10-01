# Multi-threaded Web Crawler

A straightforward Python web crawler built for a grad school assignment. Crawls web pages starting from DuckDuckGo search results and logs everything it finds.

## What it does

- Searches DuckDuckGo for your query to get starting URLs
- Crawls pages using multiple threads for speed
- Respects robots.txt (mostly)
- Prioritizes crawling across different domains instead of going deep into one site
- Logs everything to a file with stats at the end

## Setup

```bash
pip install requests beautifulsoup4 duckduckgo-search urllib3
```

## Usage

Basic:
```bash
python crawler.py --search "your search term"
```

With options:
```bash
python crawler.py --search "machine learning" --max_pages 5000 --threads 20 --seed_pages 15
```

### Parameters

- `--search` - What to search for (required)
- `--max_pages` - How many pages to crawl (default: 500)
- `--threads` - Number of threads (20 is the sweet spot, default: 10)
- `--seed_pages` - Starting URLs from search (default: 10, increase for niche topics)

## Output

Creates a timestamped log file with:
- URL, timestamp, size, depth, status code for each page
- Stats summary at the end (speed, domains, status codes, etc.)

## Notes

- Queue might run out before hitting max_pages if you're crawling something niche (increase seed_pages)
- Only crawls HTML pages, skips PDFs and images
- 5 second timeout per page
- Tested up to 10k pages without memory issues

## Assignment Context

Made for a web search engines course. Requirements included multi-threading, robots.txt compliance, and domain diversity. Added robots.txt caching because hitting it for every URL seemed wasteful.

## License

Do whatever you want with it.