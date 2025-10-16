"""
Multi-threaded web crawler that discovers and crawls web pages based on a search query.
Uses DDGS metasearch library to find seed URLs, then crawls pages while respecting robots.txt rules.
Implements a priority queue system to diversify crawling across multiple domains.
"""

import argparse
import os
import queue
import re
import threading
import time
import urllib.parse
import urllib.robotparser
from collections import defaultdict
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WebCrawler:

    def __init__(self, search_term, max_pages=1000, num_threads=10, seed_pages=10, delay=0.1): 
        self.max_pages = max_pages
        self.num_threads = num_threads
        self.seed_pages = seed_pages
        self.search_term = search_term
        self.delay = delay
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{self.search_term.replace(' ', '_')}_{timestamp}.txt"
        self.log_file = os.path.join(os.getcwd(), file_name)
        
        self.url_que = queue.PriorityQueue()
        self.visited_urls = set()
        self.domain_counts = defaultdict(int)
        self.seed_urls = set()
        self.lock = threading.Lock()
        self.pages_crawled = 0
        self.stop_crawling = False
        self.robots_cache = {}
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent':'NYU-Web Crawler/1.0 (Educational Purpose)'
        })
        
        adapter = HTTPAdapter(
            pool_connections=50,
            pool_maxsize=50,
            max_retries=Retry(total=1, backoff_factor=0.1),
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.start_time = None
        self.pbar = None
        self.crawl_stats = {
            'total_size': 0,
            'status_codes': defaultdict(int),
            'depths': defaultdict(int)
        }

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("URL\tTimestamp\tSize_Bytes\tDepth\tStatus_Code\tContent_Type\n")

    def log_debug(self, message):
        """Always print debug messages to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

    def output_statistics(self):
        """Output final crawling statistics to console and append to log file"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        stats_output = []
        stats_output.append("="*60)
        stats_output.append("FINAL CRAWLING STATISTICS")
        stats_output.append("="*60)
        stats_output.append(f"Total pages crawled: {self.pages_crawled}")
        stats_output.append(f"Total time taken: {total_time:.2f} seconds")

        if total_time > 0 and self.pages_crawled > 0:
            stats_output.append(f"Average speed: {self.pages_crawled/total_time:.2f} pages/second")
            stats_output.append(f"Average page size: {self.crawl_stats['total_size']/self.pages_crawled:.0f} bytes")
        
        stats_output.append(f"Total data downloaded: {self.crawl_stats['total_size']:,} bytes")
        
        stats_output.append(f"\nStatus Code Distribution:")
        status_items = [(str(code), count) for code, count in self.crawl_stats['status_codes'].items()]
        for code, count in sorted(status_items):
            stats_output.append(f"  {code}: {count}")
        
        stats_output.append(f"\nDepth Distribution:")
        for depth, count in sorted(self.crawl_stats['depths'].items()):
            stats_output.append(f"  Depth {depth}: {count} pages")

        stats_output.append("\n" + "=" * 60)
        stats_output.append("CRAWLING COMPLETE - FINAL REPORT")
        stats_output.append("=" * 60)
        stats_output.append(f"Pages crawled: {self.pages_crawled}")
        stats_output.append(f"Time taken: {total_time:.2f} seconds")
        if total_time > 0 and self.pages_crawled > 0:
            stats_output.append(f"Speed: {self.pages_crawled/total_time:.2f} pages/second")
        stats_output.append(f"Unique domains discovered: {len(self.domain_counts)}")
        
        stats_output.append(f"\nDOMAIN DISTRIBUTION (Top 10):")
        sorted_domains = sorted(self.domain_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (domain, count) in enumerate(sorted_domains[:10], 1):
            stats_output.append(f"{i:2d}. {domain:30} - {count:3d} pages")
        
        if len(sorted_domains) > 10:
            stats_output.append(f"... and {len(sorted_domains) - 10} more domains")

        final_stats_str = "\n".join(stats_output)
        
        print("\n" + final_stats_str)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n\n# FINAL STATISTICS\n")
            f.write(final_stats_str + "\n")

    def get_seed_pages(self):
        """Search using DDGS metasearch to get seed URLs."""
        self.log_debug(f"Searching via DDGS for: '{self.search_term}'")
        urls = []
        try:
            with DDGS() as ddgs:
                results_generator = ddgs.text(self.search_term, max_results=self.seed_pages)
                for r in results_generator:
                    urls.append(r['href'])
        except Exception as e:
            self.log_debug(f"An error occurred during DDGS search: {e}")

        self.log_debug(f"DDGS metasearch returned {len(urls)} URLs")

        self.log_debug("Seed URLs:")
        for i, url in enumerate(urls, 1):
            self.log_debug(f"{i}, {url}")
            self.seed_urls.add(url)
            priority = self.calculate_priority_score(url)
            self.url_que.put((priority, url, 0))
        return urls

    def get_domain(self, url):
        try:
            return urllib.parse.urlparse(url).netloc
        except:
            return None
    
    def calculate_priority_score(self, url):
        """
        Calculate priority score to promote domain diversity.
        Lower scores = higher priority. Domains with fewer pages get higher priority.
        """
        domain = self.get_domain(url)
        domain_counts = self.domain_counts[domain]
        if domain_counts==0: return 1
        elif domain_counts<3: return 2
        elif domain_counts<6: return 3
        else: return 10

    def can_crawl(self, url):
        """Check robots.txt to determine if URL can be crawled. Seed URLs bypass this check."""
        if url in self.seed_urls:
            return True
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            if base_url in self.robots_cache:
                rp = self.robots_cache[base_url]
                if rp is None: return True
                return rp.can_fetch('Educational-WebCrawler/1.0', url)
            
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(base_url, '/robots.txt')
            rp.set_url(robots_url)
            
            try:
                import socket
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(0.5)
                rp.read()
                socket.setdefaulttimeout(old_timeout)
            except:
                rp = None
            
            self.robots_cache[base_url] = rp
            if rp is None: return True
            return rp.can_fetch('Educational-WebCrawler/1.0', url)
        except:
            return True

    def is_valid_url(self, url):
        """Check whether a URL is valid to crawl."""
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ['http', 'https']: return False
            if not parsed.netloc: return False
            if url.startswith(('javascript:', 'mailto:', '#', 'tel:', 'ftp:')): return False
            
            skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', 
                               '.doc', '.docx', '.mp3', '.mp4', '.exe'}
            path = parsed.path.lower()
            if '.' in path:
                ext = '.' + path.split('.')[-1]
                if ext in skip_extensions: return False
            return True
        except:
            return False
    
    def extract_links(self, html_content, base_url):
        """Find all links in HTML content."""
        links = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for link_tag in soup.find_all('a', href=True):
                href = link_tag['href']
                full_url = urljoin(base_url, href)
                if self.is_valid_url(full_url):
                    links.append(full_url)
            return links
        except Exception as e:
            self.log_debug(f"Error extracting links from {base_url}: {e}")
            return []

    def crawl_single_page(self, url, page_number, depth):
        """Crawl one page and return the links found."""
        try:
            if not self.can_crawl(url):
                self.log_debug(f"Robots.txt disallows: {url}")
                with self.lock:
                    self.crawl_stats['status_codes']['ROBOTS_BLOCKED'] += 1
                return []
        
            self.log_debug(f"Crawling: {url}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            content_length = len(response.content)
            content_type = response.headers.get('content-type', '').lower()

            with self.lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{url}\t{timestamp}\t{content_length}\t{depth}\t{response.status_code}\t{content_type}\n")
                self.crawl_stats['total_size'] += content_length
                self.crawl_stats['status_codes'][response.status_code] += 1
            
            if 'text/html' not in content_type:
                return []
            
            links = self.extract_links(response.text, url)
            self.log_debug(f"Found {len(links)} links on {url} (Page {page_number}/{self.max_pages})")
            return links
            
        except Exception as e:
            self.log_debug(f"Failed to crawl {url}: {e}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{url}\t{timestamp}\t0\t{depth}\tERROR\terror\n")
                self.crawl_stats['status_codes']['ERROR'] += 1 
            return []

    def worker_thread(self, thread_id):
        """
        Worker thread that continuously pulls URLs from the queue and crawls them.
        Stops when max_pages is reached or "STOP" signal is received.
        """
        self.log_debug(f"Worker thread {thread_id} started")
        while not self.stop_crawling:
            try:
                priority, url, depth = self.url_que.get(timeout=1)

                if url == "STOP":
                    self.url_que.task_done()
                    break
                
                with self.lock:
                    if self.pages_crawled >= self.max_pages:
                        if not self.stop_crawling:
                            self.log_debug(f"Thread {thread_id}: Max pages reached, sending stop signals.")
                            self.stop_crawling = True
                            for _ in range(self.num_threads):
                                self.url_que.put((0, "STOP", 0))
                        self.url_que.task_done()
                        break
                    
                    if url in self.visited_urls:
                        self.url_que.task_done()
                        continue
                    
                    self.pages_crawled += 1
                    if self.pbar: self.pbar.update(1) 
                    current_count = self.pages_crawled
                    self.visited_urls.add(url)
                    self.crawl_stats['depths'][depth] += 1
                
                new_links = self.crawl_single_page(url, current_count, depth)
                
                with self.lock:
                    self.domain_counts[self.get_domain(url)] += 1

                links_added = 0
                if not self.stop_crawling and current_count < self.max_pages:
                    for link in new_links:
                        with self.lock:
                            if link not in self.visited_urls:
                                priority = self.calculate_priority_score(link)
                                self.url_que.put((priority, link, depth + 1))
                                links_added += 1
                
                if links_added > 0:
                    self.log_debug(f"Thread {thread_id}: Added {links_added} new URLs to queue")
                
                self.url_que.task_done()
                
            except queue.Empty:
                self.log_debug(f"Thread {thread_id}: Queue empty. Exiting.")
                break
            except Exception as e:
                self.log_debug(f"Thread {thread_id}: Error - {e}")
                if self.url_que.unfinished_tasks > 0:
                    self.url_que.task_done()

        self.log_debug(f"Worker thread {thread_id} finished")
    
    def start_crawling(self):
        self.start_time = time.time()
        print("="*50)
        print(f"Search Query: {self.search_term}")
        print(f"Target Pages: {self.max_pages}")
        print(f"Seed Pages: {self.seed_pages}")
        print(f"Threads: {self.num_threads}")
        print("="*50)
        
        self.get_seed_pages()

        # Start worker threads
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker_thread, args=(i+1, ))
            thread.daemon = False
            thread.start()
            threads.append(thread)
        
        try:
            while not self.stop_crawling and self.pages_crawled < self.max_pages:
                time.sleep(1)
                if all(not t.is_alive() for t in threads):
                    self.log_debug("All threads have unexpectedly finished.")
                    break
            
            # Send stop signals to all threads
            self.stop_crawling = True
            for _ in range(self.num_threads):
                self.url_que.put((0, "STOP", 0))

            self.log_debug("\nWaiting for all threads to finish...")
            for thread in threads:
                thread.join(timeout=3)
                if thread.is_alive():
                    self.log_debug(f"Thread {thread.name} still alive, forcing shutdown...")

        except KeyboardInterrupt:
            print("\nCrawling interrupted by user")
            self.stop_crawling = True
            for _ in range(self.num_threads):
                self.url_que.put((0, "STOP", 0))
        
        if self.pbar: self.pbar.close() 
        self.output_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A multi-threaded web crawler.")
    parser.add_argument('--search', type=str, required=True, help='The search term to find seed pages.')
    parser.add_argument('--threads', type=int, default=10, help='Number of worker threads.')
    parser.add_argument('--max_pages', type=int, default=500, help='Maximum number of pages to crawl.')
    parser.add_argument('--seed_pages', type=int, default=10, help='Number of seed pages to fetch from search engine.')
    
    args = parser.parse_args()

    crawler = WebCrawler(
        search_term=args.search,
        num_threads=args.threads,
        max_pages=args.max_pages,
        seed_pages=args.seed_pages,
    )
    crawler.start_crawling()