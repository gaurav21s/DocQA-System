import scrapy
import os
import json
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from utils.logger import logger

class WebCrawler(scrapy.Spider):
    name = 'general_docs_spider'
    allowed_domains = []
    start_urls = []
    visited_urls = set()
    max_depth = None

    custom_settings = {
        'CONCURRENT_REQUESTS': 16,  # Increase concurrent requests
        'DOWNLOAD_DELAY': 0.5,  # Adjust the delay as necessary
    }

    def __init__(self, start_url, max_depth, *args, **kwargs):
        super(WebCrawler, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]
        self.max_depth = max_depth

    def parse(self, response):
        if response.url in self.visited_urls:
            return
        self.visited_urls.add(response.url)
        yield self.extract_data(response)

        depth = response.meta.get('depth', 1)
        if depth < self.max_depth:
            soup = BeautifulSoup(response.body, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                url = link['href']
                absolute_url = urljoin(response.url, url)
                if self.allowed_domains[0] in absolute_url and absolute_url not in self.visited_urls:
                    yield scrapy.Request(
                        absolute_url,
                        callback=self.parse,
                        meta={'depth': depth + 1}
                    )

    def extract_data(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')
        title = soup.title.string if soup.title else ''
        main_content = ''

        # Selective parsing for speed optimization
        main_div = soup.find('div', class_='document') or soup.find('main')
        if main_div:
            main_content = main_div.get_text(strip=True)
        else:
            main_content = soup.body.get_text(strip=True) if soup.body else ''
        return {
            'url': response.url,
            'title': title,
            'content': main_content,
        }

def run_spider(start_url, max_depth):
    logger.info(f"Spider is running for: {start_url} and for depth: {max_depth}")
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': 'output.json'
    })
    file_path = "output.json"
    if not os.path.exists('output.json'):
        process.crawl(WebCrawler, start_url=start_url, max_depth=max_depth)
        process.start()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    filtered_data = [entry for entry in data if entry.get('content')]

    with open(file_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    logger.info(f"Preprocessing complete. {len(filtered_data)} entries remain.")


    
# if __name__ == "__main__":
#     start_url = 'https://docs.nvidia.com/cuda/'  # Example start URL
#     max_depth = 3
#     run_spider(start_url, max_depth)
