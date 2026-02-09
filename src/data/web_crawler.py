import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urlparse

class WebCrawler:
    """
    Crawls web pages for developer articles and documentation.
    """
    
    def __init__(self, output_dir: str = "data/articles"):
        """
        Initialize web crawler.
        
        Args:
            output_dir: Directory to save articles
        """
        self.output_dir = output_dir
        self.session = requests.Session()
        # Pretend to be a browser to avoid some 403s
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        })
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def crawl_article(self, url: str):
        """
        Fetch and parse a single article.
        """
        print(f"🕷️  Fetching article: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract title
            title = soup.title.string if soup.title else "Untitled"
            clean_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
            
            # Extract main content (heuristics)
            # 1. Look for <article> tag
            article_body = soup.find('article')
            if not article_body:
                # 2. Look for <main> tag
                article_body = soup.find('main')
            if not article_body:
                # 3. Fallback to body
                article_body = soup.body
                
            if not article_body:
                print("❌ Could not find article content.")
                return

            # Extract paragraphs
            paragraphs = article_body.find_all('p')
            text_content = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            
            if not text_content:
                print("⚠️  No text found in paragraphs.")
                return
                
            # Create filename from URL domain + title
            domain = urlparse(url).netloc
            filename = f"{domain}_{clean_title[:50]}.txt".replace(" ", "_")
            file_path = os.path.join(self.output_dir, filename)
            
            # Save format
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n")
                f.write(f"Source: {url}\n")
                f.write("\n")
                f.write("\n\n".join(text_content))
                
            print(f"   ✓ Saved to {filename} ({len(text_content)} paragraphs)")
            
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")

if __name__ == "__main__":
    crawler = WebCrawler()
    # crawler.crawl_article("https://react.dev/learn") 
    print("WebCrawler module ready.")
