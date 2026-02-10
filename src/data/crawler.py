"""
GitHub Crawler - Fetch training data from repositories.

This module crawls GitHub repositories and downloads JavaScript/TypeScript files
to build a large dataset for training.
"""

import os
import requests
import time
from typing import List, Set, Optional


class GitHubCrawler:
    """
    Crawls GitHub repositories for code files.
    """
    
    def __init__(self, output_dir: str = "data/crawled", token: Optional[str] = None):
        """
        Initialize crawler.
        
        Args:
            output_dir: Base directory to save crawled repos
            token: GitHub API token (optional, for higher rate limits)
        """
        self.output_dir = output_dir
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"token {token}"})
        
        # Extensions to look for
        self.extensions = {'.js', '.ts', '.jsx', '.tsx'}
        
        # Directories to ignore
        self.ignore_dirs = {
            'node_modules', 'dist', 'build', 'coverage', 
            'test', 'tests', '__tests__', 'docs', 'examples'
        }
        
        self.stats = {
            'files_downloaded': 0,
            'lines_saved': 0,
            'errors': 0
        }

    def crawl(self, repo_url: str):
        """
        Crawl a repository.
        
        Args:
            repo_url: "owner/repo" string (e.g. "facebook/react")
        """
        print(f">> Crawling {repo_url}...")
        
        # Create repo-specific directory
        repo_name = repo_url.replace("/", "_")
        target_dir = os.path.join(self.output_dir, repo_name)
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Start from root
            self._process_contents(repo_url, "", target_dir)
        except Exception as e:
            print(f"[ERROR] Error crawling {repo_url}: {e}")
            self.stats['errors'] += 1
            
        print("\n-- Crawl finished!")
        print(f"   Files: {self.stats['files_downloaded']}")
        print(f"   Lines: {self.stats['lines_saved']}")
        print(f"   Errors: {self.stats['errors']}")
        print(f"   Saved to: {target_dir}")

    def _process_contents(self, repo: str, path: str, target_dir: str):
        """
        Recursively process repository contents.
        """
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = self.session.get(url)
        
        if response.status_code == 403:
            print("[WARN] Rate limit hit! Waiting 60s...")
            time.sleep(60)
            response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"   Failed to list {path}: {response.status_code}")
            return
            
        items = response.json()
        
        for item in items:
            name = item['name']
            
            if item['type'] == 'dir':
                if name not in self.ignore_dirs and not name.startswith('.'):
                    # Recursive call for directories
                    time.sleep(0.1)
                    self._process_contents(repo, item['path'], target_dir)
                    
            elif item['type'] == 'file':
                ext = os.path.splitext(name)[1].lower()
                if ext in self.extensions:
                    self._download_file(item, target_dir)

    def _download_file(self, item_data: dict, target_dir: str):
        """Download and save a single file."""
        download_url = item_data.get('download_url')
        if not download_url:
            return
            
        try:
            response = self.session.get(download_url)
            if response.status_code == 200:
                content = response.text
                
                # Basic cleaning
                if not content.strip():
                    return
                
                # Create structured filename: path_to_file.js -> path_to_file.js
                # Replace slashes with underscores to flatten structure, or replicate structure?
                # Let's keep it flat for now to avoid deep nesting issues, but preserve path in filename check
                
                # Actually, let's replicate structure! It's cleaner.
                rel_path = item_data['path']
                file_path = os.path.join(target_dir, rel_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.stats['files_downloaded'] += 1
                self.stats['lines_saved'] += len(content.splitlines())
                print(f"   [OK] {item_data['path']}")
                
        except Exception as e:
            print(f"   [ERROR] Failed to download {item_data['path']}: {e}")
            self.stats['errors'] += 1


if __name__ == "__main__":
    # Test crawler
    crawler = GitHubCrawler(output_dir="data/test_crawled")
    print("Crawler module ready.")
