#!/bin/bash

# List of URLs to crawl
URLS=(
'https://www.reddit.com/r/programming/'

)


echo "🕷️  Starting bulk crawl of ${#URLS[@]} articles..."

# No cd needed if run from project root
for url in "${URLS[@]}"; do
    echo "----------------------------------------"
    python cli.py crawl-web "$url"
    sleep 2
done

echo "----------------------------------------"
echo "✅ Bulk crawl finished!"
