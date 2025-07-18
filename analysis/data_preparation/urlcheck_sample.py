import pandas as pd
import requests
import random
from datetime import datetime
import time
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
INPUT_FILE = 'data/processed/orgs_2012_2018_survived.csv'
OUTPUT_FILE = 'data/processed/urlcheck.csv'
CHUNK_SIZE = 20  # Write every N results
MAX_WORKERS = 2
PROGRESS_INTERVAL = 100

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
]

# Helper function to check website status
KEYWORDS_FOR_SALE = [
    'for sale', 'buy this domain', 'domain parked', 'this domain is for sale',
    'is available for purchase', 'make this domain yours', 'domainmarket', 'sedo', 'afternic', 'godaddy'
]
KEYWORDS_DOWN = [
    'under construction', 'coming soon', 'website expired', 'site not found', 'cannot be reached', 'not found', '404', '502', '503', 'maintenance', 'temporarily unavailable'
]

def is_for_sale_page(resp):
    try:
        content = resp.text.lower()
        soup = BeautifulSoup(resp.text, 'html.parser')
        provider_signatures = [
            'godaddy', 'afternic', 'sedo', 'namecheap', 'domainmarket', 'hugedomains', 'uniregistry', 'dan.com', 'parkingcrew', 'bodis'
        ]
        for_sale_phrases = [
            'for sale', 'get this domain', 'this domain is registered', 'may still be available',
            'buy now', 'make an offer', 'inquire about this domain', 'this domain may be for sale'
        ]
        # Check for provider signature anywhere in the content
        if any(sig in content for sig in provider_signatures):
            # Extract all visible text
            visible_text = ' '.join(soup.stripped_strings).lower()
            if any(phrase in visible_text for phrase in for_sale_phrases):
                return True
            # Check for key button/link text
            for button in soup.find_all(['button', 'a']):
                if any(phrase in button.get_text(strip=True).lower() for phrase in for_sale_phrases):
                    return True
            # Check for 'powered by' branding
            if 'powered by afternic' in visible_text or 'a godaddy brand' in visible_text:
                return True
            # If provider signature and very minimal content, likely for sale
            if len(soup.find_all(['div', 'button', 'a', 'span', 'p'])) < 10:
                return True
        return False
    except Exception:
        return False

def label_website_status(row):
    url = row['homepage_url']
    print(f"Checking {url}...", flush=True)
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code >= 400:
            return {**row, 'website_status': 'down', 'last_modified': resp.headers.get('Last-Modified', ''), 'error': f'HTTP {resp.status_code}'}
        if is_for_sale_page(resp):
            return {**row, 'website_status': 'for sale', 'last_modified': resp.headers.get('Last-Modified', ''), 'error': ''}
        if len(resp.text.strip()) < 100:
            return {**row, 'website_status': 'empty', 'last_modified': resp.headers.get('Last-Modified', ''), 'error': ''}
        # If the page loads, even if minimal, consider it 'active'
        return {**row, 'website_status': 'active', 'last_modified': resp.headers.get('Last-Modified', ''), 'error': ''}
    except Exception as e:
        return {**row, 'website_status': 'down', 'last_modified': '', 'error': str(e)}

TEST_MODE = True  # Set to False for full run
TEST_N = 10

def main():
    print('Loading data...')
    df = pd.read_csv(INPUT_FILE, usecols=['uuid', 'name', 'homepage_url', 'created_at', 'updated_at'])
    df = df[df['homepage_url'].notna() & (df['homepage_url'].str.strip() != '')]

    # Resume: load already processed uuids
    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        done_uuids = set(done_df['uuid'])
        print(f'Resuming: {len(done_uuids)} URLs already processed.')
    else:
        done_uuids = set()

    to_process = df[~df['uuid'].isin(done_uuids)]
    if TEST_MODE:
        to_process = to_process.head(TEST_N)
        print(f"TEST MODE: Only processing {TEST_N} rows.")
    print(f'Total to process: {len(to_process)}')

    results = []
    processed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(label_website_status, row): row for _, row in to_process.iterrows()}
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
            processed += 1
            # Write every CHUNK_SIZE
            if len(results) >= CHUNK_SIZE:
                out_df = pd.DataFrame(results)
                header = not os.path.exists(OUTPUT_FILE)
                out_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
                results = []
            # Progress logging
            if processed % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f'Processed {processed} URLs in {elapsed:.1f} seconds.')
            # Random polite delay
            time.sleep(random.uniform(1, 3))
    # Write any remaining results
    if results:
        out_df = pd.DataFrame(results)
        header = not os.path.exists(OUTPUT_FILE)
        out_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
    print('Done!')

if __name__ == '__main__':
    main() 