import pandas as pd
import requests
import random
from datetime import datetime
import time
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import dns.resolver
import dns.exception

# Parameters
INPUT_FILE      = 'data/processed/orgs_2012_2018_survived.csv'
OUTPUT_FILE     = 'data/processed/urlcheck.csv'
CHUNK_SIZE      = 20   # Write every N results
MAX_WORKERS     = 2
PROGRESS_INTERVAL = 100

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
]

# Keywords
KEYWORDS_FOR_SALE = [
    'for sale', 'buy this domain', 'domain parked', 'this domain is for sale',
    'is available for purchase', 'make this domain yours',
    'domainmarket', 'sedo', 'afternic', 'godaddy', 'get this domain',
    'buy now', 'make an offer', 'inquire about this domain'
]
KEYWORDS_DOWN = [
    'under construction', 'coming soon', 'website expired',
    'site not found', 'cannot be reached', 'not found',
    '404', '502', '503', 'maintenance', 'temporarily unavailable'
]

# Expanded provider domains for parking/sale detection
PROVIDER_DOMAINS = [
    'sedo.com', 'afternic.com', 'godaddy.com', 'dan.com', 'bodis.com', 'parkingcrew.net',
    'namecheap.com', 'domainmarket.com', 'hugedomains.com', 'undeveloped.com', 'buydomains.com',
    'uniregistry.com', 'sav.com', 'internettraffic.com', 'above.com', 'voodoo.com',
]

# Helper: DNS CNAME/A record check for known parking providers

def is_dns_parked(domain):
    try:
        # Check CNAME
        answers = dns.resolver.resolve(domain, 'CNAME', lifetime=3)
        for rdata in answers:
            target = str(rdata.target).lower()
            if any(p in target for p in PROVIDER_DOMAINS):
                return True
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout):
        pass
    except Exception:
        pass
    try:
        # Check A records (IP-based parking detection, optional: can add known IPs)
        answers = dns.resolver.resolve(domain, 'A', lifetime=3)
        # Optionally, compare to known parking IPs here
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout):
        pass
    except Exception:
        pass
    return False

# Enhanced for-sale detection

def is_for_sale_page(resp):
    try:
        soup = BeautifulSoup(resp.text, 'html.parser')
        visible = ' '.join(soup.stripped_strings).lower()
        title = soup.title.string.lower() if soup.title and soup.title.string else ''
        html = resp.text.lower() if resp.text else ''
        # 1. Expanded keyword check
        if any(kw in visible for kw in KEYWORDS_FOR_SALE) or any(kw in title for kw in KEYWORDS_FOR_SALE):
            return True
        # 2. Provider domain check in links, images, scripts
        for tag in soup.find_all(['a', 'img', 'script']):
            for attr in ['href', 'src']:
                url = tag.get(attr, '')
                if any(domain in url for domain in PROVIDER_DOMAINS):
                    return True
        # 3. Meta tag check
        for meta in soup.find_all('meta'):
            content = meta.get('content', '').lower()
            if any(kw in content for kw in KEYWORDS_FOR_SALE):
                return True
        # 4. Favicon check
        for link in soup.find_all('link', rel='icon'):
            href = link.get('href', '')
            if any(domain in href for domain in PROVIDER_DOMAINS):
                return True
        # 5. Provider signature in HTML source (not just visible text)
        provider_sigs = ['godaddy', 'afternic', 'sedo', 'namecheap', 'parkingcrew', 'bodis', 'domainmarket', 'hugedomains', 'dan.com', 'buydomains', 'uniregistry', 'sav.com']
        if any(sig in html for sig in provider_sigs):
            return True
        # 6. Provider signature in title
        if any(sig in title for sig in provider_sigs):
            return True
        # 7. Minimal visible text but provider in HTML → for sale
        if len(visible) < 300 and any(sig in html for sig in provider_sigs):
            return True
        return False
    except Exception:
        return False

def is_down_page(resp):
    """Returns True if the response contains ‘down’ or ‘error’ language."""
    try:
        soup = BeautifulSoup(resp.text, 'html.parser')
        visible = ' '.join(soup.stripped_strings).lower()
        # Only mark as down if explicit error language is found
        if any(kw in visible for kw in KEYWORDS_DOWN):
            return True
        # Remove: if len(visible) < 50: return True
        return False
    except Exception:
        return False

def redirected_to_parking(resp, original_url):
    """Detect a redirect onto a known parking provider domain."""
    try:
        orig = urlparse(original_url).netloc.lower()
        final = urlparse(resp.url).netloc.lower()
        if orig != final and any(p in final for p in ['sedo', 'afternic', 'godaddy']):
            return True
    except Exception:
        pass
    return False

def label_website_status(row):
    url = row['homepage_url']
    print(f"Checking {url}...", flush=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        # 1) HTTP 4xx/5xx → down
        if resp.status_code >= 400:
            return {**row,
                    'website_status': 'down',
                    'last_modified': resp.headers.get('Last-Modified', ''),
                    'error': f'HTTP {resp.status_code}'}
        # 2) redirect onto parking site OR parked keywords OR DNS parking OR explicit down-language OR tiny/empty page OR no visible content in <body> → down
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0].lower()
        soup = BeautifulSoup(resp.text, 'html.parser')
        body = soup.body
        visible = ''
        if body:
            visible = ' '.join(body.stripped_strings)
        # Improved logic: allow JS-heavy sites with substantial HTML to be marked as active
        if (
            redirected_to_parking(resp, url)
            or is_for_sale_page(resp)
            or is_dns_parked(domain)
            or is_down_page(resp)
            or len(resp.text.strip()) < 300
            or (len(visible.strip()) < 20 and len(resp.text.strip()) < 2000)
        ):
            return {**row,
                    'website_status': 'down',
                    'last_modified': resp.headers.get('Last-Modified', ''),
                    'error': ''}
        # 3) otherwise → active
        return {**row,
                'website_status': 'active',
                'last_modified': resp.headers.get('Last-Modified', ''),
                'error': ''}
    except Exception as e:
        return {**row,
                'website_status': 'down',
                'last_modified': '',
                'error': str(e)}

# Test mode for quick checks
TEST_MODE = False
TEST_N = 10

def main():
    print('Loading data...')
    df = pd.read_csv(INPUT_FILE, usecols=['uuid','name','homepage_url','created_at','updated_at'])
    df = df[df['homepage_url'].notna() & (df['homepage_url'].str.strip() != '')]

    # Resume if output exists
    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        done_uuids = set(done_df['uuid'])
        print(f'Resuming: {len(done_uuids)} URLs already processed.')
    else:
        done_uuids = set()

    print(f"Total URLs in input: {len(df)}")
    print(f"Already processed: {len(done_uuids)}")

    to_process = df[~df['uuid'].isin(done_uuids)]
    print(f"To process after filtering: {len(to_process)}")

    if TEST_MODE:
        to_process = to_process.head(TEST_N)
        print(f"TEST MODE: Only processing {len(to_process)} rows (limited by TEST_N={TEST_N}).")
    else:
        print(f"Processing {len(to_process)} rows.")
    print(f'Total to process: {len(to_process)}')

    results = []
    processed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(label_website_status, row): row
            for _, row in to_process.iterrows()
        }
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
            processed += 1

            # Write in chunks
            if len(results) >= CHUNK_SIZE:
                out_df = pd.DataFrame(results)
                header = not os.path.exists(OUTPUT_FILE)
                out_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
                results = []

            # Progress log
            if processed % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f'Processed {processed} URLs in {elapsed:.1f} seconds.')

            # Polite delay
            time.sleep(random.uniform(1, 3))

    # Flush remainder
    if results:
        out_df = pd.DataFrame(results)
        header = not os.path.exists(OUTPUT_FILE)
        out_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)

    print('Done!')

if __name__ == '__main__':
    main()
