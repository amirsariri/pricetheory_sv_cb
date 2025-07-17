import pandas as pd
import requests
import random
from datetime import datetime
import time
from bs4 import BeautifulSoup

# Parameters
INPUT_FILE = 'data/processed/orgs_2012_2018_survived.csv'
N_SAMPLE = 10
DATE_STR = datetime.now().strftime('%y%m%d')
OUTPUT_FILE = f'data/processed/{DATE_STR}_urlcheck_sample.csv'

# Helper function to check website status
KEYWORDS_FOR_SALE = [
    'for sale', 'buy this domain', 'domain parked', 'this domain is for sale',
    'is available for purchase', 'make this domain yours', 'domainmarket', 'sedo', 'afternic', 'godaddy'
]
KEYWORDS_DOWN = [
    'under construction', 'coming soon', 'website expired', 'site not found', 'cannot be reached', 'not found', '404', '502', '503', 'maintenance', 'temporarily unavailable'
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; CompanyStatusBot/1.0)'
}

def is_for_sale_page(resp):
    try:
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string.lower() if soup.title and soup.title.string else ''
        # Only mark as for sale if a provider signature is present in the title or in a known parking page structure
        provider_signatures = [
            'godaddy', 'sedo', 'afternic', 'namecheap', 'domainmarket', 'hugedomains', 'uniregistry', 'dan.com', 'parkingcrew', 'bodis'
        ]
        # Check title for exact provider signature
        if any(sig in title for sig in provider_signatures):
            return True
        # Check for known parking page meta tags or banners
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and any(sig in meta.get('content', '').lower() for sig in provider_signatures):
            return True
        # Check for exact for-sale titles
        exact_for_sale_titles = [
            'domain for sale', 'buy this domain', 'this domain is for sale', 'is available for purchase'
        ]
        if title.strip() in exact_for_sale_titles:
            return True
        return False
    except Exception:
        return False

def label_website_status(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code >= 400:
            return 'down', resp.headers.get('Last-Modified', '')
        if is_for_sale_page(resp):
            return 'for sale', resp.headers.get('Last-Modified', '')
        # If the page loads, even if minimal, consider it 'active'
        return 'active', resp.headers.get('Last-Modified', '')
    except Exception:
        return 'down', ''

# Load data
print('Loading data...')
df = pd.read_csv(INPUT_FILE, usecols=['uuid', 'name', 'homepage_url', 'created_at', 'updated_at'])
df = df[df['homepage_url'].notna() & (df['homepage_url'].str.strip() != '')]

# Sample 10 companies with a new random seed
sample_df = df.sample(n=N_SAMPLE, random_state=99)

results = []
for idx, row in sample_df.iterrows():
    url = row['homepage_url']
    print(f'Checking {url}...')
    status, last_modified = label_website_status(url)
    results.append({
        'uuid': row['uuid'],
        'name': row['name'],
        'homepage_url': url,
        'created_at': row['created_at'],
        'updated_at': row['updated_at'],
        'website_status': status,
        'last_modified': last_modified
    })
    time.sleep(1)  # Be polite to servers

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)
print(f'Saved results to {OUTPUT_FILE}') 