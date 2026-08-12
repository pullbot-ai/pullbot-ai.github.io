"""Wikipedia scraper + definition lookup"""
import os, sys, json, time, requests, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def scrape_wikipedia():
    print("📚 Scraping Wikipedia...")
    texts = []
    for _ in range(10):
        try:
            r = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                timeout=15,
                headers={'User-Agent': 'Pullbot/1.0'}
            )
            if r.status_code == 200:
                data = r.json()
                text = data.get('extract', '')
                title = data.get('title', '')
                if len(text) > 100:
                    texts.append({'source': f'wiki:{title}', 'text': text})
            time.sleep(0.3)
        except:
            continue
    print(f"  Got {len(texts)} articles")
    return texts

def lookup_definitions(wordbank, limit=50):
    print("📖 Looking up definitions...")
    undefined = [w for w, i in wordbank.get('words', {}).items()
                 if not isinstance(i, dict) or not i.get('has_definition')][:limit]
    defined = 0
    for word in undefined:
        try:
            r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                definition = data[0]['meanings'][0]['definitions'][0]['definition']
                wordbank['words'][word] = {'has_definition': True, 'definition': definition}
                defined += 1
            time.sleep(0.2)
        except:
            pass
    wordbank['total_defined'] = sum(1 for w in wordbank['words'].values()
                                     if isinstance(w, dict) and w.get('has_definition'))
    print(f"  Defined {defined} words")
    return wordbank

def extract_words(text):
    return list(set(re.findall(r'\b[a-z]{3,}\b', text.lower())))

def run_scraper():
    wordbank_path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    if os.path.exists(wordbank_path):
        with open(wordbank_path) as f:
            bank = json.load(f)
    else:
        bank = {"words": {}, "total_articles": 0, "total_words": 0, "total_defined": 0}
    
    articles = scrape_wikipedia()
    for a in articles:
        words = extract_words(a['text'])
        for w in words:
            if w not in bank['words']:
                bank['words'][w] = {}
        bank['total_articles'] += 1
    
    bank = lookup_definitions(bank)
    bank['total_words'] = len(bank['words'])
    
    os.makedirs(os.path.dirname(wordbank_path), exist_ok=True)
    with open(wordbank_path, 'w') as f:
        json.dump(bank, f, indent=2)
    
    print(f"✅ Wordbank: {bank['total_words']} words, {bank['total_defined']} defined")

if __name__ == '__main__':
    run_scraper()
