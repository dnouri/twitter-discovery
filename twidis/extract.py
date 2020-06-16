from datetime import datetime
import hashlib
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
from time import time
from urllib.parse import urlparse

import arxiv
from newspaper import Article
import nltk
import requests
from tqdm import tqdm


logger = logging.getLogger('twidis')
nltk.download('punkt')


USER_AGENTS = [
    'twidis/0.1',
    ]

if os.path.exists('user-agents.txt'):
    with open('user-agents.txt') as fua:
        USER_AGENTS = [
            line.strip() for line in fua
            if line.strip() and not line.strip().startswith('#')
            ]

ARTICLE_KEYS = [
    'url', 'title', 'text', 'html',
    'meta_keywords', 'meta_description',
    'authors', 'publish_date',
    'summary', 'keywords',
    ]


def extract_arxiv_article(url):
    arxiv_id = url.split('/')[4].replace('.pdf', '')
    result = tuple(arxiv.query(
        id_list=[arxiv_id],
        max_results=1,
        ))[0]
    html = '' if url.endswith('.pdf') else requests.get(url).text
    return {
        'url': f'https://arxiv.org/abs/{arxiv_id}',
        'title': result['title'],
        'text': result['summary'],
        'html': html,
        'authors': result['authors'],
        'publish_date': datetime.strptime(
            result['published'], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None),
        'summary': result['summary'],
        }


def extract_newspaper_article(url):
    article = Article(
        url,
        fetch_images=False,
        browser_user_agent=random.choice(USER_AGENTS),
        )
    article.download()
    article.parse()
    article.nlp()
    return {
        key: getattr(article, key) for key in ARTICLE_KEYS
        }


extractors = [
    (
        lambda url, ctype:
        url.startswith('https://arxiv.org/abs/') or
        url.startswith('https://arxiv.org/pdf/'),
        extract_arxiv_article,
    ),
    (
        lambda url, ctype: ctype.split(';')[0] == 'text/html',
        extract_newspaper_article,
    ),
]


def extract_article(url, timeout_head=10):
    article_info = {'request_url': url, 'url': url}

    t0 = time()
    try:
        resp = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout_head,
            headers={'User-Agent': random.choice(USER_AGENTS)},
            )
        resp.raise_for_status()
        url = article_info['url'] = resp.url
    except Exception as exc:
        logger.error(f'Failed to process {url}:')
        logger.exception(exc)
        logger.error('\n\n')
        article_info['error'] = str(exc)
        return article_info

    for checker, extractor in extractors:
        if checker(url, resp.headers.get('Content-Type', 'n/a')):
            break
    else:
        error = f'No extractor for {resp.headers}'
        logger.error(f'Failed to process {url}:')
        logger.error(error)
        logger.error('\n\n')
        article_info['error'] = error
        return article_info

    try:
        article_info.update(extractor(article_info['url']))
    except Exception as exc:
        logger.error(f'Failed to process {url}:')
        logger.exception(exc)
        logger.error('\n\n')
        article_info['error'] = str(exc)
        return article_info

    url_components = urlparse(article_info['url'])
    for key in (
            'scheme', 'path', 'netloc', 'path', 'params', 'query', 'fragment'):
        article_info[f'url_{key}'] = getattr(url_components, key)

    logger.debug(f'Processing {article_info["url"]} took {time()-t0:.2f} secs')
    return article_info


def extract_articles(urls, n_threads):
    os.makedirs('cache', exist_ok=True)
    urls_hash = hashlib.sha1(' :: '.join(urls).encode('utf-8')).hexdigest()
    fname_cache = f'cache/twidis-cache-{urls_hash}.pkl'
    if os.path.exists(fname_cache):
        with open(fname_cache, 'rb') as fcache:
            return pickle.load(fcache)

    pool = ThreadPool(n_threads)
    articles = []
    iterator = tqdm(
        pool.imap_unordered(extract_article, urls),
        total=len(urls),
        )
    good = 0
    for count, article in enumerate(iterator, start=1):
        articles.append(article)
        if 'error' not in article:
            good += 1
        url_abbr = article['url'].split('//', 1)[1]
        if len(url_abbr) > 30:
            url_abbr = url_abbr[:20] + '...' + url_abbr[-7:]
        iterator.set_description(
            f'[{good}/{count}] extract_articles: ' +
            ('[err] ' if 'error' in article else '') +
            url_abbr
            )

    articles = sorted(articles, key=lambda a: urls.index(a['request_url']))
    errors = sum(['error' in article.keys() for article in articles])
    if errors < len(articles):
        with open(fname_cache, 'wb') as fcache:
            pickle.dump(articles, fcache)
        with open(f'{fname_cache}.meta', 'w') as fmeta:
            fmeta.write(f"{len(articles)-errors}/{len(articles)}\n")

    return articles


def extract_articles_in_chunks(urls, chunk_size, n_threads):
    articles = []
    for index in range(0, len(urls), chunk_size):
        articles.extend(extract_articles(
            urls[index:index+chunk_size], n_threads=n_threads))
        print(f"extract_articles_in_chunks: "
              f"Processed {index+chunk_size}/{len(urls)}")
    return articles
