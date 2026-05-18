"""Feature extraction helpers for email, URL, header, and sender address features."""
import re
import string
from urllib.parse import urlparse
from collections import Counter
import numpy as np

from extract_email_address_features import extract_email_address_features

URL_PATTERN = r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
SUSPICIOUS_WORDS = ['urgent', 'verify', 'password', 'click', 'account', 'bank', 'login', 'suspend', 'expired']


def extract_urls(text):
    return re.findall(URL_PATTERN, str(text) or '')


def extract_email_text_features(text):
    text = str(text or '')
    text_lower = text.lower()

    words = text.split()
    num_words = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    vocab_richness = len(set(words)) / num_words if num_words else 0.0

    return [
        len(text),
        len(extract_urls(text)),
        sum(1 for c in text if c.isupper()),
        text.count('!'),
        sum(c.isdigit() for c in text),
        sum(word in text_lower for word in SUSPICIOUS_WORDS),
        num_words,
        avg_word_length,
        vocab_richness,
    ]

def extract_url_features(url):
    url = str(url or '')
    parsed = urlparse(url)
    features = []

    features.append(len(url))
    domain_parts = parsed.netloc.split('.') if parsed.netloc else []
    features.append(max(len(domain_parts) - 1, 0))
    features.append(1 if re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', parsed.netloc) else 0)
    features.append(1 if parsed.scheme == 'https' else 0)
    features.append(sum(1 for c in url if c in string.punctuation))

    char_counts = Counter(url)
    total_chars = len(url)
    entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in char_counts.values()) if total_chars else 0.0
    features.append(entropy)

    features.append(len(parsed.path))
    features.append(len(parsed.query))
    features.append(sum(c.isdigit() for c in url))
    features.append(sum(c.isalpha() for c in url))
    features.append(1 if '@' in url else 0)
    features.append(1 if '-' in parsed.netloc else 0)
    features.append(len(domain_parts[-1]) if domain_parts else 0)
    
    url_lower = url.lower()
    features.append(1 if 'login' in url_lower else 0)
    features.append(1 if 'admin' in url_lower else 0)
    features.append(1 if 'client' in url_lower else 0)
    features.append(1 if 'update' in url_lower else 0)
    features.append(1 if 'free' in url_lower else 0)
    features.append(1 if parsed.port is not None else 0)
    features.append(1 if parsed.netloc.startswith('www') else 0)

    features += [0] * (25 - len(features))
    return features[:25]


def extract_header_features(headers):
    """Placeholder for header-derived phishing signals."""
    headers = headers or {}
    features = {
        'has_mismatched_reply_to': int(headers.get('reply_to') != headers.get('from')), 
        'missing_auth_results': int('auth' not in headers.get('authentication_results', '').lower()),
        'from_contains_ip': int(bool(re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', headers.get('from', '')))),
    }
    return list(features.values())


def fuse_features(email_text_feats, url_features_list, email_addr_feats):
    if url_features_list:
        avg_url = np.mean(url_features_list, axis=0)
    else:
        avg_url = np.zeros(25)
    return np.concatenate([email_text_feats, avg_url, email_addr_feats])

FEATURE_NAMES = [

    # =================================================
    # EMAIL FEATURES (9)
    # =================================================

    'email_length',
    'num_urls',
    'num_uppercase',
    'num_exclamations',
    'num_digits',
    'suspicious_word_count',
    'num_words',
    'avg_word_length',
    'vocab_richness',

    # =================================================
    # URL FEATURES (25)
    # =================================================

    'url_length',
    'num_subdomains',
    'has_ip',
    'is_https',
    'special_chars',
    'entropy',
    'path_length',
    'query_length',
    'num_digits_url',
    'num_letters_url',
    'has_at',
    'has_hyphen',
    'tld_length',
    'has_login',
    'has_admin',
    'has_client',
    'has_update',
    'has_free_url',
    'has_port',
    'has_www',

    'feature_20',
    'feature_21',
    'feature_22',
    'feature_23',
    'feature_24',

    # =================================================
    # EMAIL ADDRESS FEATURES (5)
    # =================================================

    'email_addr_len',
    'domain_len',
    'suspicious_tld',
    'free_domain',
    'num_dots_domain',
]
