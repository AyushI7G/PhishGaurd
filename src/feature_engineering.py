import re
from urllib.parse import urlparse

def extract_features(url):
    parsed = urlparse(url)

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        1 if "https" in url else 0,
        len(re.findall(r'\d', url)),
        len(parsed.netloc),
        len(parsed.path),
        1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0,
        int(any(word in url for word in ["login", "verify", "bank", "secure"]))
    ]