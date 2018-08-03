from requests.auth import HTTPBasicAuth
from settings.basic import (logging, CACHE_ENABLED, CACHE_PATH, intrinio_username,
                            intrinio_password)

from urllib.parse import urlparse

import requests
import json
import os
import base64


def call_and_cache(url: str, **kwargs) -> dict:
    """
    Calls the URL with GET method if the url file is not cached
    :param url: url to retrieve
    :param kwargs: specify no-cache
    :return: json.loads of the response (or empty dict if error)
    """
    url_parsed = urlparse(url)

    cached_file = os.path.join(CACHE_PATH, url_parsed.netloc + url_parsed.path + "/" +
                               base64.standard_b64encode(url_parsed.query.encode()).decode())

    if not os.path.exists(os.path.dirname(cached_file)):
        os.makedirs(os.path.dirname(cached_file))

    try:
        no_cache = kwargs['no-cache']
    except KeyError:
        no_cache = False

    data_json = {}
    if CACHE_ENABLED and os.path.exists(cached_file) and not no_cache:
        logging.debug("Data was present in cache and cache is enabled, loading: %s for %s" %
                      (cached_file, url))
        with open(cached_file, 'r') as f:
            data_json = json.loads(f.read())
    else:
        logging.info(
            "Data was either not present in cache or it was disabled calling request: %s" % url)
        r = requests.get(url, auth=HTTPBasicAuth(intrinio_username, intrinio_password))

        if r.status_code != 200:
            logging.error("Request status was: %s for URL: %s" % (r.status_code, url))
            return data_json

        data_json = json.loads(r.text)

        if not len(data_json['data']) > 0:
            logging.debug("Data field is empty.\nRequest URL: %s" % (url))

        with open(cached_file, 'w') as f:
            f.write(json.dumps(data_json))
            logging.debug("Successfully cached url: %s to %s" % (url, cached_file))

    return data_json
