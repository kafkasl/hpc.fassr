import os
from string import Template

from settings.basic import logging, DATA_PATH
from utils import call_and_cache

company_url_template = Template(
    'https://api.intrinio.com/companies?identifier=${symbol}')

symbols_list_name = 'dow30'

symbols = open(os.path.join(DATA_PATH,
                            '%s_symbols.lst' % symbols_list_name)).read().split()

symbol2sic = []

for symbol in symbols:

    url = company_url_template.substitute(symbol=symbol)

    data_json = call_and_cache(url)

    try:
        symbol2sic.append((symbol, data_json['sic']))
    except KeyError:
        print(
            "KeyError in data_json[sic], probably no info about %s is available." % symbol)

with open(os.path.join(DATA_PATH, '%s_sic.txt' % symbols_list_name), 'w') as f:
    f.write('\n'.join(['%s,%s' % (s, sic) for s, sic in symbol2sic]))


def get_sic_industry_name(code) -> str:
    if 100 <= code <= 999:
        return 'Agriculture, Forestry, and Fishing'
    if 1000 <= code <= 1499:
        return 'Mining'
    if 1500 <= code <= 1799:
        return 'Construction'
    if 2000 <= code <= 3999:
        return 'Manufacturing'
    if 4000 <= code <= 4999:
        return 'Transportation, Communications, Electric, Gas and Sanitary service'
    if 5000 <= code <= 5199:
        return 'Wholesale Trade'
    if 5200 <= code <= 5999:
        return 'Retail Trade'
    if 6000 <= code <= 6799:
        return 'Finance, Insurance and Real Estate'
    if 7000 <= code <= 8999:
        return 'Services'
    if 9100 <= code <= 9729:
        return 'Public Administration'
    if 9900 <= code <= 9999:
        return 'Nonclassifiable'
    raise ValueError('Invalid sic code: %s:' % code)


sicind2cate = {'Agriculture, Forestry, and Fishing': 0,
               'Mining': 1,
               'Construction': 2,
               'Manufacturing': 3,
               'Transportation, Communications, Electric, Gas and Sanitary service': 4,
               'Wholesale Trade': 5,
               'Retail Trade': 6,
               'Finance, Insurance and Real Estate': 7,
               'Services': 8,
               'Public Administration': 9,
               'Nonclassifiable': 10}


def get_sic_industry(code) -> int:
    name = get_sic_industry_name(code)
    return sicind2cate[name]


def load_sic(symbols_list_name: str = 'sp500') -> dict:
    # sp500 contains dow30, so we just use sp500 sic info

    sic_code = {}
    for row in open(os.path.join(DATA_PATH, 'sp500_sic.txt')):
        symbol, sic = row.split(',')
        sic_code[symbol] = int(sic)
    sic_industry = {symbol: get_sic_industry(sic) for symbol, sic in
                    sic_code.items()}

    return sic_code, sic_industry
