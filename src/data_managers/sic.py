from string import Template

from utils import call_and_cache

company_url_template = Template(
    'https://api.intrinio.com/companies?identifier=${symbol}')

symbols_list_name = 'dow30'

symbols = open('../data/%s_symbols.lst' % symbols_list_name).read().split()

symbol2sic = []

for symbol in symbols:


    url = company_url_template.substitute(symbol=symbol)

    data_json = call_and_cache(url)

    try:
        symbol2sic.append((symbol, data_json['sic']))
    except KeyError:
        print("KeyError in data_json[sic], probably no info about %s is available."  % symbol)

with open('../data/%s_sic.txt' % symbols_list_name, 'w') as f:
    f.write('\n'.join(['%s,%s' % (s, sic) for s, sic in symbol2sic]))
