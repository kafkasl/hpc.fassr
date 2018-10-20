import base64
import json
import os
import pickle
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from settings.basic import (logging, CACHE_ENABLED, CACHE_PATH,
                            intrinio_username,
                            intrinio_password)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def to_df(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)

    # df.set_index(['year', 'quarter'], inplace=True)
    # df.sort_index(inplace=True)

    return df


def call_and_cache(url: str, **kwargs) -> dict:
    """
    Calls the URL with GET method if the url file is not cached
    :param url: url to retrieve
    :param kwargs: specify no-cache
    :return: json.loads of the response (or empty dict if error)
    """
    url_parsed = urlparse(url)

    cached_file = os.path.join(CACHE_PATH,
                               url_parsed.netloc + url_parsed.path + "/" +
                               base64.standard_b64encode(
                                   url_parsed.query.encode()).decode())

    if not os.path.exists(os.path.dirname(cached_file)):
        os.makedirs(os.path.dirname(cached_file))

    try:
        no_cache = kwargs['no-cache']
    except KeyError:
        no_cache = False

    data_json = {}
    if CACHE_ENABLED and os.path.exists(cached_file) and not no_cache:
        logging.debug(
            "Data was present in cache and cache is enabled, loading: %s for %s" %
            (cached_file, url))
        with open(cached_file, 'r') as f:
            data_json = json.loads(f.read())
    else:
        logging.info(
            "Data was either not present in cache or it was disabled calling request: %s" % url)
        r = requests.get(url, auth=HTTPBasicAuth(intrinio_username,
                                                 intrinio_password))

        if r.status_code != 200:
            logging.error(
                "Request status was: %s for URL: %s" % (r.status_code, url))
            return data_json

        data_json = json.loads(r.text)

        if 'data' in data_json.keys() and not len(data_json['data']) > 0:
            logging.debug("Data field is empty.\nRequest URL: %s" % (url))

        with open(cached_file, 'w') as f:
            f.write(json.dumps(data_json))
            logging.debug(
                "Successfully cached url: %s to %s" % (url, cached_file))

    return data_json


# calculation_tags = ['revenuegrowth',
#                     'nopat',
#                     'nopatmargin',
#                     'investedcapital',
#                     'investedcapitalturnover',
#                     'investedcapitalincreasedecrease',
#                     'freecashflow',
#                     'netnonopex',
#                     'netnonopobligations',
#                     'ebit',
#                     'depreciationandamortization',
#                     'ebitda',
#                     'capex',
#                     'dfcfnwc',
#                     'dfnwc',
#                     'nwc',
#                     'debt',
#                     'ltdebtandcapleases',
#                     'netdebt',
#                     'totalcapital',
#                     'bookvaluepershare',
#                     'tangbookvaluepershare',
#                     'marketcap',
#                     'enterprisevalue',
#                     'pricetobook',
#                     'pricetotangiblebook',
#                     'pricetorevenue',
#                     'pricetoearnings',
#                     'dividendyield',
#                     'earningsyield',
#                     'evtoinvestedcapital',
#                     'evtorevenue',
#                     'evtoebitda',
#                     'evtoebit',
#                     'evtonopat',
#                     'evtoocf',
#                     'evtofcff',
#                     'ebitdagrowth',
#                     'ebitgrowth',
#                     'nopatgrowth',
#                     'netincomegrowth',
#                     'epsgrowth',
#                     'ocfgrowth',
#                     'fcffgrowth',
#                     'investedcapitalgrowth',
#                     'revenueqoqgrowth',
#                     'ebitdaqoqgrowth',
#                     'ebitqoqgrowth',
#                     'nopatqoqgrowth',
#                     'netincomeqoqgrowth',
#                     'epsqoqgrowth',
#                     'ocfqoqgrowth',
#                     'fcffqoqgrowth',
#                     'investedcapitalqoqgrowth',
#                     'grossmargin',
#                     'ebitdamargin',
#                     'operatingmargin',
#                     'ebitmargin',
#                     'profitmargin',
#                     'costofrevtorevenue',
#                     'sgaextorevenue',
#                     'rdextorevenue',
#                     'opextorevenue',
#                     'taxburdenpct',
#                     'interestburdenpct',
#                     'efftaxrate',
#                     'assetturnover',
#                     'arturnover',
#                     'invturnover',
#                     'faturnover',
#                     'apturnover',
#                     'dso',
#                     'dio',
#                     'dpo',
#                     'ccc',
#                     'finleverage',
#                     'leverageratio',
#                     'compoundleveragefactor',
#                     'ltdebttoequity',
#                     'debttoequity',
#                     'roic',
#                     'nnep',
#                     'roicnnepspread',
#                     'rnnoa',
#                     'roe',
#                     'croic',
#                     'oroa',
#                     'roa',
#                     'noncontrollinginterestsharingratio',
#                     'roce',
#                     'divpayoutratio',
#                     'augmentedpayoutratio',
#                     'ocftocapex',
#                     'stdebttocap',
#                     'ltdebttocap',
#                     'debttototalcapital',
#                     'preferredtocap',
#                     'noncontrolinttocap',
#                     'commontocap',
#                     'debttoebitda']
#
# cash_flow_statement_tags = ["netincome",
#                             "netincomecontinuing",
#                             "depreciationexpense",
#                             "amortizationexpense",
#                             "noncashadjustmentstonetincome",
#                             "increasedecreaseinoperatingcapital",
#                             "netcashfromcontinuingoperatingactivities",
#                             "netcashfromoperatingactivities",
#                             "purchaseofplantpropertyandequipment",
#                             "acquisitions",
#                             "otherinvestingactivitiesnet",
#                             "netcashfromcontinuinginvestingactivities",
#                             "netcashfrominvestingactivities",
#                             "repaymentofdebt",
#                             "repurchaseofcommonequity",
#                             "paymentofdividends",
#                             "issuanceofdebt",
#                             "issuanceofcommonequity",
#                             "otherfinancingactivitiesnet",
#                             "netcashfromcontinuingfinancingactivities",
#                             "netcashfromfinancingactivities",
#                             "effectofexchangeratechanges",
#                             "netchangeincash",
#                             "cashinterestpaid",
#                             "cashincometaxespaid"]
#
# balance_sheet_tags = ["cashandequivalents",
#                       "restrictedcash",
#                       "accountsreceivable",
#                       "netinventory",
#                       "prepaidexpenses",
#                       "totalcurrentassets",
#                       "netppe",
#                       "goodwill",
#                       "intangibleassets",
#                       "othernoncurrentassets",
#                       "othernoncurrentnonoperatingassets",
#                       "totalnoncurrentassets",
#                       "totalassets",
#                       "shorttermdebt",
#                       "accountspayable",
#                       "accruedexpenses",
#                       "othercurrentliabilities",
#                       "totalcurrentliabilities",
#                       "longtermdebt",
#                       "othernoncurrentliabilities",
#                       "totalnoncurrentliabilities",
#                       "totalliabilities",
#                       "commitmentsandcontingencies",
#                       "redeemablenoncontrollinginterest",
#                       "totalpreferredequity",
#                       "commonequity",
#                       "retainedearnings",
#                       "aoci",
#                       "totalcommonequity",
#                       "totalequity",
#                       "noncontrollinginterests",
#                       "totalequityandnoncontrollinginterests",
#                       "totalliabilitiesandequity",
#                       "noncurrentnotereceivables",
#                       "customerdeposits",
#                       "currentdeferredrevenue",
#                       "noncurrentdeferredrevenue"]
#
# income_statement_tags = ["operatingrevenue",
#                          "totalrevenue",
#                          "operatingcostofrevenue",
#                          "totalcostofrevenue",
#                          "totalgrossprofit",
#                          "sgaexpense",
#                          "rdexpense",
#                          "totaloperatingexpenses",
#                          "totaloperatingincome",
#                          "totalinterestexpense",
#                          "totalinterestincome",
#                          "otherincome",
#                          "totalotherincome",
#                          "totalpretaxincome",
#                          "incometaxexpense",
#                          "netincomecontinuing",
#                          "netincome",
#                          "netincometononcontrollinginterest",
#                          "netincometocommon",
#                          "weightedavebasicsharesos",
#                          "basiceps",
#                          "weightedavedilutedsharesos",
#                          "dilutedeps",
#                          "weightedavebasicdilutedsharesos",
#                          "basicdilutedeps"]