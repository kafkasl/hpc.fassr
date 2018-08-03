import os
import logging
import sys

# Paths configuration
SETTINGS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SETTINGS_FOLDER, '../..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
FUNDAMENTAL_DB_PATH = os.path.join(PROJECT_ROOT, "data/fundamental-data.db")
LOG_PATH = os.path.join(PROJECT_ROOT, "log")

# Screening config
DATE_FORMAT = "%Y-%m-%d"
iio_symbols = open(os.path.join(DATA_PATH, "symbols.lst")).read().splitlines()

# Criteria settings
GRAHAM_CRITERIA = "graham"
SUPPORTED_CRITERIA = [GRAHAM_CRITERIA]

# GRAHAM criteria settings
GRAHAM = {'year': 2017,
          'revenue_limit': int(1.5e9)}

# Logger
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

log_file = os.path.join(LOG_PATH, "log.out")
fileHandler = logging.FileHandler(log_file)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.DEBUG)

#
# logging.basicConfig(level=logging.DEBUG,
#                     format=' %(levelname)s: %(message)s',
#                     stream=sys.stdout)
# File logger
# output = os.path.join(PROJECT_ROOT, "../../log/log.out")
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     filename=output,
#                     filemode='w')


# Intrinio API
intrinio_username = '537b80f5966d25d2caaaba7e14adbd5d'
intrinio_password = 'c4daefffaf7ea875949468c0db69d256'

# Cache config
CACHE_ENABLED = True
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INTRINIO_CACHE_PATH = os.path.join(CACHE_PATH, "intrinio")
