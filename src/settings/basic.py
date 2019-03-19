import os
import socket

# Paths configuration
SETTINGS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SETTINGS_FOLDER, '../..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
FUNDAMENTAL_DB_PATH = os.path.join(PROJECT_ROOT, "data/fundamental-data.db")
LOG_PATH = os.path.join(PROJECT_ROOT, "log")

# Screening config
DATE_FORMAT = "%Y-%m-%d"
# iio_symbols = open(os.path.join(DATA_PATH, "symbols.lst")).read().splitlines()

# Criteria settings
# GRAHAM_CRITERIA = "graham"
# SUPPORTED_CRITERIA = [GRAHAM_CRITERIA]
#
# GRAHAM criteria settings
# GRAHAM = {'year': 2017,
#           'revenue_limit': int(1.5e9)}

# Logger DISABLED BECAUSE PYCOMPSS DOES NOT SUPPORT IT....
# logFormatter = logging.Formatter(
#     "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# rootLogger = logging.getLogger()
#
# log_file = os.path.join(LOG_PATH, "log.out")
# fileHandler = logging.FileHandler(log_file)
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)
#
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)
# rootLogger.setLevel(logging.INFO)
debug = False

# Intrinio API
intrinio_username = '537b80f5966d25d2caaaba7e14adbd5d'
intrinio_password = 'c4daefffaf7ea875949468c0db69d256'

# Cache config
CACHE_ENABLED = True
CHECKPOINTING = True

if socket.gethostname() == 'Marginis':
    CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
    INTRINIO_CACHE_PATH = os.path.join(CACHE_PATH, "intrinio")

else:  # we are in MN4
    CACHE_PATH = "/gpfs/scratch/bsc19/compss/COMPSs_Sandbox/bsc19277/ydra/cache"
    INTRINIO_CACHE_PATH = os.path.join(CACHE_PATH, "intrinio")
    project = os.path.realpath(__file__).split('/')[-4]
    if 'ydra' not in project:
        print(
            "ydra keyword not present in comparison to [%s], probably comparing the wrong folder." % project)
    if project == 'ydra-test':
        CHECKPOINTING = False
