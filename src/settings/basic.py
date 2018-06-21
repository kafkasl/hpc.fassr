import os
import logging
import sys

# Screening config
CURRENT_YEAR = 2017

# Paths configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FUNDAMENTAL_DB_PATH = os.path.join(PROJECT_ROOT, "../../data/fundamental-data.db")

# Criteria settings
GRAHAM_CRITERIA = "graham"
SUPPORTED_CRITERIA = [GRAHAM_CRITERIA]

# Logger

logging.basicConfig(level=logging.DEBUG,
                    format=' %(levelname)s: %(message)s',
                    stream=sys.stdout)
# File logger
# output = os.path.join(PROJECT_ROOT, "../../log/log.out")
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     filename=output,
#                     filemode='w')
