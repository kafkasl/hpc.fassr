#!/usr/bin/python
 
import sqlite3
from sqlite3 import Error
 
 
def _create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        logging.debug(e)
 
    return None
 
 
def _select_and_logging.debug(conn, query):
    """
    Selects and logging.debug the result of query on conn
    :param conn: the Connection object
    :param query: query to be performed
    :return:
    """
    cur = conn.cursor()
    cur.execute(query)
 
    rows = cur.fetchall()
 
    for row in rows:
        logging.debug(row)


def get_stocks():

 
def main():
    database = "../data/fundamental-data.db"
 
    # create a database connection
    conn = create_connection(database)
    with conn:
	query = "SELECT * FROM eco_cpi"
        select_and_logging.debug(conn, query)
 
if __name__ == '__main__':
         main()
