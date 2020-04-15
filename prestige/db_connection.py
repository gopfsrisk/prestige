import pyodbc
import pandas as pd

# def get query function
def query_to_df(str_query, db_engine='electra'):
    # logic for different engines
    if db_engine == 'electra':
      # establish db connection
      cnxn = pyodbc.connect("Driver={SQL Server};"
                            "Server=electra;"
                            "Database=pfsdb;"
                            "Trusted_Connection=yes;")
    elif db_engine == 'medusa':
      # establish db connection
      cnxn = pyodbc.connect("Driver={SQL Server};"
                            "Server=medusa;"
                            "Database=pfsdb;"
                            "Trusted_Connection=yes;")
    else:
      print('ERROR! Selected engine not found. Supported engines include "electra" and "medusa"')
    # pull data into df
    df = pd.read_sql_query(str_query, cnxn) 
    # close connection
    cnxn.close()
    # return the df
    return df
