import pyodbc
import pandas as pd

# def read sql file function
def read_sql_file(str_file):
  # open qauery file
  query = open(str_file, 'r')
  # read the file
  str_query = query.read()
  # close query
  query.close()
  # return str_query
  return str_query

# def get query function
def query_to_df(str_query, server='electra', database='riskdb'):
    # define Python user-defined exceptions
    class Error(Exception):
      """Base class for other exceptions"""
      pass

    # error to raise if connecting to a non existent db
    class CantFindServerAndOrDatabaseError(Error):
      """Raised when server/databae combination not supported"""
      pass

    try:
      # logic for different engines
      if server == 'electra' and database == 'pfsdb':
        # establish db connection
        cnxn = pyodbc.connect("Driver={SQL Server};"
                              "Server=electra;"
                              "Database=pfsdb;"
                              "Trusted_Connection=yes;")
        # print message
        print('Successfully connected to electra: pfsdb.')
      elif server == 'electra' and database == 'riskdb':
        # establish db connection
        cnxn = pyodbc.connect("Driver={SQL Server};"
                              "Server=electra;"
                              "Database=riskdb;"
                              "Trusted_Connection=yes;")
        # print message
        print('Successfully connected to electra: riskdb')
      elif server == 'medusa' and database == 'pfsdb':
        # establish db connection
        cnxn = pyodbc.connect("Driver={SQL Server};"
                              "Server=medusa;"
                              "Database=pfsdb;"
                              "Trusted_Connection=yes;")
        # print message
        print('Successfully connected to medusa: pfsdb')
      elif server == 'medusa' and database == 'riskdb':
        # establish db connection
        cnxn = pyodbc.connect("Driver={SQL Server};"
                              "Server=medusa;"
                              "Database=riskdb;"
                              "Trusted_Connection=yes;")
        # print message
        print('Successfully connected to medusa: riskdb')
      else:
        raise CantFindServerAndOrDatabaseError
    except CantFindServerAndOrDatabaseError:
      print('Unsupported server/database combination.')

    # pull data into df
    df = pd.read_sql_query(str_query, cnxn) 
    # print message
    print(f'Successfully pulled query from {server}: {database} and stored in dataframe.')
    # close connection
    cnxn.close()
    # print message
    print('Connection closed.')
    # return the df
    return df
