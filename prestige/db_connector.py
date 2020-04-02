import pyodbc
import pandas as pd

# def get query function
def query_to_df(str_query):
    # establish db connection
    cnxn = pyodbc.connect("Driver={SQL Server};"
                      "Server=electra;"
                      "Database=pfsdb;"
                      "Trusted_Connection=yes;")
    # pull data into df
    df = pd.read_sql_query(str_query, cnxn) 
    # close connection
    cnxn.close()
    # return the df
    return df
