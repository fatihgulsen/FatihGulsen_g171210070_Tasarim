import pandas as pd
import os
import sqlalchemy as sa
import pyodbc

import urllib

def sqlcol(data):
    dtypedict = {}
    for i, j in zip(data.columns, data.dtypes):
        if "object" in str(j):
            dtypedict.update({i: sa.types.NVARCHAR})

        if "datetime" in str(j):
            dtypedict.update({i: sa.types.DateTime()})

        if "float" in str(j):
            dtypedict.update({i: sa.types.NVARCHAR(length=255)})

        if "int" in str(j):
            dtypedict.update({i: sa.types.NVARCHAR(length=255)})

    return dtypedict


params = "Driver={SQL Server Native Client 11.0};"\
                      "Server=localhost\SQLEXPRESS;"\
                      "Database=TezData;"\
                      "Trusted_Connection=yes;"
## VERITABANI ISMI GIRILECEK YER

dosya_dir = r'C:\Users\fatih\Desktop\TezData\pythonAtılacak'  # DOSYA KAYNAGI




params = urllib.parse.quote_plus(params)

engine = sa.create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

os.chdir(dosya_dir)
dosya_list = os.listdir()

excel_list = []
access_list = []
txt_list = []

for dosya in dosya_list:
    if dosya.endswith('.xlsx'):
        excel_list.append(dosya)
        data = pd.read_excel(dosya)
        dosya = dosya.replace('.xlsx', '')
        dtypes_dict = sqlcol(data)
        data.to_sql(dosya, con=engine, if_exists='replace', index=False, dtype=dtypes_dict)

    elif dosya.endswith('.accdb') or dosya.endswith('.mdb'):
        access_list.append(dosya)
        klasor = os.getcwd()
        conn_string = (r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
                              r"DBQ=%s\\%s;"%(klasor, dosya))
        conn = pyodbc.connect(conn_string)
        table_name = ''
        crsr = conn.cursor()

        for table_info in crsr.tables(tableType='TABLE'):

            table_name = table_info.table_name
            data = pd.read_sql_query('select * from %s' % table_name, conn)
            dtypes_dict = sqlcol(data)
            if dosya.endswith('.accdb'):
                dosya = dosya.replace('.accdb', '')
            elif dosya.endswith('.mdb'):
                dosya = dosya.replace('.mdb', '')
            data.to_sql(dosya+'_'+table_name, con=engine, if_exists='replace', index=False, dtype=dtypes_dict)
    elif dosya.endswith('.txt'):
        excel_list.append(dosya)
        data = pd.read_csv(dosya,sep='\t')
        dosya = dosya.replace('.txt', '')
        dtypes_dict = sqlcol(data)
        data.to_sql(dosya, con=engine, if_exists='replace', index=False, dtype=dtypes_dict)





