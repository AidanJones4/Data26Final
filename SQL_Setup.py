import pyodbc
import json

server = 'localhost, 1433'
database = 'spartaGlobal'
username = 'SA'
password = 'Pa55word'
spartaGlobal = pyodbc.connect(('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password))

cursor = spartaGlobal.cursor()

#cursor.execute('CREATE TABLE Candidates(CandidateID INT)')

tables = json.load(open('SQL/tables.json', encoding = "utf8"))

for table in tables.keys():
    print(table)
    cursor.execute('DROP TABLE IF EXISTS ' + table)
    command = 'CREATE TABLE ' + table + '('
    for column in tables[table].keys():
        command += column + ' ' + tables[table][column] + ','
    command = command[: -1] + ')'
    cursor.execute(command)

cursor.commit()
cursor.close()
