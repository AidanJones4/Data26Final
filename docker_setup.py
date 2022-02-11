import os
import pyodbc
import json
from termcolor import colored

class dockerSetUp:
    """
    This class connects to the docker container. And creates tables and column names according to the tables.json.
    """

    def __init__(self):
        # Setting up link between Python & SQL
        self.server = 'localhost, 1433'
        self.database = 'spartaGlobal'
        self.username = 'SA'
        self.password = 'Pa55word'
        self.docker_data26project = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + self.server +
                                                   ';UID=' + self.username + ';PWD=' + self.password, autocommit=True)
        self.cursor = self.docker_data26project.cursor()
        self.tables = json.load(open('SQL/tables.json')) #Open json file
        self.my_list = []
        self.table_names = list(self.tables.keys())
        self.df_files = os.listdir("output_tables")

    def create_database(self):
        sql_drop = (
            "DECLARE @sql AS NVARCHAR(MAX);"
            "SET @sql = 'DROP DATABASE IF EXISTS ' + QUOTENAME(?);"
            "EXEC sp_executesql @sql"
        )
        sql_create = (
            "DECLARE @sql AS NVARCHAR(MAX);"
            "SET @sql = 'CREATE DATABASE ' + QUOTENAME(?);"
            "EXEC sp_executesql @sql"
        )
        self.cursor.execute(sql_drop, self.database)
        self.cursor.execute(sql_create, self.database)
        self.cursor.close()

    def connect_to_database(self):
        self.docker_data26project = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + self.server +
                                                   ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)  # connecter
        self.cursor = self.docker_data26project.cursor()  # pyodbc.Cursor object

    #Output the column names
    def get_column_names(self, table_name):
        return list(self.tables.get(table_name).keys())

    # Creates a new table and adds table columns
    def add_table_columns(self):

        for table in self.tables.keys():
            self.cursor.execute('DROP TABLE IF EXISTS ' + table) #Drop table if exists
            command = 'CREATE TABLE ' + table + '(' #SQL query to create table with table columns
            for column in self.tables[table].keys():
                command += column + ' ' + self.tables[table][column] + ','
            command = command[: -1] + ')'
            self.cursor.execute(command)
        return None

    # Checks jf candidate table columns have been added
    def check_table(self):
        self.cursor.execute("SELECT * FROM Candidates")
        columns = [columns[0] for columns in self.cursor.description] #Table Column Titles
        return columns

    # Upload table columns into local SQL database
    def all_tables_upload(self):
        for a in self.cursor.tables():
            if a.table_name == "trace_xe_action_map":
                break
            self.my_list.append(a.table_name) # Add table columns to list
        return self.my_list

    def pandas_to_SQL(self, df_list):
        for df in df_list:
            df.reset_index(drop=True, inplace=True)
            print(f"Uploading to table: {df.name}...")
            for index, row in df.iterrows():
                command = f'INSERT INTO {df.name} VALUES ('
                for column in df.keys():
                    if type(row[f'{column}']) == str:
                        entry = row[f'{column}'].replace("'", "''")
                        command = command + f"'{entry}',"
                    else:
                        command = command + f"'{row[f'{column}']}',"
                command = command[: -1] + ')'
                command = command.replace("'NaT'", 'NULL').replace("'NaN'", 'NULL').replace("'None'", 'NULL')

                self.cursor.execute(command)
        print(colored("Upload Completed Successfully", "green"))

    def close_cursor(self):
        self.cursor.commit()
        self.cursor.close()


if __name__ == "__main__":
    yi = dockerSetUp()
    yi.create_database()
    yi.connect_to_database()
    yi.get_column_names(yi.table_names[0])
    yi.add_table_columns()
    yi.check_table()
    yi.all_tables_upload()
    yi.close_cursor()
