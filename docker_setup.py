import os

import pyodbc
import json

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
        self.docker_data26project = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)  # connecter
        self.cursor = self.docker_data26project.cursor()  # pyodbc.Cursor object

        self.tables = json.load(open('SQL/tables.json')) #Open json file
        self.my_list = []
        self.table_names = list(self.tables.keys())
        self.df_files = os.listdir("output_tables")

    #Output the column names
    def get_column_names(self, table_name):
        return list(self.tables.get(table_name).keys())

    # Creates a new table and adds table columns
    def add_table_columns(self):

        for table in self.tables.keys():
            self.cursor.execute('DROP TABLE IF EXISTS ' + table) # Drop table if exists
            command = 'CREATE TABLE ' + table + '(' # SQL query to create table with table columns
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
            print(f"{df.name}")
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


    def close_cursor(self):
        self.cursor.commit()
        self.cursor.close()


if __name__ == "__main__":
    yi = dockerSetUp()
    yi.get_column_names(yi.table_names[0])
    yi.add_table_columns()
    yi.check_table()
    yi.all_tables_upload()
    yi.close_cursor()

