import os
import pyodbc
from pprint import pprint as pp
import json
import pandas as pd
import sqlalchemy

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
        self.engine = sqlalchemy.create_engine(f"mssql+pyodbc://{self.username}:{self.password}@localhost:1433/{self.database}?driver=ODBC+Driver+17+for+SQL+Server")

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
        #print(self.my_list)
        return self.my_list


    def pandas_to_SQL(self, interview, candidates, tech_skill, tech_skill_score_j,
                      quality, interview_quality_j, benchmark, sparta_day,
                      sparta_day_results, trainer, course, candidate_course_j):
        self._interview_to_sql(interview)
        self._candidates_to_sql(candidates)
        self._tech_skill_to_sql(tech_skill)
        self._tech_skill_score_j_to_sql(tech_skill_score_j)
        self._quality_to_sql(quality)
        self._interview_quality_j_to_sql(interview_quality_j)
        self._benchmark_to_sql(benchmark)
        self._sparta_day_to_sql(sparta_day)
        self._sparta_day_results_to_sql(sparta_day_results)
        self._trainer_to_sql(trainer)
        self._course_to_sql(course)
        self._candidate_course_j_to_sql(candidate_course_j)

    def _interview_to_sql(self, df):
        print("Interview")
        values = "?, " * 4 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2], row[3], row[4])
        self.docker_data26project.commit()

    def _candidates_to_sql(self, df):
        print("Candidate")
        values = "?, "*12 +"?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2], row[3], row[4],
                                    row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12])

        self.docker_data26project.commit()

    def _tech_skill_to_sql(self, df):
        print("Tech Skill")
        values = "?, " + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1])
        self.docker_data26project.commit()

    def _tech_skill_score_j_to_sql(self, df):
        print("Tech Skill Junction")
        values = "?, " * 2 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2])
        self.docker_data26project.commit()

    def _quality_to_sql(self, df):
        print("Quality")
        values = "?, " * 2 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2])
        self.docker_data26project.commit()

    def _interview_quality_j_to_sql(self, df):
        print("Interview Quality Junction")
        values = "?, " + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1])
        self.docker_data26project.commit()

    def _benchmark_to_sql(self, df):
        print("Benchmark")
        values = "?, " * 3 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2], row[3])
        self.docker_data26project.commit()

    def _sparta_day_to_sql(self, df):
        print("Sparta Day")
        values = "?, " * 2 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2])
        self.docker_data26project.commit()

    def _sparta_day_results_to_sql(self, df):
        print("Sparta Day Results")
        values = "?, " * 3 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2], row[3])
        self.docker_data26project.commit()

    def _trainer_to_sql(self, df):
        print("Trainer")
        values = "?, " + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1])
        self.docker_data26project.commit()

    def _course_to_sql(self, df):
        print("Course")
        values = "?, " * 3 + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1], row[2], row[3])
        self.docker_data26project.commit()

    def _candidate_course_j_to_sql(self, df):
        print("Candidate Course Junction")
        values = "?, " + "?"
        df.reset_index(drop=True, inplace=True)
        df_list = df.values.tolist()
        for row in df_list:
            self.cursor.execute(f"INSERT INTO {df.name} VALUES({values})", row[0], row[1])


    def close_cursor(self):
        self.cursor.commit()
        self.cursor.close()


if __name__ == "main":
    yi = dockerSetUp()
    yi.get_column_names(yi.table_names[0])
    yi.add_table_columns()
    yi.check_table()
    yi.all_tables_upload()
    yi.close_cursor()
