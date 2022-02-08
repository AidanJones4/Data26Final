import boto3
import numpy as np
import pandas as pd
from bson.json_util import loads
import os
from datetime import datetime


class Pipeline:

    def __init__(self, bucket_name, folder, filetype, local_filename):
        self.bucket_name = bucket_name
        self.folder = folder
        self.client = boto3.client("s3")
        self.filetype = filetype
        self.file_names = []
        self.data_array = []
        self.local_filename = local_filename
        self.dataframe = pd.DataFrame()

        self.attributes = {}
        self.attribute_tables = []

    # Extract Methods
    def populate_filenames(self):
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name)
        for page in pages:
            for name in page["Contents"]:
                if name["Key"].startswith(f"{self.folder}/") and name["Key"].endswith(f".{self.filetype}"):
                    self.file_names.append(name["Key"])

    def json_dataframe(self):
        """
        Creates list of all json data (list of dictionaries), then creates dataframe from the list
        """
        for file in self.file_names:
            data_obj = self.client.get_object(Bucket=self.bucket_name, Key=file)["Body"]
            data = loads(data_obj.next())
            data["date"] = datetime.strptime(data["date"].replace("/", ""), "%d%m%Y").strftime("%Y/%m/%d")
            self.data_array.append(data)

        self.dataframe = pd.DataFrame(self.data_array)

    def csv_dataframe(self):
        """
        Creates dataframe from each csv file then concatenates along rows
        """
        frames = []
        start_dates = []
        course_names = []
        for file in self.file_names:
            data_obj = self.client.get_object(Bucket=self.bucket_name, Key=file)["Body"]
            frames.append(pd.read_csv(data_obj))
            if self.folder == 'Academy':
                for i in range(frames[-1].shape[0]):
                    start_dates.append(file[-14:-4].replace("-", "/"))
                    course_names.append(file.split('/')[1][:-15])

        self.dataframe = pd.concat(frames, axis=0, ignore_index=True)

        if self.folder == 'Academy':
            self.dataframe['start_date'] = pd.Series(start_dates)
            self.dataframe['course_names'] = pd.Series(course_names)


    def txt_dataframe(self):
        #loop through all file names
        for file in self.file_names:
            #get data, split into lines
            data_obj = self.client.get_object(Bucket=self.bucket_name,Key=file)["Body"].read().decode('utf-8')
            # print(data_obj)
            lines = data_obj.splitlines()
            
            #get academy and date for each file
            academy = lines[1][:lines[1].index(" ")].strip()
            date = lines[0][lines[0].index(" "):].strip()
            new_date = datetime.strptime(date,"%d %B %Y").strftime('%Y-%m-%d')

            # print(new_date)
            
            #loop through other lines and get data
            for i in range(lines.index("")+1, len(lines)):
                current_line = lines[i]
                names_txt = current_line[0:int(current_line.index(" - "))].title().strip()
                psychometrics_score = current_line[current_line.index(": ")+1:current_line.index(",")].strip()
                presentation_score = current_line[-5:].strip()

                #append data to array
                self.data_array.append({"name": names_txt, "psychometrics_score": psychometrics_score,
                     "presentation_score": presentation_score, "date": new_date, "academy": academy})
        
        #make dataframe from array
        self.dataframe = pd.DataFrame(data=self.data_array)

    def combine_date_columns(self):
        day = self.dataframe['invited_date'].map(lambda x: str(int(x)), na_action='ignore')
        month_yr = self.dataframe['month'].map(lambda x: x.strip(), na_action='ignore')
        date = pd.to_datetime(day + month_yr)
        self.dataframe.drop(['invited_date', 'month'], axis=1, inplace=True)
        self.dataframe['invited_date'] = pd.Series(date).map(lambda x: str(x).split(" ")[0].replace("-", "/")).map(lambda x: None if x == 'NaT' else x)

    def fix_phone_number(self):
        self.dataframe['phone_number'] = self.dataframe['phone_number'].map(lambda x: str("".join(x.replace("  ", "").replace("-", "").replace(" ", "").replace("(", " ").replace(")", "").split())), na_action='ignore')

    def combine_address_columns(self):
        address = self.dataframe['address']
        city = self.dataframe['city']
        postcode = self.dataframe['postcode']
        full_address = (address + ', ' + city + ', ' + postcode)
        self.dataframe.drop(['address', 'city', 'postcode'], axis=1, inplace=True)
        self.dataframe['full_address'] = pd.Series(full_address).map(lambda x: None if x == 'NaN' else x)

    def talent_clean(self):
        self.combine_date_columns()
        self.fix_phone_number()
        self.combine_address_columns()
        self.dataframe.drop(["id"], axis=1, inplace=True)

    def create_dataframe(self):

        if self.filetype == "json":
            self.json_dataframe()

        elif self.filetype == "csv":
            self.csv_dataframe()
            if self.folder == "Talent":
                self.talent_clean()

        elif self.filetype == "txt":
            self.txt_dataframe()

    def write_data(self):
        self.dataframe.to_json(self.local_filename)

    def load_local_dataframe(self):
        try:
            self.dataframe = pd.read_json(self.local_filename, dtype={"phone_number": str},
                                          convert_dates=["date", "start_date", "invited_date", "dob"])

        except FileNotFoundError:
            print(f"{self.local_filename} does not exist in local directory.")
            return None

    def extract_from_s3(self):
        self.populate_filenames()
        self.create_dataframe()
        self.write_data()

    def extract(self, force=False):
        try:
            if os.stat(self.local_filename).st_size == 0 or force:
                self.extract_from_s3()
            self.load_local_dataframe()
        except FileNotFoundError:
            self.extract_from_s3()
            self.load_local_dataframe()

    # Transform Methods

    def remove_duplicates(self):
        if self.dataframe.empty:
            self.load_local_dataframe()

        dup_mask = self.dataframe.applymap(lambda x: str(x)).duplicated()
        dup_rows = self.dataframe[dup_mask]
        self.dataframe = self.dataframe[dup_mask.map(lambda x: not x)]

        return dup_rows

    def list_attributes(self):
        """
        :return: Dictionary of attributes. Each key corresponds to a column that needs to be atomized.
                 Dictionary values contain list of unique values present in column
        """
        for col in self.dataframe:
            self.attributes[col] = []
            for val in self.dataframe[col]:
                if type(val) == list:
                    for elt in val:
                        if elt not in self.attributes[col]:
                            self.attributes[col].append(elt)
                elif type(val) == dict:
                    for key in val.keys():
                        if key not in self.attributes[col]:
                            self.attributes[col].append(key)
            if not self.attributes[col]:
                self.attributes.pop(col)
        return self.attributes

    def create_attribute_tables(self):

        """
        Crates separate dataframe for columns needing atomizing
        """
        for category in self.attributes:
            attribute_dataframe = pd.DataFrame({f"{category}": self.attributes[category]})
            attribute_dataframe[f"{category}_id"] = attribute_dataframe.index
            attribute_dataframe.to_json(f"{category}.json")
            self.attribute_tables.append(attribute_dataframe)

class Transformer:

    def __init__(self, candidates_sparta, candidates, academy, sparta_day, output_filepath):
        self.candidates_sparta = candidates_sparta
        self.candidates = candidates
        self.academy = academy
        self.sparta_day = sparta_day
        self.output_filepath = output_filepath
        self.big_table = pd.DataFrame()
        self._create_big_table()

        self.candidates_table = pd.DataFrame()
        #csv group
        self.benchmarks_table = pd.DataFrame()
        self.sparta_day_results_table =pd.DataFrame()
        self.sparta_day_table = pd.DataFrame()

    def remove_duplicates(self,df):
        dup_mask = df.applymap(lambda x: str(x)).duplicated()
        return df[dup_mask.map(lambda x: not x)]

    def _create_big_table(self):
        self.candidates_sparta.rename(columns={'date': 'invited_date'}, inplace=True)
        self.sparta_day.rename(columns={'date': 'invited_date'}, inplace=True)

        big_table = pd.merge(self.candidates_sparta, self.candidates,
                                                on=["name", "invited_date"], how='outer')
        big_table = pd.merge(big_table, self.academy,
                                                on=["name"], how='outer')
        big_table = pd.merge(big_table, self.sparta_day,
                                                on=["name", "invited_date"], how='outer')

        big_table_drop_dupes = self.remove_duplicates(big_table).copy()
        big_table_drop_dupes.reset_index(inplace=True)
        big_table_drop_dupes.drop("index", axis=1, inplace=True)
        big_table_drop_dupes["candidate_id"] = big_table_drop_dupes.index.map(lambda x: x + 10001)

        self.big_table = big_table_drop_dupes



    def create_candidates_table(self):
        self.candidates_table = self.big_table[["candidate_id", "name", "gender", "dob", "email", "full_address",
                                                     "phone_number", "uni", "degree", "invited_date", "invited_by",
                                                      "geo_flex", "course_interest"]].copy()
        print(self.candidates_table)

    def create_interview_table(self):
        pass

    def create_benchmarks_table(self):
        self.benchmarks_table = self.big_table[
            ['candidate_id', 'Analytic_W1', 'Independent_W1', 'Determined_W1', 'Professional_W1', 'Studious_W1',
             'Imaginative_W1', 'Analytic_W2', 'Independent_W2', 'Determined_W2', 'Professional_W2', 'Studious_W2',
             'Imaginative_W2', 'Analytic_W3', 'Independent_W3', 'Determined_W3', 'Professional_W3', 'Studious_W3',
             'Imaginative_W3', 'Analytic_W4', 'Independent_W4', 'Determined_W4', 'Professional_W4', 'Studious_W4',
             'Imaginative_W4', 'Analytic_W5', 'Independent_W5', 'Determined_W5', 'Professional_W5', 'Studious_W5',
             'Imaginative_W5', 'Analytic_W6', 'Independent_W6', 'Determined_W6', 'Professional_W6', 'Studious_W6',
             'Imaginative_W6', 'Analytic_W7', 'Independent_W7', 'Determined_W7', 'Professional_W7', 'Studious_W7',
             'Imaginative_W7', 'Analytic_W8', 'Independent_W8', 'Determined_W8', 'Professional_W8', 'Studious_W8',
             'Imaginative_W8', 'Analytic_W9', 'Independent_W9', 'Determined_W9', 'Professional_W9', 'Studious_W9',
             'Imaginative_W9', 'Analytic_W10', 'Independent_W10', 'Determined_W10', 'Professional_W10', 'Studious_W10',
             'Imaginative_W10']].copy()
        self.benchmarks_table.dropna(subset=['Analytic_W1'], inplace=True)

        melt = pd.melt(self.benchmarks_table, id_vars=['candidate_id'])
        val = melt['variable'].str.split('_')
        melt['benchmarks'] =val.str.get(0)
        melt['week'] = val.str.get(1)
        melt["week"] = melt["week"].map(lambda x: x.replace("W", ""))
        melt.drop(columns='variable', inplace=True)
        melt.rename(columns= {"value": "score"}, inplace=True)
        melt.dropna(subset=["score"], inplace=True)
        self.benchmarks_table = melt
        self.benchmarks_table = self.benchmarks_table[['candidate_id','benchmarks', "week", "score"]].copy()
        # print(melt)
        self.benchmarks_table['score'] = self.benchmarks_table['score'].astype('int64')
        print(self.benchmarks_table)

    def create_sparta_day_table(self):

        self.sparta_day_table = self.big_table[['academy', 'invited_date']].copy()
        self.sparta_day_table['sparta_day_id'] = self.sparta_day_table.index + 1
        self.sparta_day_table = self.sparta_day_table[['sparta_day_id','academy', 'invited_date']].copy()
        self.sparta_day_table.dropna(subset=['academy', 'invited_date'], inplace=True)
        self.sparta_day_table.drop_duplicates(subset=['academy', 'invited_date'], inplace=True)
        self.sparta_day_table.reset_index()

        print(self.sparta_day_table.head())
        print(self.sparta_day_table.shape)



    def create_sparta_day_results_table(self):

        self.sparta_day_results_table = pd.merge(self.big_table[["candidate_id", 'psychometrics_score', 'presentation_score', 'academy', 'invited_date']],
                                                 self.sparta_day_table, on=['academy','invited_date'], how='left')
        self.sparta_day_results_table = self.sparta_day_results_table[["candidate_id", 'sparta_day_id', 'psychometrics_score', 'presentation_score']].copy()
        self.sparta_day_results_table.dropna(subset=['sparta_day_id', 'psychometrics_score', 'presentation_score'], inplace=True)
        self.sparta_day_results_table['sparta_day_id'] = self.sparta_day_results_table['sparta_day_id'].astype('int64')

        print(self.sparta_day_results_table.head())
        print(self.sparta_day_results_table.shape)

    #More methods...
















