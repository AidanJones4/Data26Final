import boto3
import numpy as np
import pandas as pd
from bson.json_util import loads
import json
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
        self.attributes = {}
        self.attribute_tables = []
        self.interview_table = pd.DataFrame()
        self.candidates_table = pd.DataFrame()
        self.tech_skills_table = pd.DataFrame()
        self.tech_junction_table = pd.DataFrame()
        self.quality_table = pd.DataFrame()
        self.quality_junction_table = pd.DataFrame()

    def list_attributes(self):
        """
        :return: Dictionary of attributes. Each key corresponds to a column that needs to be atomized.
                 Dictionary values contain list of unique values present in column
        """
        for col in self.big_table:
            self.attributes[col] = []
            for val in self.big_table[col]:

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

    def create_tech_skill_tables(self):
        big_table_nonan = self.big_table.dropna(subset=["tech_self_score"])
        big_table_numpy = big_table_nonan.to_numpy()

        tech_skills_df = pd.read_json("tech_self_score.json")
        tech_skills_df["tech_self_score_id"] = tech_skills_df["tech_self_score_id"].map(lambda x: x+1)

        self.tech_skills_table = tech_skills_df.copy()
        self.tech_skills_table.columns = ["skill_name", "tech_skill_id"]


        tech_skills_df.index = tech_skills_df["tech_self_score"]
        tech_skills_df.drop(["tech_self_score"], axis=1, inplace=True)
        tech_skills_df.T.to_json("tech_skills.json", orient="records")
        with open("tech_skills.json") as f:
            tech_skills_dict = json.load(f)

        jt_tech_skills = []
        for each in big_table_numpy:
            if each[2] is not None:
                for x,y in each[2].items():
                    jt_tech_skills.append([each[-2],tech_skills_dict[0][x],y])
        jt_tech_skills_df = pd.DataFrame(jt_tech_skills)

        jt_tech_skills_df.columns = ["candidate_id","skill_id","score"]
        self.tech_junction_table = jt_tech_skills_df

    def create_quality_junction(self):
        big_table_nonan = self.big_table.dropna(subset=["qualities"])
        qualities_df = pd.read_json("qualities.json")
        self.quality_table = qualities_df.copy()
        qualities_df.index = qualities_df["qualities"]
        qualities_df.drop("qualities", inplace=True, axis=1)

        qualities_df.T.to_json("quality.json", orient="records")

        with open("quality.json") as f:
            quality_dict = json.load(f)

        big_table_np = big_table_nonan.to_numpy()

        jt_qualities = []

        for each in big_table_np:
            for quality in each[-1]:
                jt_qualities.append([each[-2], quality_dict[0][quality]])

        jt_qualities_df = pd.DataFrame(jt_qualities)
        jt_qualities_df.columns = ["Candidate ID", "Quality ID"]

        self.quality_junction_table = jt_qualities_df

    def create_quality_table(self):
        strengths = self.attributes["strengths"]
        self.quality_table["is_strengths"] = self.quality_table["qualities"].map(lambda x: 1 if x in strengths else 0)

    def remove_duplicates(self, df):
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
        big_table_drop_dupes["qualities"] = big_table_drop_dupes["strengths"] + big_table_drop_dupes["weaknesses"]

        self.big_table = big_table_drop_dupes

    def create_candidates_table(self):
        self.candidates_table = self.big_table[["candidate_id", "name", "gender", "dob", "email", "full_address",
                                                     "phone_number", "uni", "degree", "invited_date", "invited_by",
                                                      "geo_flex", "course_interest"]].copy()

    def create_interview_table(self):
        self.interview_table = self.big_table[["candidate_id", "invited_date", "self_development",
                                               "geo_flex", "result"]].copy()
        self.interview_table.dropna(axis=0,subset=["invited_date", "self_development",
                                                            "geo_flex", "result"], how="all", inplace=True)
        self.interview_table.reset_index(inplace=True)
        self.interview_table.drop(["index"], axis=1, inplace=True)

        self.interview_table.dropna(subset=["self_development"], axis=0, inplace=True)

        join = pd.merge(self.interview_table, self.candidates_table, how="inner")

    def create_benchmarks_table(self):
        pass


















