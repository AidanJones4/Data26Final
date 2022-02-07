import boto3
import numpy as np
import pandas as pd
from bson.json_util import loads
import os


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
            self.data_array.append(data)

        self.dataframe = pd.DataFrame(self.data_array)

    def csv_dataframe(self):
        """
        Creates dataframe from each csv file then concatenates along rows
        """
        frames = []
        for file in self.file_names:
            data_obj = self.client.get_object(Bucket=self.bucket_name, Key=file)["Body"]
            frames.append(pd.read_csv(data_obj))
        self.dataframe = pd.concat(frames, axis=0, ignore_index=True)

    def create_dataframe(self):

        if self.filetype == "json":
            self.json_dataframe()

        elif self.filetype == "csv":
            self.csv_dataframe()

        elif self.filetype == "txt":
            # txt data handling code
            pass

    def write_data(self):
        self.dataframe.to_json(self.local_filename)

    def load_local_dataframe(self):
        try:
            self.dataframe = pd.read_json(self.local_filename)
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

    # def create_junction_tables(self):









