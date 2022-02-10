from pipeline import *
from docker_setup import *

# Pandas display settings
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

# Extract and print the data from the json files
print("\n\njson Dataframe:")
json_pip = Extractor("data-26-final-project-files", "Talent", "json", "candidates_sparta_data.json")
json_pip.extract()
print(json_pip.dataframe.head())

# Extract and print the data from the csv files in the Talent folder
print("\n\ncsv Talent Dataframe:")
csv_talent_pip = Extractor("data-26-final-project-files", "Talent", "csv", "candidate_data.json")

csv_talent_pip.extract()
print(csv_talent_pip.dataframe.head())


# Extract and print the data from the csv files in the Academy folder
print("\n\ncsv Academy Dataframe:")
csv_academy_pip = Extractor("data-26-final-project-files", "Academy", "csv", "academy_data.json")
csv_academy_pip.extract()
print(csv_academy_pip.dataframe.head())

# Extract and print the data from the txt files
print("\n\ntxt Dataframe:")
txt_pip = Extractor("data-26-final-project-files", "Talent", "txt", "sparta_day_data.json")
txt_pip.extract()
print(txt_pip.dataframe.head())

# Instantiate the transform and load classes
transform = Transformer(json_pip.dataframe, csv_talent_pip.dataframe, csv_academy_pip.dataframe, txt_pip.dataframe)
to_docker = dockerSetUp()

# Run the transformations to create tables according to ERD
transform.create_tables()
transform.name_tables()
transform.upload_tables_to_s3()

# Extract a list of all pandas dataframes
df_list = transform.list_tables()

# Run the load functions to upload dataframes to Docker database
to_docker.pandas_to_SQL(df_list)
to_docker.close_cursor()
