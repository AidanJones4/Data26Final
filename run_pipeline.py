from pipeline import *


# Pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


json_pip = Pipeline("data-26-final-project-files", "Talent", "json", "candidates_sparta_data.json")
json_pip.extract()
print(type(json_pip.dataframe['date'][0].date()))
print(json_pip.dataframe['date'][0].date())

print(f"Before {json_pip.dataframe.shape[0]}")
dups = json_pip.remove_duplicates()
print(dups)
print(f"After {json_pip.dataframe.shape[0]}")

json_pip.list_attributes()
json_pip.create_attribute_tables()

csv_pip = Pipeline("data-26-final-project-files", "Talent", "csv", "candidate_data.json")
csv_pip.extract()
print(csv_pip.dataframe.shape)












