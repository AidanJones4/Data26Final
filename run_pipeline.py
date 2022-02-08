from pipeline import *


# Pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

print("\n\njson Dataframe:")
json_pip = Pipeline("data-26-final-project-files", "Talent", "json", "candidates_sparta_data.json")
json_pip.extract()
print(json_pip.dataframe)

print("\n\ncsv Talent Dataframe:")
csv_talent_pip = Pipeline("data-26-final-project-files", "Talent", "csv", "candidate_data.json")
csv_talent_pip.extract()
print(csv_talent_pip.dataframe)

print("\n\ncsv Academy Dataframe:")
csv_academy_pip = Pipeline("data-26-final-project-files", "Academy", "csv", "academy_data.json")
csv_academy_pip.extract()
print(csv_academy_pip.dataframe)

print("\n\ntxt Dataframe:")
txt_pip = Pipeline("data-26-final-project-files", "Talent", "txt", "sparta_day_data.json")
txt_pip.extract()
print(txt_pip.dataframe)



transform = Transformer(json_pip.dataframe, csv_talent_pip.dataframe, csv_academy_pip.dataframe, txt_pip.dataframe, "foo.json")

transform.list_attributes()
transform.create_attribute_tables()

transform.create_candidates_table()
transform.create_interview_table()

transform.create_tech_skill_tables()

transform.create_quality_junction()
transform.create_quality_table()


print(transform.interview_table)
print(transform.candidates_table)
print(transform.tech_skills_table)
print(transform.tech_junction_table)
print(transform.quality_table)
print(transform.quality_junction_table)




















