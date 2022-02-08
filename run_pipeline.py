from pipeline import *


# Pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

print("\n\njson Dataframe:")
json_pip = Pipeline("data-26-final-project-files", "Talent", "json", "candidates_sparta_data.json")
json_pip.extract()
print(json_pip.dataframe.head())

print("\n\ncsv Talent Dataframe:")
csv_talent_pip = Pipeline("data-26-final-project-files", "Talent", "csv", "candidate_data.json")
csv_talent_pip.extract()
print(csv_talent_pip.dataframe.head())

print("\n\ncsv Academy Dataframe:")
csv_academy_pip = Pipeline("data-26-final-project-files", "Academy", "csv", "academy_data.json")
csv_academy_pip.extract()
print(csv_academy_pip.dataframe.head())

print("\n\ntxt Dataframe:")
txt_pip = Pipeline("data-26-final-project-files", "Talent", "txt", "sparta_day_data.json")
txt_pip.extract()
print(txt_pip.dataframe)


print("\n\nBIG TABLE sick\n")


big_table = Transformer(json_pip.dataframe, csv_talent_pip.dataframe, csv_academy_pip.dataframe, txt_pip.dataframe, "foo.json")
print(big_table.big_table.columns)

print("\n candindate table")
big_table.create_candidates_table()
print(big_table.big_table.columns)

print("\nSparta Day")
sparta_day = big_table.create_sparta_day_table()
# print(big_table.create_sparta_day_table())
#
print("\n sparta day results table")
sparta_day = big_table.create_sparta_day_results_table()

print("\n benchmark table table")

big_table.create_benchmarks_table()



print("\n\n\n\n")
# print(big_table.big_table[["start_date", "invited_date"]].loc[:200])
# print(big_table.big_table[["start_date"].notnull().sum())
#print(big_table.big_table[["start_date"]].unique())