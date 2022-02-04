import datetime
import re
import boto3
import pandas as pd

s3_resource = boto3.resource("s3")
bucket_name = "data-26-final-project-files"
bucket = s3_resource.Bucket(bucket_name)


def read_all_txt_to_lst():
    prefix_objs = bucket.objects.filter(Prefix="Talent/Sparta")
    text_lst = []
    for obj in prefix_objs:
        try:
            body = obj.get()['Body'].read().decode('utf-8')
            lines = body.splitlines()
            text_lst.append(lines)
        except:
            continue
    return text_lst


def lst_to_df(text_lst):
    dict_lst = []
    for row in text_lst:
        date = row[0]
        new_date = datetime.datetime.strptime(date, '%A %d %B %Y').strftime('%d/%m/%Y')
        academy = re.split(" ", row[1])[0].strip()
        for ele in row[3:]:
            name = re.split("[-:]", ele)[0].strip().title()
            psy_score = re.split("[-:,/]", ele)[2].strip()
            pres_score = re.split("[-:,/]", ele)[5].strip()
            d = {"Name": name, "Psychometrics": psy_score, "Presentation": pres_score,
                 "Academy": academy, "Date": new_date}
            dict_lst.append(d)
    df = pd.DataFrame(dict_lst)
    return df


if __name__ == "__main__":
    lst = read_all_txt_to_lst()
    df1 = lst_to_df(lst)
    print(df1)
    df1.to_csv("text_files_combined.csv")