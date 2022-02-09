
# Data 26 Final Project Docker Set Up



## CREATE CONTAINER

### NOTE: Docker will not run from command line unless the docker.exe file location is in path

#### To set up docker image and container run from the command line:

docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=Pa55word" -p 1433:1433 --name data26 -d mcr.microsoft.com/mssql/server:2019-latest

#### To check if the container is setup correctly, enter into command line:

docker images

You should get:

![image](https://user-images.githubusercontent.com/97161073/153217630-12ed001b-4d17-449c-8e2b-1c263b9296b4.png)


Open Azure Data Studio and connect to localhost, 1433 
Login details:
•	Username: SA
•	Password: Pa55word

<img src="https://user-images.githubusercontent.com/97161073/153218018-21ba044a-4ec7-4cad-a3f9-44ea39856cd8.png" width="550" height="400">

## CREATE DATABASE

#### A database called “spartaGlobal” should be created in Azure, by running command

**CREATE DATABASE** spartaGlobal

After connecting to data26 container, open a python script. From here everything can be handled using “pyodbc”, using

import pyodbc

server = 'localhost, 1433'
database = 'spartaGlobal'
username = 'SA'
password = 'Pa55word'
spartaGlobal = pyodbc.connect(('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password))

to connect & use SQL queries in Python

#### CREATE TABLES

setUpTable.py can be used to populate this database with tables, using configuration found in tables.json.
Be aware that running setUpTable.py will drop any tables that currently exist before repopulating.

