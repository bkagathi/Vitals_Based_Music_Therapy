import pandas as pd
from pandas import DataFrame as df
import csv

df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # clinical information
df_trks = pd.read_csv("https://api.vitaldb.net/trks")  # track list
df_labs = pd.read_csv('https://api.vitaldb.net/labs')  # laboratory results

df_cases.to_csv(r'C:\Users\mirza\Documents\Senior_Design_Project\my_data.csv', index=False)