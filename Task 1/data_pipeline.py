import pandas as pd
import numpy as np

#Load the original datase
data = pd.read_csv("ev_campus_dataset (1).csv")

#Check the dataset size and display the first
print(data.shape)
print(data.head(5))
print()

#Check the number of missing values in each column
print("Missing values in each column:")
print(data.isnull().sum())
print()

#Check for duplicate session IDs
print("Number of duplicate Session_ID values:")
print(data["Session_ID"].duplicated().sum())
print()

#Fill missing values in numeric columns with their median values
numeric_columns = data.select_dtypes(include=np.number).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

#Remove duplicate records based on Session_ID
before = len(data)
data = data.drop_duplicates(subset="Session_ID", keep="first")
print("Number of duplicate records removed", before - len(data))
print()

#Check for missing values again after cleaning
print("Missing values after cleaning:")
print(data.isnull().sum())
print()

#Calculate charging power in kW
data["Power_kW"] = data["Energy_kWh"] / (data["Duration_Mins"] / 60)

print("Energy, duration, and calculated power:")
print(data[["Energy_kWh", "Duration_Mins", "Power_kW"]].head(5))
print()

#Create a peak-hour flag: 12:00 to before 17:00 is peak time
data["Peak_Hour_Flag"] = (
    (data["Start_Hour"] >= 12) &
    (data["Start_Hour"] < 17)
).astype(int)

print("Start hour and peak-hour flag:")
print(data[["Start_Hour", "Peak_Hour_Flag"]].head(5))
print()

#Divide sessions into low, medium, and high energy-use groups
data["Energy_Efficiency_Group"] = pd.qcut(
    data["Energy_kWh"],
    q=3,
    labels=["Low", "Medium", "High"]
)

print("Energy group examples:")
print(data[["Energy_kWh", "Energy_Efficiency_Group"]].head(10))
print()

#Summarize the data by charger type and peak-hour flag
summary = data.groupby(["Charger_Type", "Peak_Hour_Flag"]).agg(
    Session_Count=("Session_ID", "count"),
    Average_Energy_kWh=("Energy_kWh", "mean"),
    Average_Battery_Temp_C=("Battery_Temp_C", "mean")
).reset_index()

print("Summary by charger type and peak-hour flag:")
print(summary)

# Save the cleaned dataset and summary table
data.to_csv("cleaned_ev_dataset.csv", index=False, encoding="utf-8-sig")
summary.to_csv("charger_peak_summary.csv", index=False, encoding="utf-8-sig")

print("The cleaned dataset and summary table have been saved")