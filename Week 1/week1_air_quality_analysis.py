import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("AirQualityUCI.csv", sep=";")
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
df = df.drop(columns=["Date", "Time"]).set_index("DateTime").sort_index()

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -200 is the dataset's missing-value marker
df = df.replace(-200, np.nan)
df = df.drop_duplicates()
df = df.interpolate(method="time", limit_direction="both")

for c in df.columns:
    df[c] = df[c].fillna(df[c].median())

# EDA
df["Hour"] = df.index.hour
print(df.describe())
print(df.corr(numeric_only=True))

plt.figure(figsize=(9,5))
plt.hist(df["CO(GT)"], bins=40)
plt.title("Distribution of CO Concentration")
plt.xlabel("CO (mg/m³)")
plt.ylabel("Frequency")
plt.show()

hourly = df.groupby("Hour")[["CO(GT)", "NOx(GT)", "NO2(GT)"]].mean()
plt.figure(figsize=(9,5))
plt.plot(hourly.index, hourly["CO(GT)"], label="CO")
plt.plot(hourly.index, hourly["NOx(GT)"]/100, label="NOx / 100")
plt.plot(hourly.index, hourly["NO2(GT)"], label="NO2")
plt.legend()
plt.title("Average Pollution Levels by Hour")
plt.show()

plt.figure(figsize=(9,5))
plt.scatter(df["CO(GT)"], df["NOx(GT)"], s=5, alpha=.25)
plt.xlabel("CO (mg/m³)")
plt.ylabel("NOx (ppb)")
plt.title("CO vs NOx Concentration")
plt.show()
