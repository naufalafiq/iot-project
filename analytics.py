import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "<MongoDB Database>"
DB_NAME = "asta"
COLLECTION_NAME = "telemetrix"

local = "UTC"
convert = "Asia/Singapore"

# Connect to MongoDB
client = pymongo.MongoClient(url)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def fetch_data():
    cursor = collection.find({}, {"_id": 0, "time": 1, "params": 1})
    data = list(cursor)
    df = pd.DataFrame(data)
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])  # Convert time to datetime
        df['time'] = df['time'].dt.tz_localize(local)
        df['time'] = df['time'].dt.tz_convert(convert)
        # Flatten 'params' into separate columns
        params_df = pd.json_normalize(df['params'])
        df = pd.concat([df, params_df], axis=1)
        df.set_index('time', inplace=True)  # Set time as index
    return df

binary_params = ['door', 'motion']  # Example binary params
decimal_params = ['temperature', 'humidity']  # Example decimal params

# Plot Time Series
def plot_time_series(df, interval, param, title):
    start_date = "2024-12-06"
    end_date = "2024-12-06"
    df[param] = pd.to_numeric(df[param], errors='coerce')
    if param in binary_params:
        # Resample and calculate counts of 0 and 1
        if interval == 'h':
            df = df.sort_index()
            df = df.loc[start_date]
        resampled_data = df.resample(interval)
        resampled_data = resampled_data[param].value_counts().unstack(fill_value=0)
        print(resampled_data.head())
        # Extract counts for 0 and 1
        count_0 = resampled_data[0] if 0 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        count_1 = resampled_data[1] if 1 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        
        # Plot using Matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(count_0.index, count_0, label='Count of 0', marker='o', color='blue')
        plt.plot(count_1.index, count_1, label='Count of 1', marker='o', color='orange')
        if interval == 'h':
            plt.title(f"{title} from {start_date} to {end_date}")
        else:
            plt.title(f"{title}")
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    elif param in decimal_params:
        if interval == 'h':
            df = df.sort_index()
            df = df.loc[start_date]
        # Resample to calculate min, max, and mean
        resampled_data = df[param].resample(interval).agg(['min', 'max', 'mean'])
        print(resampled_data.head())
        # Plot using Matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(resampled_data.index, resampled_data['min'], label='Min', marker='o', color='red')
        plt.plot(resampled_data.index, resampled_data['max'], label='Max', marker='o', color='green')
        plt.plot(resampled_data.index, resampled_data['mean'], label='Mean', marker='o', color='blue')
        if interval == 'h':
            plt.title(f"{title} from {start_date} to {end_date}")
        else:
            plt.title(f"{title}")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

# Main Program
df = fetch_data()

if not df.empty:
    for param in binary_params:
        # Hourly Aggregation
        plot_time_series(df, "h", param, f"Hourly {param} State Count")

        # Daily Aggregation
        plot_time_series(df, "D", param,f"Daily {param} State Count")

        # Weekly Aggregation
        plot_time_series(df, "W", param,f"Weekly {param} State Count")

        # Monthly Aggregation
        plot_time_series(df, "ME", param,f"Monthly {param} State Count")
    for param in decimal_params:
        # Hourly Aggregation
        plot_time_series(df, "h", param,f"Hourly {param}")

        # Daily Aggregation
        plot_time_series(df, "D", param,f"Daily {param}")

        # Weekly Aggregation
        plot_time_series(df, "W", param,f"Weekly {param}")

        # Monthly Aggregation
        plot_time_series(df, "ME", param,f"Monthly {param}")
else:
    print("No data found in MongoDB!")  
