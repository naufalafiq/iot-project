import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "<mongodb>"
#url = "<mongodb>"
DB_NAME = "asta"
#DB_NAME = "hivemq"
#COLLECTION_NAME = "sensor"
COLLECTION_NAME = "telemetrix"

local = "UTC"
convert = "Asia/Singapore"

# Connect to MongoDB
client = pymongo.MongoClient(url)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

device_brands = collection.distinct("deviceBrand")
device_models = collection.distinct("deviceModel")
device_types = collection.distinct("deviceType")
device_names = ['Dining Room', 'Guest', 'Fridge', 'Living', 'Patio', 'Pooja', 'Utility', 'Kitchen', 'Master', 'UPS']

def fetch_data():
    cursor = collection.find()
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

def init_params():
    for record in collection.find({}, {"params": 1}):  # Fetch only the params field
        params = record.get("params", {})
        for key, value in params.items():
            try:
                # Attempt to convert the value to float
                numeric_value = float(value)
                if numeric_value in [0, 1]:  # Check if it's binary
                    if key not in binary_params:  # Ensure no duplicates
                        binary_params.append(key)
                else:  # Otherwise, treat it as decimal
                    if key not in decimal_params:  # Ensure no duplicates
                        decimal_params.append(key)
            except ValueError:
                if key not in string_params:  # Ensure no duplicates
                    string_params.append(key)

binary_params = ['door', 'motion']  # Example binary params
decimal_params = ['temperature', 'humidity', 'voltage']  # Example decimal params
string_params = ['outlet']

# Plot Time Series
def plot_time_series(df, interval, param, title):
    if param in binary_params:
        df = df[df[param] != "unavailable"]
        df[param] = pd.to_numeric(df[param], errors='coerce')
        df = df[df[param].notna()]
        # Resample and calculate counts of 0 and 1
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
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
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
        df = df[df[param] != "unavailable"]
        df[param] = pd.to_numeric(df[param], errors='coerce')
        df = df[df[param].notna()]
        # Resample to calculate min, max, and mean
        resampled_data = df[param].resample(interval).agg(['min', 'max', 'mean'])
        print(resampled_data.head())
        # Plot using Matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(resampled_data.index, resampled_data['min'], label='Min', marker='o', color='red')
        plt.plot(resampled_data.index, resampled_data['max'], label='Max', marker='o', color='green')
        plt.plot(resampled_data.index, resampled_data['mean'], label='Mean', marker='o', color='blue')
        # Get the start and end date from the DataFrame and format to remove time
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        if interval == 'h':
            plt.title(f"{title} from {start_date} to {end_date}")
        else:
            plt.title(f"{title}")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif param in string_params:
        df = df[df[param] != "unavailable"]
        df["outlet"] = df["outlet"].map({"on": 1, "off": 0})

        resampled_data = df.resample(interval)
        resampled_data = resampled_data[param].value_counts().unstack(fill_value=0)

        # Extract counts for 0 and 1
        count_0 = resampled_data[0] if 0 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        count_1 = resampled_data[1] if 1 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        
        # Plot using Matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(count_0.index, count_0, label='Count of 0', marker='o', color='blue')
        plt.plot(count_1.index, count_1, label='Count of 1', marker='o', color='orange')
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        if interval == 'h':
            plt.title(f"{title} from {start_date} to {end_date}")
        else:
            plt.title(f"{title}")
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.show()

    
def filter_data(df, brand, model, dtype, name):
    # Dictionary to store filters
    filters = {}
    
    # Gather user input for each filter
    filters["deviceBrand"] = brand
    filters["deviceModel"] = model
    filters["deviceType"] = dtype
    facility_keywords = name
    
    # Clean up empty filters
    filters = {key: [v.strip() for v in values if v.strip()] for key, values in filters.items()}
    
    # Apply filters dynamically
    for key, values in filters.items():
        if values:  # If the user provided values for this filter
            if key in df.columns:
                df = df[df[key].isin(values)]
            else:
                print(f"Warning: '{key}' is not a valid column in the DataFrame. Skipping this filter.")

    # Apply facility keyword filter
    if facility_keywords:
        keyword_pattern = "|".join(facility_keywords)  # Create a regex pattern from the keywords
        df = df[df["deviceName"].str.contains(keyword_pattern, case=False, na=False)]  # Case-insensitive match
        print(f"Successfully filtered {facility_keywords}")
    
    # Check for empty DataFrame
    if df.empty:
        print("The filtered DataFrame is empty. No matching records found.")
    else:
        print(f"Filtered DataFrame:\n{df.head()}")
    
    return df

def prob():
    while True:
        try:
            # Fetch your data (replace this with your actual data fetching method)
            df = fetch_data()
            param = input("Enter parameter to analyse: ")

            # Remove unavailable values and convert the param column to numeric
            df = df[df[param] != "unavailable"]
            df[param] = pd.to_numeric(df[param], errors='coerce')
            df = df[df[param].notna()]

            # Extract hour and date components
            df['hour'] = df.index.hour
            df['day'] = df.index.date
            df['day_name'] = df.index.day_name()

            # Ask for user input
            print("'1' for hourly")
            print("'2' for day")
            user_input = input("Enter choice: ")

            if user_input == '1':
                # Get user input for specific date and hour
                date_input = input("Enter the date (YYYY-MM-DD): ")
                time_input = input("Enter the hour (0-23): ")
                
                # Convert input to datetime
                user_date = pd.to_datetime(date_input)
                user_hour = int(time_input)
                
                # Filter data for the specific date and hour
                hour_data = df[(df['day'] == user_date.date()) & (df['hour'] == user_hour)]
                
                # Calculate actual frequency (param = 1 and param = 0)
                actual_count_1 = hour_data[param].sum()
                actual_count_0 = hour_data[param].count() - actual_count_1
                
                # Calculate probability using a rolling window of previous hours (e.g., last 24 hours)
                rolling_window = 24  # This is the number of hours to look back to calculate the probability
                df['rolling_prob'] = df[param].rolling(rolling_window, min_periods=1).mean()

                # Get the probability for the specific hour
                prob_1 = df.loc[df['hour'] == user_hour, 'rolling_prob'].mean()  # Calculate probability for the given hour
                print(f"Probability for Param = 1 at hour {user_hour} on {user_date}: {prob_1:.2f}")

                # Calculate expected frequency for param = 1
                total_count = hour_data[param].count()
                expected_count_1 = prob_1 * total_count
                expected_count_0 = (1 - prob_1) * total_count
                
                # Print Table Report
                report = pd.DataFrame({
                    'Metric': [f'Actual {param} Freq = 1', f'Expected {param} Freq = 1',
                               f'Actual {param} Freq = 0', f'Expected {param} Freq = 0'],
                    'Value': [actual_count_1, expected_count_1, actual_count_0, expected_count_0]
                })

                if user_hour < 10:
                    user_hour = f"0{user_hour}:00:00"
                else:
                    user_hour = f"{user_hour}:00:00"

                print("\n--- Hourly Report ---")
                print(f"Date: {user_date.date()} {user_hour}")
                print(report.to_string(index=False))

                # Plot the actual vs expected frequencies for param = 1 and param = 0 (scatter plot)
                plt.figure(figsize=(10, 6))
                plt.scatter([user_hour], [actual_count_1], color='red', label='Actual Param = 1', s=100, marker='^')
                plt.scatter([user_hour], [actual_count_0], color='orange', label='Actual Param = 0', s=100)
                plt.scatter([user_hour], [expected_count_1], color='blue', label='Expected Param = 1', s=100, marker='x')
                plt.scatter([user_hour], [expected_count_0], color='green', label='Expected Param = 0', s=100, marker='x')
                plt.title(f'Actual vs Expected Frequency for {param} (Hourly) at {user_date} {user_hour}')
                plt.xlabel('Hour of Day')
                plt.ylabel('Frequency')
                plt.legend(loc='upper right')  # Position of the legend
                plt.grid(True)
                plt.show()

            elif user_input == '2':
                # Get user input for specific date
                date_input = input("Enter the date (YYYY-MM-DD): ")
                
                # Convert input to datetime
                user_date = pd.to_datetime(date_input).date()
                
                # Filter data for the specific date
                day_data = df[df['day'] == user_date]
                
                # Calculate actual frequency (param = 1 and param = 0)
                actual_count_1 = day_data[param].sum()
                actual_count_0 = day_data[param].count() - actual_count_1
                
                # Calculate probability using a rolling window of previous days (e.g., last 7 days)
                rolling_window = 7  # This is the number of days to look back to calculate the probability
                df['rolling_prob'] = df[param].rolling(rolling_window, min_periods=1).mean()

                # Get the probability for the specific day
                prob_1 = df.loc[df['day'] == user_date, 'rolling_prob'].mean()  # Calculate probability for the given day
                print(f"Probability for Param = 1 on {user_date}: {prob_1:.2f}")
                
                # Calculate expected frequency for param = 1
                total_count = day_data[param].count()
                expected_count_1 = prob_1 * total_count
                expected_count_0 = (1 - prob_1) * total_count

                # Print Table Report
                report = pd.DataFrame({
                    'Metric': [f'Actual {param} Freq = 1', f'Expected {param} Freq = 1',
                               f'Actual {param} Freq = 0', f'Expected {param} Freq = 0'],
                    'Value': [actual_count_1, expected_count_1, actual_count_0, expected_count_0]
                })
                print("\n--- Daily Report ---")
                print(f"Date: {user_date}")
                print(report.to_string(index=False))

                # Plot the actual vs expected frequencies for param = 1 and param = 0 (scatter plot)
                plt.figure(figsize=(10, 6))
                plt.scatter([user_date], [actual_count_1], color='red', label='Actual Param = 1', s=100, marker='^')
                plt.scatter([user_date], [actual_count_0], color='orange', label='Actual Param = 0', s=100)
                plt.scatter([user_date], [expected_count_1], color='blue', label='Expected Param = 1', s=100, marker='x')
                plt.scatter([user_date], [expected_count_0], color='green', label='Expected Param = 0', s=100, marker='x')
                plt.title(f'Actual vs Expected Frequency for {param} (Daily) on {user_date}')
                plt.xlabel('Date')
                plt.ylabel('Frequency')
                plt.legend(loc='upper right')  # Position of the legend
                plt.grid(True)
                plt.show()

            else:
                print("Invalid input.")
        except KeyError:
            print("Parameter not found.")
            continue
        except KeyboardInterrupt:
            print("Exiting...")
            exit()
        except:
            print("Error parsing date.")
            continue


def prob():
    while True:
        try:
            print("Expected Probability")
            param = input("Enter parameter to analyse: ")
            print("'1' for hourly")
            print("'2' for daily")
            choice = int(input("Enter your choice: "))

            df = fetch_data()
            if param not in df.columns:
                print(f"Parameter '{param}' not found in data.")
                continue
            df = df[df[param] != "unavailable"]
            df[param] = pd.to_numeric(df[param], errors='coerce')
            df = df[df[param].notna()]

            if choice == 1:
                try:
                    start_date = input("Enter your date (hourly): ")
                    data = df.loc[start_date:start_date]
                except:
                    print("Date not found.")
                    continue
                
                try:
                    start_time_1 = input("Enter start time to use for probability: ")
                    end_time_1 = input("Enter end time to use for probability: ")
                    start_time_2 = input("Enter start time to expect frequency: ")
                    end_time_2 = input("Enter end time to expect frequency: ")
                    # Filter for the time window of 10 AM to 11 AM for 16/12/2024 (based on hour)
                    df_filtered = data.between_time(start_time_1, end_time_1)
                    df_filtered_next = data.between_time(start_time_2, end_time_2)
                    if df_filtered.empty:
                        print("The data for the first period doesn't exist.")
                        continue
                    elif df_filtered_next.empty:
                        print("The data for the second period doesn't exist.")
                        continue
                except:
                    print("Time not found.")
                    continue

                # Calculate the probability of light=1 in this window
                prob_light_1 = df_filtered[param].mean()
                exp_freq = prob_light_1 * len(df_filtered_next)
                num_light_1 = df_filtered_next[df_filtered_next[param] == 1].shape[0]
                total = len(df_filtered_next)

                print(f"Probability of {param} = 1 from {start_time_1} PM to {end_time_1} PM on {start_date}: {prob_light_1:.2f}")
                print(f"Expected frequency of {param} = 1 from {start_time_2} PM to {end_time_2} PM on {start_date}: {exp_freq:.2f}")
                print(f"Actual frequency of {param} = 1 from {start_time_2} PM to {end_time_2} PM on {start_date}: {num_light_1:.2f}")
                print(f"Total data for {param} from {start_time_2} PM to {end_time_2} PM on {start_date}: {total} ")

                resampled_df = df_filtered.resample('h')
                resampled_df = resampled_df[param].value_counts().unstack(fill_value=0)
                resampled_df_next = df_filtered_next.resample('h')
                resampled_df_next = resampled_df_next[param].value_counts().unstack(fill_value=0)
                print(resampled_df)
                print(resampled_df_next)
            elif choice == 2:
                # Convert input date to pandas datetime
                date_input_og = input("Enter date to analyse (daily): ")
                date_input = pd.to_datetime(date_input_og).tz_localize("Asia/Singapore")
                input_day = date_input.day_name()  # Determine the day of the week

                # Filter for all data before the specified date
                df_filtered = df.loc[df.index < date_input]

                # Filter rows where the day of the week matches the input day
                df_same_day = df_filtered[df_filtered.index.day_name() == input_day]

                if df_same_day.empty:
                    print(f"No previous occurrences for {input_day} before {date_input.date()}.")
                    return

                # Calculate average frequency of light = 1
                average_frequency = df_same_day[param].mean()

                # Count how many previous occurrences of the day exist
                num_occurrences = df_same_day.shape[0]

                df_filtered_day = df.loc[date_input_og]
                exp_freq = average_frequency * num_occurrences
                num_param = df_filtered_day[df_filtered_day[param] == 1].shape[0]
                

                print(f"Day: {input_day}")
                print(f"Expected frequency of {param} = 1 for {input_day}: {exp_freq:.2f}")
                print(f"Actual frequency of {param} = 1 for {input_day}: {num_param:.2f}")

        except KeyboardInterrupt:
            print("Exiting...")
            exit()

def main():
    intervals = {
        "1": ("h", "Hourly"),
        "2": ("D", "Daily"),
        "3": ("W", "Weekly"),
        "4": ("ME", "Monthly"),
        "5": ("YS", "Yearly")
    }
    
    while True:
        try:
            df = fetch_data()
            print(f"List of binary parameters: {binary_params}")
            print(f"List of decimal parameters: {decimal_params}")
            print(f"List of string parameters: {string_params}")
            
            param = input("Enter the parameter to analyze (choose above): ")
            print("'1' for Hourly")
            print("'2' for Daily")
            print("'3' for Weekly")
            print("'4' for Montly")
            print("'5' for Yearly")
            interval = input("Enter the interval to analyze (choose above): ")

            # Gather user input for each filter
            print(f"Device Brands: {device_brands}")
            brand = input("Enter deviceBrand(s) (comma-separated, or leave blank to skip): ").strip().split(",")
            print(f"Device Models: {device_models}")
            model = input("Enter deviceModel(s) (comma-separated, or leave blank to skip): ").strip().split(",")
            print(f"Device Types: {device_types}")
            dtype = input("Enter deviceType(s) (comma-separated, or leave blank to skip): ").strip().split(",")
            print(f"Device Names: {device_names}")
            name = input("Enter deviceName (comma-separated, or leave blank to skip): ").strip().split(",")
            
            try:
                if interval == '1':
                    start_date = input("Enter start date to analyze in the format (YYYY-MM-DD): ")
                    end_date = input("Enter end to analyze in the format (YYYY-MM-DD): ")
                    df = df.sort_index()
                    data = df.loc[start_date:end_date]
                    filtered_df = filter_data(data, brand, model, dtype, name)
                else:
                    df = df.sort_index()
                    filtered_df = filter_data(df, brand, model, dtype, name)
                interval_result, interval_name = intervals[interval]
            except:
                print("Time interval/filter value not found.")
                continue

            if param in binary_params:
                title = f"{param} | {interval_name} State Count"
                plot_time_series(filtered_df, interval_result, param, title)
            elif param in decimal_params:
                title = f"{param} | {interval_name}"
                plot_time_series(filtered_df, interval_result, param, title)
            elif param in string_params:
                title = f"{param} | {interval_name} State Count"
                plot_time_series(filtered_df, interval_result, param, title)
            else:
                print("Parameter not found.")
                continue
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except ValueError:
            print("No data found after filtering.")
            continue
        except:
            continue


# Main Program
df = fetch_data()
main()
prob()
