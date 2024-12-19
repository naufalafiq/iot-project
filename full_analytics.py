'''from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from pandas.plotting import autocorrelation_plot'''
from pymannkendall import original_test
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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
device_names = collection.distinct("deviceName")

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
                # Ignore non-numeric values
                continue

binary_params = ['door', 'motion']  # Example binary params
decimal_params = ['temperature', 'humidity', 'voltage']  # Example decimal params  

# Plot Time Series
def plot_time_series(df, interval, param, title):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    if param in binary_params:
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
        '''plt.figure(figsize=(12, 6))
        plt.plot(binary_df.index, binary_df['0'], label='Count of Closed (0)', marker='o', color='blue')
        plt.plot(binary_df.index, binary_df['1'], label='Count of Open (1)', marker='o', color='orange')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()'''
    elif param in decimal_params:
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
    '''# Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(counts.index, counts['Closed'], label='Count of Closed (0)', marker='o', color='blue')
    plt.plot(counts.index, counts['Open'], label='Count of Open (1)', marker='o', color='orange')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

def adf_test(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    data = df[param]

    # Perform the ADF test
    result = adfuller(data)

    # Extract results
    '''print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])'''

    if result[1] < 0.05:
        print(f"{param} is stationary.")
    else:
        print(f"{param} is not stationary.")

def decompose(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    data = df[param]
    
    # Perform decomposition
    result = seasonal_decompose(data, model='additive', period=4)
    result.plot()
    plt.show()



def sma(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    #data = df[param].resample(interval).mean()
    df = df[df[param].notna()]
    data = df[param]
    print(data)
    sma10 = data.rolling(10).mean()
    sma20 = data.rolling(20).mean()
    # Plot the original data and SMA
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original Data', color='blue')
    plt.plot(sma10, label=f'10-Point SMA', color='green')
    plt.plot(sma20, label=f'20-Point SMA', color='orange')
    plt.xlabel('Time')
    plt.ylabel(f'{param}')
    plt.grid()
    plt.title('Simple Moving Average')
    plt.legend()
    plt.show()

def cma(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    #df = df[param].resample(interval).mean()
    df = df[df[param].notna()]
    data = df[param]
    print(data)
    ema10 = data.ewm(10, adjust=False).mean()
    ema20 = data.ewm(20, adjust=False).mean()

    # Plot the original data and EMA
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original Data', color='blue')
    plt.plot(ema10, label=f'10-Point EMA', color='orange')
    plt.plot(ema20, label=f'20-Point EMA', color='yellow')
    plt.title('Exponential Moving Average')
    plt.xlabel('Time')
    plt.ylabel(f'{param}')
    plt.grid()
    plt.legend()
    plt.show()

def arima(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    data = df[param]
    data = data.asfreq('H')
    model = ARIMA(data, order=(1,0,1))
    fitted_model = model.fit()
    print(fitted_model.summary())

    
    # Step 8: Forecasting
    forecast_steps = 24
    forecast = fitted_model.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='H')[1:]

    # Plot Original Data and Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Original Time Series")
    plt.plot(forecast_index, forecast, label="Forecast", color='red')
    plt.title("Time Series Forecasting")
    plt.xlabel("Date")
    plt.ylabel(f"{param}")
    plt.legend()
    plt.show()

    # Step 9: Calculate Forecast Error
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    model = ARIMA(train, order=(1, 0, 1))
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test))
    error = mean_squared_error(test, predictions)
    print(f"Mean Squared Error: {error:.2f}")

def adf_test(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    data = df[param]
    result = adfuller(data)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

def sarimax(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    data = df[param]

    # Define the SARIMAX model
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))

    # Fit the model
    fitted_model = model.fit()
    # Forecast for the next 50 steps
    forecast_steps = 50
    forecast = fitted_model.forecast(steps=forecast_steps)
    # Fitted values (in-sample predictions)
    fitted_values = fitted_model.fittedvalues[1:]
    data = data[1:]

    # Create a datetime index for the forecast
    forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='h')[1:]
    forecast = pd.Series(forecast, index=forecast_index)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data')
    #plt.plot(fitted_values, label='Fitted', color='green')
    plt.plot(forecast, label='Forecast', color='orange')
    plt.xlabel(f"{param}")
    plt.ylabel('Time')
    plt.grid()
    plt.legend()
    plt.show()

def acf_pacf(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    if param in binary_params:
        resampled_data = df.resample('D')
        resampled_data = resampled_data[param].value_counts().unstack(fill_value=0)
        count_0 = resampled_data[0] if 0 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        count_1 = resampled_data[1] if 1 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        adf_test_0 = adfuller(count_0)
        adf_test_1 = adfuller(count_1)
        print(f"--- {param} ADF Test ---")
        print(f'ADF Statistic (0): {adf_test_0[0]}')
        print(f'p-value (0): {adf_test_0[1]}')
        print(f'ADF Statistic (1): {adf_test_1[0]}')
        print(f'p-value (1): {adf_test_1[1]}')

        plot_acf(count_0, lags=6)
        plt.title(f"{param} 0 Autocorrelation")
        plot_acf(count_1, lags=6)
        plt.title(f"{param} 1 Autocorrelation")
        plt.show()

        df_diff_0 = np.diff(count_0, n=1)
        df_diff_1 = np.diff(count_0, n=1)
        adf_test_0 = adfuller(df_diff_0)
        adf_test_1 = adfuller(df_diff_1)
        # stationary
        print(f"--- {param} ADF Test ---")
        print(f'ADF Statistic (0): {adf_test_0[0]}')
        print(f'p-value (0): {adf_test_0[1]}')
        print(f'ADF Statistic (1): {adf_test_1[0]}')
        print(f'p-value (1): {adf_test_1[1]}')

        plt.plot(df_diff_0)
        plt.title(f'{param} Differenced Closing Prices (0)')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.tight_layout()

        plt.plot(df_diff_1)
        plt.title(f'{param} Differenced Closing Prices (1)')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.tight_layout()

        plot_acf(df_diff_0, lags=5)
        plt.title(f"{param} Autocorrelation 0 (after Diff)")
        plt.tight_layout()

        plot_acf(df_diff_1, lags=5)
        plt.title(f"{param} Autocorrelation 1 (after Diff)")
        plt.tight_layout()
    elif param in decimal_params:
        data = df[param].resample('D').mean()
        adf_test = adfuller(data)
        print(f"--- {param} ADF Test ---")
        print(f'ADF Statistic: {adf_test[0]}')
        print(f'p-value: {adf_test[1]}')

        plot_acf(data, lags=6)
        plt.title(f"{param} Autocorrelation")
        plt.show()

        df_diff = np.diff(data, n=1)
        adf_test = adfuller(df_diff)
        # stationary
        print(f'ADF Statistic (after Diff): {adf_test[0]}')
        print(f'p-value (after Diff): {adf_test[1]}')

        plt.plot(df_diff)
        plt.title(f'{param} Differenced Closing Prices')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.tight_layout()

        plot_acf(df_diff, lags=5)
        plt.title(f"{param} Autocorrelation (after Diff)")
        plt.tight_layout()
    
    

def holt(df, param):
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df[df[param].notna()]
    #data = data.asfreq('D')
    if param in binary_params:
        #start_day = "2024-12-02"
        #df = df.sort_index()
        #df = df.loc[start_day]
        # Resample and calculate counts of 0 and 1
        resampled_data = df.resample('D')
        resampled_data = resampled_data[param].value_counts().unstack(fill_value=0)
 
        # Extract counts for 0 and 1
        count_0 = resampled_data[0] if 0 in resampled_data.columns else pd.Series(0, index=resampled_data.index)
        count_1 = resampled_data[1] if 1 in resampled_data.columns else pd.Series(0, index=resampled_data.index)

        # Fit a Holt-Winters model for counts of 0
        model_0 = ExponentialSmoothing(count_0, trend='add', seasonal=None, seasonal_periods=None).fit()

        # Fit a Holt-Winters model for counts of 1
        model_1 = ExponentialSmoothing(count_1, trend='add', seasonal=None, seasonal_periods=None).fit()

        fitted_values_0 = model_0.fittedvalues
        fitted_values_1 = model_1.fittedvalues
        forecast_steps = 3
        forecast_0 = model_0.forecast(steps=forecast_steps)
        forecast_1 = model_1.forecast(steps=forecast_steps)
        forecast_0 = forecast_0.clip(lower=0)
        forecast_1 = forecast_1.clip(lower=0)
        forecast_index = pd.date_range(resampled_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        forecast_df = pd.DataFrame({'Forecast_0': forecast_0, 'Forecast_1': forecast_1}, index=forecast_index)

        plt.figure(figsize=(12, 6))
        plt.plot(resampled_data.index, resampled_data[0], label='Actual 0', marker='o', color='blue')
        plt.plot(resampled_data.index, fitted_values_0, label='Fitted Values 0', marker='o', color='yellow')
        plt.plot(forecast_df.index, forecast_df['Forecast_0'], label='Forecast 0', marker='o', color='orange')

        
        plt.plot(resampled_data.index, resampled_data[1], label='Actual 1', marker='o', color='black')
        plt.plot(resampled_data.index, fitted_values_1, label='Fitted Values 1', marker='o', color='green')
        plt.plot(forecast_df.index, forecast_df['Forecast_1'], label='Forecast 1', marker='o', color='red')
        plt.title("Holt-Winters Forecasts")
        plt.xlabel("Time")
        plt.ylabel(f"Counts of {param}")
        plt.grid()
        plt.legend()
        plt.show()
    elif param in decimal_params:
        steps = 3
        # Resample to daily frequency and interpolate missing values
        data = df[param].resample('D').mean()
        print(data)

        model = ExponentialSmoothing(data, trend='add', seasonal=None, seasonal_periods=None)
        fitted_model = model.fit()
        # Forecast
        forecast = fitted_model.forecast(steps=steps)
        fitted_values = fitted_model.fittedvalues
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Original Data', marker='o')
        plt.plot(fitted_values, label='Fitted Values', marker='o', color='purple')
        plt.plot(forecast_index, forecast, label='Forecast', color='orange', marker='o')
        plt.title(f'Holt-Winters Forecast for Continuous Data ({param})')
        plt.xlabel('Time')
        plt.ylabel(f'{param}')
        plt.legend()
        plt.grid()
        plt.show()

def ensure_stationarity(df):
    result = adfuller(df)
    if result[1] > 0.05:
        df = df.diff().dropna()
        return df
    else:
        return df
    
def split_data(df, split_ratio=0.8):
    train_size = int(len(df) * split_ratio)
    return df[:train_size], df[train_size:]

def holt_winters(df, param):
    seasonal_periods = 3
    interval = 'h'
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df.dropna(subset=[f'{param}'])
    if param in binary_params:
        df = df.resample(interval)
        data = df[param].value_counts().unstack(fill_value=0)

        count_0 = data[0] if 0 in data.columns else pd.Series(0, index=data.index)
        count_1 = data[1] if 1 in data.columns else pd.Series(0, index=data.index)

        count_0_stationary = ensure_stationarity(count_0)
        count_1_stationary = ensure_stationarity(count_1)
        train_0, test_0 = split_data(count_0_stationary)
        train_1, test_1 = split_data(count_1_stationary)

        model_0 = ExponentialSmoothing(train_0, seasonal=None, seasonal_periods=None).fit()
        model_1 = ExponentialSmoothing(train_1, seasonal=None, seasonal_periods=None).fit()

        forecast_0 = model_0.forecast(len(test_0))
        forecast_1 = model_1.forecast(len(test_1))

        mse = mean_squared_error(test_0, forecast_0)
        print(f"{param} Mean Squared Error (0):", mse)
        mse = mean_squared_error(test_1, forecast_1)
        print(f"{param} Mean Squared Error (1):", mse)

        plt.figure(figsize=(12, 6))
        plt.plot(train_0.index, train_0, label="Train Data", color='blue')
        plt.plot(test_0.index, test_0, label="Test Data", color='green')
        plt.plot(forecast_0.index, forecast_0, label="Forecast", color='red')
        plt.title(f"{param} Holt-Winters (0)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(train_1.index, train_1, label="Train Data", color='blue')
        plt.plot(test_1.index, test_1, label="Test Data", color='green')
        plt.plot(forecast_1.index, forecast_1, label="Forecast", color='red')
        plt.title(f"{param} Holt-Winters (1)")
        plt.legend()
        plt.show()

    elif param in decimal_params:
        data = df[param].resample(interval).mean()
        decimal_data_stationary = ensure_stationarity(data)
        train_decimal, test_decimal = split_data(decimal_data_stationary)

        model = ExponentialSmoothing(train_decimal, seasonal='add', seasonal_periods=seasonal_periods).fit()

        forecast = model.forecast(len(test_decimal))

        mse = mean_squared_error(test_decimal, forecast)
        print("Mean Squared Error:", mse)

        plt.figure(figsize=(12, 6))
        plt.plot(train_decimal.index, train_decimal, label="Train Data", color='blue')
        plt.plot(test_decimal.index, test_decimal, label="Test Data", color='green')
        plt.plot(forecast.index, forecast, label="Forecast", color='red')
        plt.title(f"{param} Holt-Winters")
        plt.legend()
        plt.show()

# Function to filter by facility in 'deviceName'
def is_facility_in_device_name(name, facilities):
    if not facilities:
        return True  # No filtering if facility is not provided
    facilities = [f.strip().lower() for f in facilities.split(",")]
    return any(facility in name.lower() for facility in facilities)

# Function to check if a value matches a list of allowed values
def is_value_in_filter(field_value, allowed_values):
    if not allowed_values:
        return True  # No filtering if allowed values are not provided
    return field_value in allowed_values

def fetch_filter(brand, model, dtype, facility):
    try:
        filters = {}

        if brand:
            filters["deviceBrand"] = brand.split(",")
        if model:
            filters["deviceModel"] = model.split(",")
        if dtype:
            filters["deviceType"] = dtype.split(",")
        

        filtered_data = []
        cursor = collection.find()
        for record in cursor:
            if not is_value_in_filter(record.get("deviceBrand"), filters.get("deviceBrand")):
                continue
            if not is_value_in_filter(record.get("deviceModel"), filters.get("deviceModel")):
                continue
            if not is_value_in_filter(record.get("deviceType"), filters.get("deviceType")):
                continue

            device_name = record.get("deviceName", "")
            if not is_facility_in_device_name(device_name, facility):
                continue
        filtered_data.append(record)
    
        df = pd.DataFrame(filtered_data)
        
        return df, filters
    except:
        print("Error fetching or filtering data.")
    
def filter_data(df, brand, model, dtype, name):
    # Dictionary to store filters
    filters = {}
    
    # Gather user input for each filter
    filters["deviceBrand"] = input("Enter deviceBrand(s) (comma-separated, or leave blank to skip): ").strip().split(",")
    filters["deviceModel"] = input("Enter deviceModel(s) (comma-separated, or leave blank to skip): ").strip().split(",")
    filters["deviceType"] = input("Enter deviceType(s) (comma-separated, or leave blank to skip): ").strip().split(",")
    facility_keywords = input("Enter keywords to search in deviceName (comma-separated, or leave blank to skip): ").strip().split(",")
    
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
        print(f"Filtered DataFrame:\n{df}")
    
    return df

def main():
    intervals = {
        "1": ("h", "Hourly"),
        "2": ("D", "Daily"),
        "3": ("W", "Weekly"),
        "4": ("ME", "Monthly"),
        "5": ("YS", "Yearly")
    }
    
    while True:
        #try:
        df = fetch_data()
        print(f"List of binary parameters: {binary_params}")
        print(f"List of decimal parameters: {decimal_params}")
        
        param = input("Enter the parameter to analyze (choose above): ")
        print("'1' for Hourly")
        print("'2' for Daily")
        print("'3' for Weekly")
        print("'4' for Montly")
        print("'5' for Yearly")
        interval = input("Enter the interval to analyze (choose above): ")

        # Gather user input for each filter
        brand = input("Enter deviceBrand(s) (comma-separated, or leave blank to skip): ")
        model = input("Enter deviceModel(s) (comma-separated, or leave blank to skip): ")
        dtype = input("Enter deviceType(s) (comma-separated, or leave blank to skip): ")
        name = input("Enter keywords to search in deviceName (comma-separated, or leave blank to skip): ")
        
        try:
            if int(interval) == 1:
                start_date = input("Enter start date to analyze in the format (YYYY-MM-DD): ")
                end_date = input("Enter end to analyze in the format (YYYY-MM-DD): ")
                df = df.sort_index()
                data = df.loc[start_date:end_date]
                filtered_df = filter_data(data, brand, model, dtype, )
            else:
                df = df.sort_index()
                filtered_df = filter_data(df)
            interval_result, interval_name = intervals[interval]
        except:
            print("Time interval not found.")
            continue

        if param in binary_params:
            plot_time_series(filtered_df, interval_result, param, f"{param} | {interval_name} State Count")
        elif param in decimal_params:
            plot_time_series(filtered_df, interval_result, param, f"{param} | {interval_name}")
        else:
            print("Parameter not found.")
            continue
        '''except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except:
            continue'''


# Main Program
df = fetch_data()
init_params()
main()
'''for param in binary_params:
    holt_winters(df, param)
for param in decimal_params:
    holt_winters(df, param)'''
'''for param in binary_params:
    decompose(df, param)
for param in decimal_params:
    decompose(df, param)'''
#print(df.columns.values)
#print(df.to_string())
#adf_test(df, 'humidity')
#for param in binary_params:
#    holt(df, param)
#for param in decimal_params:
#    holt(df, param)
'''for param in binary_params:
    acf_pacf(df, param)
for param in decimal_params:
    acf_pacf(df, param)'''
#for param in binary_params:
#    sarimax(df, param)
#for param in decimal_params:
#    sarimax(df, param)
#for param in decimal_params:
#    decompose(df, param)
#for param in binary_params:
#    arima(df, param)
#for param in decimal_params:
#    sma(df, param)
#    cma(df, param)
# Example: Test for stationarity on temperature data
# Determine Door Status
#determine_door_status()
# ARIMA Model
#arima_pred(df)
# SVR
#svr_pred(df)
# Time Series Aggregations
'''
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

        # Yearly Aggregation
        plot_time_series(df, "YS", param,f"Yearly {param} State Count")
    for param in decimal_params:
        # Hourly Aggregation
        plot_time_series(df, "h", param,f"Hourly {param}")

        # Daily Aggregation
        plot_time_series(df, "D", param,f"Daily {param}")

        # Weekly Aggregation
        plot_time_series(df, "W", param,f"Weekly {param}")

        # Monthly Aggregation
        plot_time_series(df, "ME", param,f"Monthly {param}")
        
        # Yearly Aggregation
        plot_time_series(df, "YS", param,f"Yearly {param}")
else:
    print("No data found in MongoDB!")  
'''
'''
# Determine how long the door was open for a particular Time Period
def determine_door_status():
    cursor = collection.find({}, {"_id": 0, "time": 1, "params.Door": 1})
    data = list(cursor)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])  # Convert time to datetime
        df['Door'] = df['params'].apply(lambda x: int(x["Door"]))  # Extract Door state as integer
        df.drop(columns=['params'], inplace=True)  # Drop params column
    # Calculate time differences
    df['duration'] = df['time'].diff().dt.total_seconds().fillna(0)

    # Select a specific time range (e.g., last 24 hours)
    time_filter_start = "2024-12-05 20:00:00"
    time_filter_end = "2024-12-05 23:00:00"
    filtered_df = df[(df['time'] >= time_filter_start) & (df['time'] <= time_filter_end)]

    # Summarize durations for each state
    summary = filtered_df.groupby('Door')['duration'].sum()
    
    # Convert seconds to hours for better readability
    summary_in_hours = summary / 3600
    print("\nDuration the Door was closed (0) or open (1) in hours:")
    print(summary_in_hours)

# ARIMA Model
def arima_pred(df):
    # Resample to hourly intervals, forward-fill missing values
    df = df.resample('W').ffill()

    # Step 2: Train-test split
    X = df.index
    y = df['Door']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Convert training and testing data into a time series format
    train = pd.DataFrame({'time': X_train, 'Door': y_train}).set_index('time')
    test = pd.DataFrame({'time': X_test, 'Door': y_test}).set_index('time')

    # Step 3: Fit the ARIMA model
    model = ARIMA(train['Door'], order=(1, 1, 1))  # Example order; adjust for your dataset
    model_fit = model.fit()

    # Step 4: Forecast on the test data
    forecast = model_fit.forecast(steps=len(test))
    test['Forecast'] = forecast.values

    # Step 5: Evaluate the model using MAPE
    mape = mean_absolute_percentage_error(test['Door'], test['Forecast'])
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}")

    # Step 6: Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Door'], label="Training Data", color='blue')
    plt.plot(test.index, test['Door'], label="Test Data (Original)", color='orange')
    plt.plot(test.index, test['Forecast'], label="Forecast (Model)", color='green', linestyle='dashed')
    plt.axvline(train.index[-1], color='red', linestyle='--', label="Train-Test Split")
    plt.title("ARIMA Model: Forecast vs Original Data")
    plt.xlabel("Time")
    plt.ylabel("Door State (0: Closed, 1: Open)")
    plt.yticks([0, 1], labels=["Closed", "Open"])
    plt.legend()
    plt.grid()
    plt.show()

# Support Vector Regression for prediction
def svr_pred(df):
    # Create lag features
    df['lag_1'] = df['Door'].shift(1).fillna(0)
    df['lag_2'] = df['Door'].shift(2).fillna(0)

    # Prepare data for ML
    X = df[['lag_1', 'lag_2']].values
    y = df['Door'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVR model
    model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon = 0.05)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Convert predictions to binary (threshold = 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Predict future values
    future_lags = np.array([df['Door'][-2:].values])
    future_predictions = []
    for _ in range(10):  # Predict next 10 steps
        next_value = model.predict(future_lags[-1].reshape(1, -1))[0]
        future_predictions.append(next_value)
        future_lags = np.append(future_lags, [[next_value, future_lags[0][-1]]], axis=0)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_test)), y_test, label="Actual")
    plt.plot(range(len(y_test)), y_pred_binary, label="Predicted", linestyle='dashed')
    plt.title('SVR Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Door State')
    plt.legend()
    plt.show()

def log_reg(df):
    df['prev_state'] = df['Door'].shift(1)
    X = df[['prev_state', df.index]]
    y = df['Door']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
'''
