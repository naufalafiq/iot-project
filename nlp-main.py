import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import re
from datetime import datetime, timedelta
from fpdf import FPDF
from io import BytesIO
import os
import tempfile

# MongoDB connection setup
client = MongoClient('<MongoDB>')
db = client['asta']
collection = db['telemetrix']

# Function to convert date string to ISO format string for MongoDB
def convert_date(date_str, is_end=False):
    #return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00")
    try:
        if date_str is None:
            return None
        # Case when the date string includes time
        if "T" in date_str:
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        else:
            # Default case for standard dates
            date = datetime.strptime(date_str, "%Y-%m-%d")

        # If is_end is True, set time to 23:59:59 (end of the day)
        if is_end:
            return date.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%S")
        else:
            # Set time to 00:00:00 (start of the day)
            return date.replace(hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S")
    
    except ValueError as e:
        raise ValueError(f"Invalid date format for '{date_str}': {e}")

# Function to parse user input
def parse_input(user_input):

    # Check for the 'expected' keyword and remove it if present
    display_type = "counts"  # Default to "counts"
    if "expected" in user_input:
        display_type = "expected"
        user_input = user_input.replace("expected", "")  # Remove "expected" from the input

    # Regular expression to match the full input pattern with from-to date range
    match_full = re.match(r"([\w, ]+)\s*(hourly|daily|weekly|monthly)\s*from\s*(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})\s*(expected)?", user_input)
    if match_full:
        params = match_full.group(1).split(",")  # Split by commas to handle multiple parameters
        interval = match_full.group(2)
        start_date = match_full.group(3)
        end_date = match_full.group(4)
        
        # Clean up spaces around parameter names
        params = [param.strip() for param in params]
        
        print(f"Input - Date Range, Params: {params}, Interval: {interval}, Start: {start_date}, End: {end_date}, Display Type: {display_type}")
        return params, interval, start_date, end_date, display_type

    # Match when only one date is specified (single date, no end date)
    match_single = re.match(r"([\w, ]+)\s*(hourly|daily|weekly|monthly)\s*from\s*(\d{4}-\d{2}-\d{2})\s*(expected)?", user_input)
    if match_single:
        params = match_single.group(1).split(",")  # Split by commas
        interval = match_single.group(2)
        start_date = match_single.group(3)
        end_date = None
        
        # Clean up spaces around parameter names
        params = [param.strip() for param in params]
        
        print(f"Input - Single Date - Params: {params}, Interval: {interval}, Start: {start_date}, Display Type: {display_type}")
        return params, interval, start_date, end_date, display_type

    # Case when no dates are specified (fetch all data)
    params = user_input.split()[0].split(",")
    interval = user_input.split()[1]
    params = [param.strip() for param in params]  # Clean up spaces around parameter names
    print(f"Input - No Date - Params: {params}, Interval: {interval}, Display Type: {display_type}")
    return params, interval, None, None, display_type  # Default to "counts"


# Function to query MongoDB data for the params field and a specific date range
def query_data(start_date=None, end_date=None):
    start_iso = convert_date(start_date)

    # If no date is provided, query for the entire dataset
    if not start_date or not end_date:
        # To fetch all data, we can either query by range with no specific limit or set a very broad query
        print(f"Querying all...")
        return list(collection.find())
    else:
        # If only one date is provided, adjust the end_date to the end of that day
        if start_date == end_date:
            # Set end_date to the last moment of the given date (23:59:59)
            end_iso = convert_date(start_date, is_end=True)
        else:
            # Convert end_date to ISO format (end of the day)
            end_iso = convert_date(end_date, is_end=True)
    
        # MongoDB query to retrieve data in the specified range
        query = {
            "time": {"$gte": start_iso, "$lte": end_iso},
            "params": {"$exists": True}  # Retrieve documents containing the 'params' field
        }
        print(f"Query: {query}")

    return list(collection.find(query)), query

# Function to process binary parameters (0 and 1 counts, expected frequency) as a time series
def process_binary_data(df, param, interval, display_type, output_dir, pdf):
    df = df.copy()
    # Ensure that the time column is in datetime format and set it as the index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Filter data for the specified parameter and convert string values ('0', '1') to integers
    df[param] = df['params'].apply(lambda x: int(x.get(param, '0')) if param in x else None)  # Convert to int

    # Only keep rows where the parameter is binary (0 or 1)
    df = df[df[param].isin([0, 1])]

    # If there are no valid rows for the binary parameter, return early and print a message
    if df.empty:
        print(f"No binary data found for parameter: {param}")
        return

    # Custom aggregation functions for counting 0's and 1's
    def count_1(x):
        return (x == 1).sum()

    def count_0(x):
        return (x == 0).sum()
    
    # Initialize df_resampled as None
    df_resampled = None
    
    # Resample data based on the given interval and count both 0's and 1's
    if interval == 'hourly':
        df_resampled = df.resample('h').agg({param: [count_1, count_0]})
    elif interval == 'daily':
        df_resampled = df.resample('D').agg({param: [count_1, count_0]})
    elif interval == 'weekly':
        df_resampled = df.resample('W').agg({param: [count_1, count_0]})
    elif interval == 'monthly':
        df_resampled = df.resample('M').agg({param: [count_1, count_0]})

    # Ensure proper handling of multi-level column names
    df_resampled.columns = [f'{param}_{agg}' for agg in df_resampled.columns.get_level_values(1)]

    print(df_resampled.head())  # Debugging output

    # Remove any points where both count_0 and count_1 are 0 (i.e., no data for that interval)
    df_resampled = df_resampled[(df_resampled[f'{param}_count_0'] > 0) | (df_resampled[f'{param}_count_1'] > 0)]

    # Extract counts of 0 and 1 for plotting
    counts_1 = df_resampled[f'{param}_count_1']  # Count of 1's
    counts_0 = df_resampled[f'{param}_count_0']  # Count of 0's

    # Calculate expected frequency for binary events (1 or 0)
    # Probability for previous event being 1 and 0
    prev_prob_1 = counts_1.shift(1) / (counts_1.shift(1) + counts_0.shift(1))  # Probability of previous event being 1
    prev_prob_0 = counts_0.shift(1) / (counts_1.shift(1) + counts_0.shift(1))  # Probability of previous event being 0

    # Expected frequency based on probability for 1 and 0
    expected_freq_1 = prev_prob_1 * (counts_0 + counts_1)  # Expected frequency for 1 based on probability
    expected_freq_0 = prev_prob_0 * (counts_0 + counts_1)  # Expected frequency for 0 based on probability

# Add a new page for the dataframe summary
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Summary Table for {param} ({interval.capitalize()})", ln=True, align="C")
    
    # Convert the dataframe to text and add to the PDF
    pdf.set_font("Courier", size=9)
    table_text = df_resampled.to_string()
    pdf.multi_cell(0, 10, table_text)

    # Add a new page for the plot
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Plot for Binary Parameter '{param}' ({interval.capitalize()})", ln=True, align="C")

    # Display based on user input (either counts or expected frequencies)
    if display_type == "expected":
        # Plot the expected frequencies for 0's and 1's as scatter points
        plt.figure(figsize=(12, 6))
        plt.scatter(df_resampled.index, counts_0, label='Count 0', color='blue', marker='o', alpha=0.6)
        plt.scatter(df_resampled.index, counts_1, label='Count 1', color='red', marker='o', alpha=0.6)
        plt.scatter(df_resampled.index, expected_freq_1, label='Expected Frequency for 1', color='green', marker='x', alpha=0.6)
        plt.scatter(df_resampled.index, expected_freq_0, label='Expected Frequency for 0', color='purple', marker='x', alpha=0.6)
    elif display_type == "counts":
        # Plot the counts of 0's and 1's as scatter points
        plt.figure(figsize=(12, 6))
        plt.plot(df_resampled.index, counts_0, label='Count 0', color='blue', marker='o', alpha=0.6)
        plt.plot(df_resampled.index, counts_1, label='Count 1', color='red', marker='o', alpha=0.6)

    # Add titles and labels
    plt.title(f"Parameter '{param}' - {display_type.capitalize()} ({interval})")
    plt.xlabel("Time")
    plt.ylabel("Count / Expected Frequency")
    plt.legend()
    plt.grid(True)

    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file, format='png')
        plot_file_path = tmp_file.name

    # Add the plot to the PDF
    pdf.image(plot_file_path, x=8, y=25, w=160)

    # Save resampled data as CSV
    resampled_file_path = os.path.join(output_dir, f'{param}_binary_resampled.csv')
    df_resampled.to_csv(resampled_file_path)

    plt.show()
    plt.close()


# Function to process decimal parameters (min, mean, max) as a time series
def process_decimal_data(df, param, interval, output_dir, pdf):
    df = df.copy()

    # Convert the time column to datetime and set it as the index
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter data for the specified parameter
    df[param] = df['params'].apply(lambda x: x.get(param, None))  # Extract the specific parameter

    # Convert the parameter column to numeric, coercing errors (non-numeric values will be set to NaN)
    df[param] = pd.to_numeric(df[param], errors='coerce')

    # Only keep rows where the parameter is a valid number (non-NaN)
    df = df[df[param].notnull()]

    df_resampled = df

    # Resample data based on the given interval
    if interval == 'hourly':
        df.set_index('time', inplace=True)
        df_resampled = df.resample('h').agg({param: ['min', 'mean', 'max']})
    elif interval == 'daily':
        df.set_index('time', inplace=True)
        df_resampled = df.resample('D').agg({param: ['min', 'mean', 'max']})
    elif interval == 'weekly':
        df.set_index('time', inplace=True)
        df_resampled = df.resample('W').agg({param: ['min', 'mean', 'max']})
    elif interval == 'monthly':
        df.set_index('time', inplace=True)
        df_resampled = df.resample('M').agg({param: ['min', 'mean', 'max']})

    print(df_resampled.head())

# Add a new page for the dataframe summary
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Summary Table for {param} ({interval.capitalize()})", ln=True, align="C")
    
    # Convert the dataframe to text and add to the PDF
    pdf.set_font("Courier", size=9)
    table_text = df_resampled.to_string()
    pdf.multi_cell(0, 10, table_text)

    # Add a new page for the plot
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Plot for Decimal Parameter '{param}' ({interval.capitalize()})", ln=True, align="C")

    # Plot min, mean, and max as time series
    plt.figure(figsize=(12, 6))
    df_resampled[param, 'min'].plot(label='Min', color='blue', marker='o')
    df_resampled[param, 'mean'].plot(label='Mean', color='green', marker='o')
    df_resampled[param, 'max'].plot(label='Max', color='red', marker='o')
    plt.title(f"Parameter '{param}' Statistics ({interval})")
    plt.xlabel("Time")
    plt.ylabel(param)
    plt.legend()
    plt.grid(True)

    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file, format='png')
        plot_file_path = tmp_file.name

    # Add the plot to the PDF
    pdf.image(plot_file_path, x=8, y=25, w=160)

    # Save resampled data as CSV
    resampled_file_path = os.path.join(output_dir, f'{param}_decimal_resampled.csv')
    df_resampled.to_csv(resampled_file_path)

    plt.show()
    plt.close()


# Function to determine the date range
def get_date_range(start_date, interval):
    if start_date is None:
        start = None
        end = None
        return start, end
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    
    if interval == 'hourly':
        # For hourly, use the single day as the range
        end = start + timedelta(days=1) - timedelta(seconds=1)  # End of the same day
    elif interval == 'daily':
        # For daily, use the week (Monday to Sunday) that contains the date
        start = start - timedelta(days=start.weekday())  # Start of the week
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)  # End of the week
    elif interval == 'weekly':
        # For weekly, use the week that contains the date (one point)
        end = start + timedelta(days=6)  # End of the week
    elif interval == 'monthly':
        # For monthly, use the entire month of the date
        start = start.replace(day=1)  # Start of the month
        next_month = (start.month % 12) + 1
        year = start.year + (start.month // 12)
        end = datetime(year, next_month, 1) - timedelta(seconds=1)  # Last second of the current month
    else:
        raise ValueError("Invalid interval")
    
    if start_date is None:
        end = datetime(2026, 12, 31)
    
    return start.isoformat(), end.isoformat()

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

def cleanup_output_dir(output_dir, extensions_to_remove=[".png", ".csv"]):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        # Check if the file has an extension that needs to be removed
        if any(file.endswith(ext) for ext in extensions_to_remove):
            os.remove(file_path)
            print(f"Removed temporary file: {file_path}")
    print("Cleanup completed.")

user_input = input("Enter your query: ")
params, interval, start_date, end_date, display_type = parse_input(user_input)

if not end_date:
    # Get the appropriate date range if only one date is specified
    start_date, end_date = get_date_range(start_date, interval)

# Query MongoDB data for the given time range and all params
data, query = query_data(start_date, end_date)

# Convert to DataFrame for further processing
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])
min_date = df['time'].min()
max_date = df['time'].max()
current_time = datetime.now()
report_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

# Example parameter lists (binary and decimal)
binary_params = ['door', 'motion']
decimal_params = ['temperature', 'humidity', 'voltage']
string_params = ['outlet']
init_params()

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="IoT Data Report", ln=True, align="C")
output_dir = ".\\Report"
os.makedirs(output_dir, exist_ok=True)
# Add a title page with additional information
pdf.ln(20)  # Add a line break

# User input and query
pdf.set_font("Arial", size=10)
pdf.cell(200, 10, txt=f"User Input: {user_input}", ln=True)
pdf.cell(200, 10, txt=f"Query Generated: {query}", ln=True)

# Time frame of the data
pdf.cell(200, 10, txt=f"Time Frame: {min_date} to {max_date}", ln=True)

# Report generation time
pdf.cell(200, 10, txt=f"Report Generated At: {report_time}", ln=True)

# Iterate over all provided parameters and process them
for param in params:
    if param in binary_params:  # Add more binary params as needed
        process_binary_data(df, param, interval, display_type, output_dir, pdf)
    elif param in decimal_params:
        process_decimal_data(df, param, interval, output_dir, pdf)

# Save the PDF to the output directory
pdf_output_path = os.path.join(output_dir, 'IoT_Data_Report.pdf')
pdf.output(pdf_output_path)
cleanup_output_dir(output_dir)

print(f"Report saved to: {pdf_output_path}")
