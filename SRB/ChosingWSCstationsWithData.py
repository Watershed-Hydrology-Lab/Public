#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
import numpy as np
import shutil
from calendar import monthrange

# Define the criteria function
def meets_criteria(data, start_date, end_date):
    data1 = data.copy()  # Create a copy of the data to avoid modifying the original DataFrame
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Criteria 1: Period selected
    data1['Date'] = pd.to_datetime(data1['Date'])

    # Check if data starts from start year or start date and ends at end year or end date
    criteria_1 = (data1['Date'].min().year <= start_date.year and data1['Date'].max().year >= end_date.year) or \
                  (data1['Date'].min() <= start_date and data1['Date'].max() >= end_date)
    
    data_filtered = data1[(data1['Date'] >= start_date) & (data1['Date'] <= end_date)]

    # Check if any data is available for the selected period
    criteria_1 = criteria_1 and not data_filtered.empty
    
    # Criteria 2: > 90% of data should be present in any month
    total_days_per_month = data_filtered.groupby(['Year', 'Month'])['Days_in_Month'].first()

    # Calculate the ratio of available days to total days in each month for each year
    monthly_data_ratio = data_filtered.groupby(['Year', 'Month']).size() / total_days_per_month

    criteria_2 = all(monthly_data_ratio >= 0.92)

    # Criteria 3: No more than 3 continuous missing days in any water year
    start_year = int(start_date.year)
    end_year = int(end_date.year)
    criteria_3_met = True  # Initialize the flag for Criteria 3
    for year in range(start_year, end_year + 1):
        start_date_year = f"{year}-10-01"
        end_date_year = f"{year + 1}-09-30"
        year_data = data_filtered[(data_filtered['Date'] >= start_date_year) & (data_filtered['Date'] <= end_date_year)]
        
        missing_sequence = []
        continuous_missing = 0

        for i in range(len(year_data) - 1):
            current_date = pd.to_datetime(year_data.iloc[i]['Date'])
            next_date = pd.to_datetime(year_data.iloc[i + 1]['Date'])
            if (next_date - current_date).days > 3:
                continuous_missing += (next_date - current_date).days - 1
            else:
                if continuous_missing > 0:
                    missing_sequence.append(continuous_missing)
                    continuous_missing = 0
        
        # Check if there are more than 3 continuous missing dates
        if any(num_days >3 for num_days in missing_sequence):
            print(f"Missed water year {year}")
            criteria_3_met = False  # Update the flag if Criteria 3 is not met for this year

    # Update Criteria 3 based on the flag
    criteria_3 = criteria_3_met

    # Calculate the overall data gap in terms of water years
    water_years = data_filtered.groupby(data_filtered['Date'].dt.to_period("A-SEP"))  # Group by water years (starting from October)
    days_per_water_year = water_years.size()  # Count the number of days in each water year

    # Check if the period is > 30 years
    #if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days > 30 * 365.25:
    water_years_count = len(days_per_water_year)
    overall_gap= year_difference -water_years_count 
    if overall_gap > 10:
            print(f"overall data gap > 10 water years criteria4 not met.")
            print(overall_gap)
            criteria_4 = False
    else:
        criteria_4 = True  # If the period is <= 30 years, Criteria 4 is automatically met

    return criteria_1, criteria_2, criteria_3, criteria_4

# Function to print if criteria are met or not and copy file if criteria are met
def print_and_copy_criteria(criteria_1, criteria_2, criteria_3, criteria_4, input_path, output_directory):
    print("Criteria 1 (data available for the selected period):", "Met" if criteria_1 else "Not Met")
    print("Criteria 2 (> 90% data present in any month):", "Met" if criteria_2 else "Not Met")
    print("Criteria 3 (no more than 3 continuous missing days in a water year):", "Met" if criteria_3 else "Not Met")
    print("Criteria 4 (overall data gap within 10 years for a > 30-year period):", "Met" if criteria_4 else "Not Met")

    if criteria_1 and criteria_2 and criteria_3 and criteria_4:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_directory, filename)
        shutil.copy(input_path, output_path)
        print("File copied to output directory:", output_path)

# Specify the input and output directories
input_directory = r"C:\Users\TIWARIDI\OneDrive - Government of Ontario\Documents\WSCdata\GaugeSelection\R\stations\active_station"
output_directory = os.path.join(input_directory, "CNP1990_2020")

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Specify the start and end dates for the selected period
start_date = '1990-10-01'
end_date = '2020-09-30'

# Iterate through CSV files, filter, and save
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        data = pd.read_csv(file_path)  # Load the CSV file into a DataFrame

        # Calculate the number of days in each month (accounting for leap years)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Days_in_Month'] = data['Date'].dt.days_in_month

        # Check criteria and print status, and copy if criteria are met
        year_difference = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
        #print(year_difference)
         
      # Check criteria and print status, and copy if criteria are met
        criteria_1, criteria_2, criteria_3, criteria_4 = meets_criteria(data, start_date, end_date)
        print("Checking criteria for", filename)
        print_and_copy_criteria(criteria_1, criteria_2, criteria_3, criteria_4, file_path, output_directory)


# In[ ]:




