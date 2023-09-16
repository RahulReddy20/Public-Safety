import streamlit as st
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv("filtered_data_3.csv")

# Function to display the "View Data" tab
def view_data():
    st.write('View Data')
    st.write(data)

# Function to display the "Input Data" tab
def input_data():
    st.write('Enter Details of your plan:')
    user_input1 = st.text_input('Zip Code:')
    user_input2 = st.text_input('Day:')
    user_input3 = st.text_input('Time:')
    
    if st.button('Submit'):
        st.write('Zip Code:', user_input1)
        st.write('Day:', user_input2)
        st.write('Time:', user_input3)
        st.write('Safety Score:', 0)

# Function to display the "Select Field" tab
def select_field():
    st.write('Select Field')
    selected_field = st.selectbox('Select an option: ',['Day', 'Time', 'ZipCode'])

    # Display an image associated with the selected field
    if selected_field == 'Day':
        st.image('Incidents-Day.png')
    elif selected_field == 'Time':
        st.image('Incidents-Time.png')
    elif selected_field == 'ZipCode':
        st.image('Incidents-ZipCode.png')

def ZipCode_DataFrame():
    # Group the data by zip code and count the number of entries for each zip code
    zipcode_counts = data['Zip Code'].value_counts().reset_index()
    zipcode_counts.columns = ['Zip Code', 'Count']

    # Filter out zip codes with fewer than 10,000 entries
    zipcode_counts = zipcode_counts[zipcode_counts['Count'] >= 1000]

    # Sort the data by zip code
    zipcode_counts = zipcode_counts.sort_values(by='Zip Code')
    st.write(zipcode_counts)

def Time_DataFrame():
        
    # Define time bins
    time_bins = {
        'Night': ('18:00', '23:59'),
        'Early Morning': ('00:00', '05:59'),
        'Morning': ('06:00', '11:59'),
        'Afternoon': ('12:00', '17:59')
    }

    # Create a function to assign time bins
    def assign_time_bin(time):
        for bin_label, (start_time, end_time) in time_bins.items():
            if start_time <= time <= end_time:
                return bin_label
        return 'Unknown'

    # Convert the "Time1 of Occurrence" column to datetime format
    data['Time1 of Occurrence'] = pd.to_datetime(data['Time1 of Occurrence'], format='%H:%M').dt.strftime('%H:%M')

    # Assign time bins to each row in the DataFrame
    data['Time Bin'] = data['Time1 of Occurrence'].apply(assign_time_bin)

    # Calculate the counts for each time bin
    time_bin_counts = data['Time Bin'].value_counts().reset_index()
    time_bin_counts.columns = ['Time Bin', 'Count']

    st.write(time_bin_counts)

def Day_DataFrame():
    # Calculate the counts for each day of the week
    day_counts = data['Day1 of the Week'].value_counts().reset_index()
    day_counts.columns = ['Day of the Week', 'Count']
    st.write(day_counts)


def show_dataframe():

    st.write('Data Frame Insights')

    selected_field = st.selectbox('Select an option: ',['Day', 'Time', 'ZipCode'])
    # Display data frame associated with the selected field
    if selected_field == 'Day':
        Day_DataFrame()
    elif selected_field == 'Time':
        Time_DataFrame()
    elif selected_field == 'ZipCode':
        ZipCode_DataFrame()

def main():
    st.title('SafeCity Insights')

    # Tab selection
    tab = st.sidebar.selectbox('Select a tab:', ['View Data', 'Predictions', 'Visualizations', 'DataFrame'])

    if tab == 'View Data':
        view_data()
    elif tab == 'Predictions':
        input_data()
    elif tab == 'Visualizations':
        select_field()
    elif tab == 'DataFrame':
        show_dataframe()

if __name__ == '__main__':
    main()
