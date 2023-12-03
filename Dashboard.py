import streamlit as st
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import seaborn as sns
import pydeck as pdk
from datetime import datetime

import joblib

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

data = load_data("filtered_data_3.csv")
data = data.dropna(subset=['Time1 of Occurrence'])

## construct data for graphs



# data = pd.read_csv("processed_police_incidents.csv")
graph_data = load_data("crime_data_final.csv")
features = ['Day1 of the Week', 'Time Bin' , 'Zip Code', 'Sector', 'Division', 'X Coordinate', 'Y Cordinate', 'Council District', 'Type  Location']
label = ['Incident_Score']

data_filter = graph_data[features + label]
data_filter = data_filter.dropna()

X = data_filter[features]
Y = data_filter[label]
# print("feature info: \n", X.info())
# print()
# print("label info: \n", Y.info())


# Function to display the "View Data" tab
def view_data():
    st.write('View Data')
    st.write(data)


@st.cache_data
def random_selector(arr):
    selected = np.random.choice(arr, size=1)
    return selected

def input_data():
    st.write('Enter Details of your plan:')

    file_path = './new_data_unique2.csv'
    df = load_data(file_path)

    zip_code_options = df['Zip Code'].unique()
    zip_code_options = [ int(x) for x in zip_code_options ]
    selected_zip_code = st.selectbox('Select Zip Code', zip_code_options)

    filtered_df = df[df['Zip Code'] == selected_zip_code]

    sector_options = filtered_df['Sector'].unique()
    # print(np.array(sector_options))
    selected_sector = random_selector(sector_options)[0]
    # print(np.array(sector_options))
    # selected_sector = st.selectbox('Select Sector', sector_options, index=None, placeholder="Select Sector")

    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

    division_options = filtered_df['Division'].unique()
    selected_division = random_selector(division_options)[0]
    # selected_division = st.selectbox('Select Division', division_options, index=None, placeholder="Select Division")

    filtered_df = filtered_df[filtered_df['Division'] == selected_division]

    x_coordinate_options = filtered_df['X Coordinate'].unique()
    selected_x_coordinate = random_selector(x_coordinate_options)[0]
    # selected_x_coordinate = st.selectbox('Select X Coordinate', x_coordinate_options, index=None, placeholder="Select X Coordinate")

    filtered_df = filtered_df[filtered_df['X Coordinate'] == selected_x_coordinate]

    y_coordinate_options = filtered_df['Y Cordinate'].unique()
    selected_y_coordinate = random_selector(y_coordinate_options)[0]
    # selected_y_coordinate = st.selectbox('Select Y Coordinate', y_coordinate_options, index=None, placeholder="Select Y Coordinate")

    filtered_df = filtered_df[filtered_df['Y Cordinate'] == selected_y_coordinate]

    council_district_options = filtered_df['Council District'].unique()
    selected_council_district = random_selector(council_district_options)[0]
    # selected_council_district = st.selectbox('Select Council District', council_district_options, index=None, placeholder="Select Council District")

    filtered_df = filtered_df[filtered_df['Council District'] == selected_council_district]
    # print(filtered_df)

    type_location_options = filtered_df['Type Location'].unique()
    selected_type_location = random_selector(type_location_options)[0]
    # selected_type_location = st.selectbox('Select Type Location', type_location_options, index=None, placeholder="Select Type Location")

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    user_day = st.selectbox('Select Day:', days)
    day_bins = {
            1: 'Monday',
            5: 'Tuesday', 
            6: 'Wednesday', 
            4: 'Thursday', 
            0: 'Friday', 
            2: 'Saturday', 
            3: 'Sunday'
        }
    user_day_bin = list(day_bins.keys())[list(day_bins.values()).index(user_day)]
    # print(user_day_bin)
    
    user_time = st.text_input('Time:')
    # print(type(user_time))
    user_time_format = '%H:%M'

    if len(user_time) == len(user_time_format) and all(a.isdigit() or b == ':' for a, b in zip(user_time, user_time_format)):
        pass
    else:
        st.write('Invalid time format. Please enter Time in the format HH: mm.')
    time_bin = assign_time_bin2(user_time)
    # input_user_time = assign_time_bin2(user_time)
    # print(time_bin)

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    user_month = st.selectbox('Select Month:', months)
    month_bins = {
            4: 'January', 
            3: 'February', 
            7: 'March', 
            0: 'April', 
            8: 'May', 
            6: 'June', 
            5: 'July', 
            1: 'August', 
            11: 'September', 
            10: 'October', 
            9: 'November', 
            2: 'December'
        }
    user_month_bin = list(month_bins.keys())[list(month_bins.values()).index(user_month)]
    # print(user_month_bin)

    x_pred = [user_day_bin, time_bin, selected_zip_code, selected_sector, selected_division, selected_x_coordinate, selected_y_coordinate, selected_council_district, selected_type_location, user_month_bin]
    
    # x_pred = label_encoding(x_pred)
    column_names = ['user_day_bin', 'time_bin', 'selected_zip_code', 'selected_sector', 'selected_division', 'selected_x_coordinate', 'selected_y_coordinate', 'selected_council_district', 'selected_type_location', 'user_month_bin']

# Create a DataFrame
    df_pred = pd.DataFrame([x_pred], columns=column_names)
    # print(df_pred)
    # print(df_pred)
    # for element, data_type in zip(x_pred, x_pred.dtype):
    #     print(f"{element}: {data_type}")
    y_pred = 0
    

    if st.button('Submit'):
        st.write('Zip Code:', selected_zip_code)
        st.write('Day:', user_day)
        st.write('Time:', user_time)
        st.write('Month:', user_month)
        y_pred = load_and_run_rfc(df_pred)[0]
        safety_bins = {
            4: 'Very Unsafe',
            1: 'Moderately Unsafe',
            0: 'Moderately Safe',
            2: 'Safe',
            3: 'Very Safe'
        }
        safety = safety_bins[y_pred]
        st.write('Safety:', safety)



def load_and_run_rfc(X):
    loaded_model = joblib.load('rfc_model.joblib')

    X_reshaped = np.array(X).reshape(1, -1)
    # Make predictions
    y_predictions = loaded_model.predict(X_reshaped)

    # print("Predictions:", y_predictions)

    return y_predictions

def assign_time_bin2(time):
        # Define time bins
        time_bins = {
            3 : ('18:00', '23:59'),
            1 : ('00:00', '05:59'),
            2 : ('06:00', '11:59'),
            0 : ('12:00', '17:59')
        }
        for bin_label, (start_time, end_time) in time_bins.items():
            if start_time <= time <= end_time:
                return bin_label
        return 'Unknown'



def plot_boxplot_time_bin():
    st.subheader("Boxplot of Incident Scores for Each Time Bin")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_filter, x='Time Bin', y='Incident_Score', palette='viridis')
    plt.title("Distribution of Incident Scores for Each Time Bin")
    plt.xlabel("Time Bin")
    plt.ylabel("Incident Scores")
    st.pyplot(plt)

def plot_boxplot_incident_scores():
    st.subheader("Boxplot of Incident Scores")

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_filter['Incident_Score'], vert=True, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    plt.title("Boxplot of Incident Scores")
    plt.xlabel("Incident Scores")
    st.pyplot(plt)

def plot_distribution_of_incident_scores():
    st.subheader("Distribution of Incident Scores")

    plt.figure(figsize=(10, 6))
    plt.hist(data_filter['Incident_Score'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Incident Scores")
    plt.xlabel("Incident Scores")
    plt.ylabel("Frequency")
    st.pyplot(plt)

def plot_temporal_trends_of_crime():
    st.subheader("Temporal Trends of Crime Incidents")

    temporal_data = X.groupby(['Day1 of the Week', 'Time Bin']).size().reset_index(name='Incident Count')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=temporal_data, x='Day1 of the Week', y='Incident Count', hue='Time Bin', marker='o')
    plt.title("Temporal Trends of Crime Incidents")
    plt.xlabel("Day of the Week")
    plt.ylabel("Incident Count")
    plt.legend(title='Time Bin', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

def plot_geospatial_distribution_of_crime():
    st.subheader("Geospatial Distribution of Crime Incidents")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data_filter, x='X Coordinate', y='Y Cordinate', hue='Incident_Score', palette='viridis', s=50)
    plt.title("Geospatial Distribution of Crime Incidents")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    st.pyplot(plt)

def get_top_10_council_districts(data):
    top_council_districts = data['Council District'].value_counts().nlargest(10)
    return top_council_districts

def plot_top_10_council_districts():
    top_council_districts = get_top_10_council_districts(X)

    st.subheader("Top 10 Council Districts Based on Crime Count")
    fig = px.bar(top_council_districts, x=top_council_districts.index, y=top_council_districts.values, labels={'y': 'Crime Count'})
    fig.update_layout(xaxis_title='Council District', yaxis_title='Crime Count')
    st.plotly_chart(fig)

def get_top_10_locations(data):
    top_locations = data['Type  Location'].value_counts().nlargest(10)
    return top_locations

def plot_top_10_locations():
    top_locations = get_top_10_locations(X)

    st.subheader("Top 10 Locations Based on Crime Count")
    fig = px.bar(top_locations, x=top_locations.index, y=top_locations.values, labels={'y': 'Crime Count'})
    fig.update_layout(xaxis_title='Location', yaxis_title='Crime Count')
    st.plotly_chart(fig)

def calculate_percentage_by_time_bin(data):
    total_crimes = len(data)
    data['Percentage of Crime'] = (data.groupby('Time Bin')['Time Bin'].transform('count') / total_crimes) * 100
    return data.drop_duplicates(subset='Time Bin')

def plot_percentage_by_time_bin():
    X_percentage = calculate_percentage_by_time_bin(X)

    st.subheader("Percentage of Crime by Time Bin")
    fig = px.bar(X_percentage, x='Time Bin', y='Percentage of Crime', labels={'Percentage of Crime': 'Percentage'})
    st.plotly_chart(fig)

def plot_crime_by_weekday():
    crime_by_weekday = X.groupby('Day1 of the Week').size() / len(X) * 100

    fig = px.bar(
        crime_by_weekday,
        x=crime_by_weekday.index,
        y=crime_by_weekday.values,
        labels={'y': 'Percentage of Crime', 'x': 'Day of the Week'},
        title='Percentage of Crime by Weekday',
    )
    st.plotly_chart(fig)

# Function to display the "Select Field" tab
def select_field():
    st.write('Select Field')
    selected_field = st.selectbox('Select an option: ',['Boxplot of Incident Scores for Each Time Bin', 
                                                        'Boxplot of Incident Scores', 
                                                        'Distribution of Incident Scores', 
                                                        'Temporal Trends of Crime Incidents', 
                                                        'Geospatial Distribution of Crime Incidents', 
                                                        'Top 10 Council Districts Based on Crime Count', 
                                                        'Top 10 Locations Based on Crime Count', 
                                                        'Percentage of Crime by Time Bin', 
                                                        'Percentage of Crime by Weekday', 
                                                        #'Day', 'Time', 'ZipCode', 'Co-ordinate density plot', 'Boxplot of Incident Scores', 'Geospatial Distribution of Crime Incidents', 'Temporal Trends of Crime Incidents', 'Distribution of Incident Scores', 'Correlation Matrix'
                                                        ])

    # Display an image associated with the selected field
    # if selected_field == 'Day':
    #     st.image('Incidents-Day.png')
    # elif selected_field == 'Time':
    #     st.image('Incidents-Time.png')
    # elif selected_field == 'ZipCode':
    #     st.image('Incidents-ZipCode.png')
    # elif selected_field == 'Co-ordinate density plot':
    #     st.image('./plots/2D_Density_Plot_of_X_and_Y_Coordinates.png')
    # elif selected_field == 'Boxplot of Incident_Scores':
    #     st.image('./plots/Boxplot_of_Incident_Scores.png')
    # elif selected_field == 'Geospatial Distribution of Crime Incidents':
    #     st.image('./plots/Geospatial_Distribution_of_Crime_Incidents.png')
    # elif selected_field == 'Temporal Trends of Crime Incidents':
    #     st.image('./plots/Temporal_Trends_of_Crime_Incidents.png')
    # elif selected_field == 'Distribution of Incident Scores':
    #     st.image('./plots/Distribution_of_Incident_Scores.png')
    # elif selected_field == 'Correlation Matrix':
    #     st.image('./plots/Correlation_Matrix_Heatmap.png')
    if selected_field == 'Boxplot of Incident Scores for Each Time Bin':
        plot_boxplot_time_bin()
    elif selected_field == 'Boxplot of Incident Scores':
        plot_boxplot_incident_scores()
    elif selected_field == 'Distribution of Incident Scores':
        plot_distribution_of_incident_scores()
    elif selected_field == 'Temporal Trends of Crime Incidents':
        plot_temporal_trends_of_crime()
    elif selected_field == 'Geospatial Distribution of Crime Incidents':
        plot_geospatial_distribution_of_crime()
    elif selected_field == 'Top 10 Council Districts Based on Crime Count':
        plot_top_10_council_districts()
    elif selected_field == 'Top 10 Locations Based on Crime Count':
        plot_top_10_locations()
    elif selected_field == 'Percentage of Crime by Time Bin':
        plot_percentage_by_time_bin()
    elif selected_field == 'Percentage of Crime by Weekday':
        plot_crime_by_weekday()

def ZipCode_DataFrame():
    # Group the data by zip code and count the number of entries for each zip code
    zipcode_counts = data_filter['Zip Code'].value_counts().reset_index()
    zipcode_counts.columns = ['Zip Code', 'Count']

    # Filter out zip codes with fewer than 10,000 entries
    zipcode_counts = zipcode_counts[zipcode_counts['Count'] >= 1000]

    # Sort the data by zip code
    zipcode_counts = zipcode_counts.sort_values(by='Zip Code')
    st.write(zipcode_counts)

def Time_DataFrame():

    # Convert the "Time1 of Occurrence" column to datetime format
    
    # data_filter['Time1 of Occurrence'] = pd.to_datetime(data_filter['Time1 of Occurrence'], format='%H:%M').dt.strftime('%H:%M')
    # print(data['Time1 of Occurrence '][1])
    # Assign time bins to each row in the DataFrame
    # data['Time Bin'] = data['Time1 of Occurrence'].apply(assign_time_bin)
    # print(data['Time1 of Occurrence'])
    # Calculate the counts for each time bin
    time_bin_counts = data_filter['Time Bin'].value_counts().reset_index()
    time_bin_counts.columns = ['Time Bin', 'Count']

    st.write(time_bin_counts)

def Day_DataFrame():
    # Calculate the counts for each day of the week
    day_counts = data_filter['Day1 of the Week'].value_counts().reset_index()
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