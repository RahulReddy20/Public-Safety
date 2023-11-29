import streamlit as st
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import seaborn as sns
import pydeck as pdk

data = pd.read_csv("filtered_data_3.csv")

# data = pd.read_csv("processed_police_incidents.csv")
graph_data = pd.read_csv("crime_data_final.csv")
features = ['Day1 of the Week', 'Time Bin' , 'Zip Code', 'Sector', 'Division', 'X Coordinate', 'Y Cordinate', 'Council District', 'Type  Location']
label = ['Incident_Score']

data_filter = graph_data[features + label]
data_filter = data_filter.dropna()

X = data_filter[features]
Y = data_filter[label]
print("feature info: \n", X.info())
print()
print("label info: \n", Y.info())
# data_filter.to_csv('crime_data_final.csv', index=False)

#######

crime_by_weekday = X.groupby('Day1 of the Week').size() / len(X) * 100

fig = px.bar(
    crime_by_weekday,
    x=crime_by_weekday.index,
    y=crime_by_weekday.values,
    labels={'y': 'Percentage of Crime', 'x': 'Day of the Week'},
    title='Percentage of Crime by Weekday',
)

st.title("Crime Data Analysis")
st.plotly_chart(fig)

#######

def calculate_percentage_by_time_bin(data):
    total_crimes = len(data)
    data['Percentage of Crime'] = (data.groupby('Time Bin')['Time Bin'].transform('count') / total_crimes) * 100
    return data.drop_duplicates(subset='Time Bin')

X_percentage = calculate_percentage_by_time_bin(X)

st.subheader("Percentage of Crime by Time Bin")
fig = px.bar(X_percentage, x='Time Bin', y='Percentage of Crime', labels={'Percentage of Crime': 'Percentage'})
st.plotly_chart(fig)

#######

def get_top_10_locations(data):
    top_locations = data['Type  Location'].value_counts().nlargest(10)
    return top_locations

top_locations = get_top_10_locations(X)

st.subheader("Top 10 Locations Based on Crime Count")
fig = px.bar(top_locations, x=top_locations.index, y=top_locations.values, labels={'y': 'Crime Count'})
fig.update_layout(xaxis_title='Location', yaxis_title='Crime Count')
st.plotly_chart(fig)

#######

def get_top_10_council_districts(data):
    top_council_districts = data['Council District'].value_counts().nlargest(10)
    return top_council_districts

top_council_districts = get_top_10_council_districts(X)

st.subheader("Top 10 Council Districts Based on Crime Count")
fig = px.bar(top_council_districts, x=top_council_districts.index, y=top_council_districts.values, labels={'y': 'Crime Count'})
fig.update_layout(xaxis_title='Council District', yaxis_title='Crime Count')
st.plotly_chart(fig)

#######

st.subheader("Geospatial Distribution of Crime Incidents")
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data_filter, x='X Coordinate', y='Y Cordinate', hue='Incident_Score', palette='viridis', s=50)
plt.title("Geospatial Distribution of Crime Incidents")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
st.pyplot(plt)

#######

st.subheader("Temporal Trends of Crime Incidents")

temporal_data = X.groupby(['Day1 of the Week', 'Time Bin']).size().reset_index(name='Incident Count')

plt.figure(figsize=(10, 6))
sns.lineplot(data=temporal_data, x='Day1 of the Week', y='Incident Count', hue='Time Bin', marker='o')
plt.title("Temporal Trends of Crime Incidents")
plt.xlabel("Day of the Week")
plt.ylabel("Incident Count")
plt.legend(title='Time Bin', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(plt)

#######

st.subheader("Boxplot of Incident Scores for Each Time Bin")

plt.figure(figsize=(10, 6))
sns.boxplot(data=data_filter, x='Time Bin', y='Incident_Score', palette='viridis')
plt.title("Distribution of Incident Scores for Each Time Bin")
plt.xlabel("Time Bin")
plt.ylabel("Incident Scores")
st.pyplot(plt)

#######

st.subheader("Distribution of Incident Scores")

plt.figure(figsize=(10, 6))
plt.hist(data_filter['Incident_Score'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Incident Scores")
plt.xlabel("Incident Scores")
plt.ylabel("Frequency")
st.pyplot(plt)

#######

st.subheader("Boxplot of Incident Scores")

plt.figure(figsize=(10, 6))
plt.boxplot(data_filter['Incident_Score'], vert=True, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.title("Boxplot of Incident Scores")
plt.xlabel("Incident Scores")
st.pyplot(plt)

#######

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
    selected_field = st.selectbox('Select an option: ',['Day', 'Time', 'ZipCode', 'Co-ordinate density plot', 'Boxplot of Incident Scores', 'Geospatial Distribution of Crime Incidents', 'Temporal Trends of Crime Incidents', 'Distribution of Incident Scores', 'Correlation Matrix'])

    # Display an image associated with the selected field
    if selected_field == 'Day':
        st.image('Incidents-Day.png')
    elif selected_field == 'Time':
        st.image('Incidents-Time.png')
    elif selected_field == 'ZipCode':
        st.image('Incidents-ZipCode.png')
    elif selected_field == 'Co-ordinate density plot':
        st.image('./plots/2D_Density_Plot_of_X_and_Y_Coordinates.png')
    elif selected_field == 'Boxplot of Incident_Scores':
        st.image('./plots/Boxplot_of_Incident_Scores.png')
    elif selected_field == 'Geospatial Distribution of Crime Incidents':
        st.image('./plots/Geospatial_Distribution_of_Crime_Incidents.png')
    elif selected_field == 'Temporal Trends of Crime Incidents':
        st.image('./plots/Temporal_Trends_of_Crime_Incidents.png')
    elif selected_field == 'Distribution of Incident Scores':
        st.image('./plots/Distribution_of_Incident_Scores.png')
    elif selected_field == 'Correlation Matrix':
        st.image('./plots/Correlation_Matrix_Heatmap.png')

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
