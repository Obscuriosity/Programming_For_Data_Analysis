
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress TensorFlow GPU warnings if no GPU is available
tf.config.set_visible_devices([], 'GPU')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error
import pickle

# function to calculate Individual Air Quality Index (IAQI)
def calculate_iaqi(C, pollutant_name):
    breakpoints = {
        'PM2.5': { # Units: ug/m^3 (24-hour average for AQI)
            'C': [0, 35, 75, 115, 150, 250, 500],
            'IAQI': [0, 50, 100, 150, 200, 300, 500]
        },
        'PM10': { # Units: ug/m^3 (24-hour average for AQI)
            'C': [0, 50, 150, 250, 350, 420, 600],
            'IAQI': [0, 50, 100, 150, 200, 300, 500]
        },
        'SO2': { # Units: ug/m^3 (24-hour average for AQI, adjusted for 1601-2100 breakpoint)
            'C': [0, 50, 150, 475, 800, 1600, 2100, 2620],
            'IAQI': [0, 50, 100, 150, 200, 300, 400, 500]
        },
        'NO2': { # Units: ug/m^3 (24-hour average for AQI, but 1-hour breakpoints for higher levels)
            'C': [0, 40, 80, 180, 280, 565, 1130],
            'IAQI': [0, 50, 100, 150, 200, 300, 500]
        },
        'CO': { # Units: mg/m^3 (24-hour average for AQI)
            'C': [0, 2, 4, 14, 24, 36, 60],
            'IAQI': [0, 50, 100, 150, 200, 300, 500]
        },
        'O3': { # Units: ug/m^3 (8-hour average for AQI, but 1-hour breakpoints for higher levels)s)
            'C': [0, 100, 160, 215, 265, 800, 1200],
            'IAQI': [0, 50, 100, 150, 200, 300, 500]
        }
    }

    if pollutant_name not in breakpoints:
        return np.nan

    bp_c = breakpoints[pollutant_name]['C']
    bp_iaqi = breakpoints[pollutant_name]['IAQI']

    # treat negative concentrations as 0 for calculation
    if C < 0:
        C = 0

    # find the breakpoint interval
    for i in range(len(bp_c) - 1):
        if bp_c[i] <= C < bp_c[i+1]:
            C_low = bp_c[i]
            C_high = bp_c[i+1]
            IAQI_low = bp_iaqi[i]
            IAQI_high = bp_iaqi[i+1]

            if C_high == C_low:
                return IAQI_low

            iaqi = ((IAQI_high - IAQI_low) / (C_high - C_low)) * (C - C_low) + IAQI_low
            return round(iaqi)

    # if concentration is greater than or equal to the highest breakpoint
    if C >= bp_c[-1]:
        return bp_iaqi[-1]

    return np.nan

# Data Generation and Preprocessing
@st.cache_data
def generate_all_dataframes():
    # load monitoring station data CSV and make pandas dataframes
    df_dongsi = pd.read_csv('/content/Programming_for_Data_Analysis/Data/PRSA_Data_Dongsi_20130301-20170228.csv')
    df_Gucheng = pd.read_csv('/content/Programming_for_Data_Analysis/Data/PRSA_Data_Gucheng_20130301-20170228.csv')
    df_Changpingzhen = pd.read_csv('/content/Programming_for_Data_Analysis/Data/PRSA_Data_Changping_20130301-20170228.csv')
    df_Huairou = pd.read_csv('/content/Programming_for_Data_Analysis/Data/PRSA_Data_Huairou_20130301-20170228.csv')

    # combine dataframes into one
    df_raw = pd.concat([df_dongsi, df_Gucheng, df_Changpingzhen, df_Huairou])

    # station-wise imputation
    df_station_imputed = df_raw.copy()
    unique_stations = df_station_imputed['station'].unique()
    imputed_dfs_list = []

    for station_name in unique_stations:
        station_df = df_station_imputed[df_station_imputed['station'] == station_name].copy()
        numerical_cols_with_missing_station = station_df.select_dtypes(include=np.number).columns[station_df.select_dtypes(include=np.number).isnull().any()].tolist()

        if numerical_cols_with_missing_station:
            imputer = IterativeImputer(max_iter=10, random_state=0)
            station_df[numerical_cols_with_missing_station] = imputer.fit_transform(station_df[numerical_cols_with_missing_station])

        if 'wd' in station_df.columns and station_df['wd'].isnull().any():
            mode_wd_station = station_df['wd'].mode()[0]
            station_df['wd'] = station_df['wd'].fillna(mode_wd_station)

        imputed_dfs_list.append(station_df)

    df = pd.concat(imputed_dfs_list)

    # Replace negative values with 0 for relevant columns
    pollutant_and_rain_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN']
    for col in pollutant_and_rain_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # convert date and time to 'datetime' format and drop original columns
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.drop(columns=['day', 'hour'])

    # add coarse column
    df['Coarse'] = df['PM10'] - df['PM2.5']
    df['Coarse'] = df['Coarse'].clip(lower=0)
    col_coarse = df.pop('Coarse')
    df.insert(5, col_coarse.name, col_coarse)

    # create a daily DataFrame for AQI calculations
    df_daily = df.groupby(['station', df['datetime'].dt.date]).agg({
        'PM2.5': 'mean',
        'PM10': 'mean',
        'SO2': 'mean',
        'NO2': 'mean',
        'CO': 'mean',
        'O3': 'mean',
        'TEMP': 'mean',
        'PRES': 'mean',
        'DEWP': 'mean',
        'RAIN': 'sum',
        'WSPM': 'mean',
        'wd': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }).reset_index()

    df_daily = df_daily.rename(columns={'datetime': 'date'})

    # convert CO from ug/m^3 to mg/m^3 before calculating IAQI for CO
    df_daily['CO'] = df_daily['CO'] / 1000.0

    # calculate IAQI for each pollutant in the daily df
    pollutants_for_aqi_daily = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    for p in pollutants_for_aqi_daily:
        if p in df_daily.columns:
            df_daily[f'IAQI_{p}'] = df_daily[p].apply(lambda x: calculate_iaqi(x, p))

    # calculate the overall Daily AQI as the maximum of all individual AQIs
    iaqi_columns_daily = [f'IAQI_{p}' for p in pollutants_for_aqi_daily if f'IAQI_{p}' in df_daily.columns]
    if iaqi_columns_daily:
        df_daily['Daily_AQI'] = df_daily[iaqi_columns_daily].max(axis=1)
    else:
        df_daily['Daily_AQI'] = np.nan

    # create df_ann
    df_ann = df_daily[['station', 'date', 'Daily_AQI', 'WSPM', 'wd', 'TEMP']].copy()
    if 'wd_angle' not in df_ann.columns:
        wind_direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        df_ann['wd_angle'] = df_ann['wd'].map(wind_direction_map)

    # Convert datetime columns to string for consistent caching
    if 'date' in df_daily.columns:
        df_daily['date'] = pd.to_datetime(df_daily['date']).dt.round('us').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime']).dt.round('us').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    if 'date' in df_ann.columns:
        df_ann['date'] = pd.to_datetime(df_ann['date']).dt.round('us').dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    return df, df_daily, df_ann

@st.cache_resource
def load_model_and_history():
    model = tf.keras.models.load_model('aqi_prediction_model.keras')
    with open('history.pkl', 'rb') as f:
        history_data = pickle.load(f)
    with open('X_test_scaled.pkl', 'rb') as f:
        X_test_scaled = pickle.load(f)
    with open('y_test_ts_original.pkl', 'rb') as f:
        y_test_ts_original = pickle.load(f)
    with open('y_pred_ts.pkl', 'rb') as f:
        y_pred_ts = pickle.load(f)
    return model, history_data, X_test_scaled, y_test_ts_original, y_pred_ts

# Generate dataframes and load model
df, df_daily, df_ann = generate_all_dataframes()
model, history_data, X_test_scaled, y_test_ts_original, y_pred_ts = load_model_and_history()

# plotting Functions
def plot_distribution_histograms(data_frame):
    columns_to_plot = ['PM2.5', 'Coarse', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for column in columns_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data_frame[column], kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        plt.close(fig)

def plot_time_series_by_station(data_frame, column_name):
    station_colors = sns.color_palette('viridis', n_colors=len(data_frame['station'].unique()))
    color_map = dict(zip(data_frame['station'].unique(), station_colors))

    for station_name in data_frame['station'].unique():
        fig, ax = plt.subplots(figsize=(15, 8))
        station_df = data_frame[data_frame['station'] == station_name].copy()
        # Convert string back to datetime for plotting
        station_df['datetime'] = pd.to_datetime(station_df['datetime'])
        ax.plot(station_df['datetime'], station_df[column_name], label=f'{station_name} - {column_name}', alpha=0.8, color=color_map[station_name])
        ax.set_title(f'{column_name} Over Time for {station_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel(column_name)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

def make_rose_plot(df_to_process, column_name):

    if 'wd_angle' not in df_to_process.columns:
        wind_direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        df_to_process['wd_angle'] = df_to_process['wd'].map(wind_direction_map)

    all_aggregated_data = []
    all_mean_var_values = []

    # Define the order of stations for consistent plotting
    station_order = ['Dongsi', 'Gucheng', 'Changping', 'Huairou']

    for station_name in station_order:
        station_df_plot = df_to_process[df_to_process['station'] == station_name].copy()
        if not station_df_plot.empty:
            grouped_data = station_df_plot.groupby('wd_angle').agg(
                frequency=(column_name, 'count'),
                mean_var=(column_name, 'mean')
            ).reset_index()
            all_aggregated_data.append((station_name, grouped_data))
            all_mean_var_values.extend(grouped_data['mean_var'].tolist())

    global_min_var = np.min(all_mean_var_values) if all_mean_var_values else 0
    global_max_var = np.max(all_mean_var_values) if all_mean_var_values else 1

    norm = colors.Normalize(vmin=global_min_var, vmax=global_max_var)
    cmap = cm.magma

    fig, axes = plt.subplots(2, 2, figsize=(18, 18), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()
    fig.suptitle(f'Pollution Rose: Wind Frequency (Length) and {column_name} Concentration (Color) by Wind Direction', fontsize=18)

    for i, (station_name, grouped_data) in enumerate(all_aggregated_data):
        ax = axes[i]
        grouped_data = grouped_data.sort_values('wd_angle')
        radians = np.deg2rad(grouped_data['wd_angle'])
        frequencies = grouped_data['frequency']
        concentrations = grouped_data['mean_var']
        bar_colors = cmap(norm(concentrations))

        ax.bar(radians, frequencies, width=np.deg2rad(22.5),
             bottom=0.0, color=bar_colors, edgecolor='black', alpha=0.8)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'{station_name}', va='bottom', fontsize=16, y=1.05)
        ax.set_xlabel('Wind Direction', labelpad=20)
        ax.set_ylabel('Frequency (Number of Observations)', labelpad=30)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.05)
    cbar.set_label(f'Average {column_name} Concentration (ug/m^3)', rotation=270, labelpad=20)
    st.pyplot(fig)
    plt.close(fig)

def plot_correlation_heatmap(data_frame, title_suffix=""):
    correlation_cols = ['PM2.5', 'Coarse', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

    if 'Coarse' not in data_frame.columns:
        correlation_cols.remove('Coarse')

    if 'Daily_AQI' in data_frame.columns:
        correlation_cols.append('Daily_AQI')
    correlation_matrix = data_frame[correlation_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title(f'Correlation Heatmap of Pollutants and Meteorological data {title_suffix}')
    st.pyplot(fig)
    plt.close(fig)

def plot_aqi_boxenplot(data_frame):
    fig, ax = plt.subplots(figsize=(12, 7))
    station_order = ['Dongsi', 'Gucheng', 'Changping', 'Huairou']
    sns.boxenplot(x='station', y='Daily_AQI', data=data_frame, hue='station', palette='viridis', legend=False, order=station_order, ax=ax)
    ax.set_title('Daily AQI Distribution by Station')
    ax.set_xlabel('Station')
    ax.set_ylabel('Daily AQI')
    ax.set_ylim(0, 550)
    ax.set_yticks(np.arange(0, 551, 50))
    ax.grid(axis='y', linestyle='--', alpha=0.9)
    st.pyplot(fig)
    plt.close(fig)


# app Layout
st.set_page_config(layout="wide", page_title="Air Quality Analysis in Beijing")

st.title('Air Quality Analysis in Beijing (2013-2017)')
st.sidebar.title('Navigation')

page = st.sidebar.radio("Go to", ["Dataset Overview", "Data Visualizations", "Model Outputs"])

# page: Dataset Overview
if page == "Dataset Overview":
    st.header("Dataset Overview")

    st.subheader("Column Definitions")
    st.markdown("""
    **No** - is simply the row number.

    **Year, Month, day and Hour** - are self explanatory records of when the samples were taken

    **PM2.5** - are particulate matter below 2.5 micrometres measured in ug/m^3, these fine particles can penetrate deeply into the lungs.

    **PM10** - This value includes PM2.5 and is a record of particulate matter below 10 micrometres measured in ug/m^3. One should subtract the PM2.5 value to get coarse and fine particles separately.

    **Coarse** - PM10 particulate matter with the PM2.5 content removed, measured in ug/m^3.

    **Daily_AQI** - is the Air Quality Index for the day, using the chinese standards.

    **SO2** - Sulphur Dioxide  measured in ug/m^3.

    **NO2** - Nitrogen Dioxide,  measured in ug/m^3.

    **CO** - Carbon Monoxide  measured in ug/m^3.

    **O3** - Ozone concentration measured in ug/m^3.

    **TEMP** - is the local temperature in Celsius.

    **PRES** - is local air pressure in hPa.

    **DEWP** - is the dew point in Celsius and gives a good indication of humidity, a high value indicates high humidity.

    **RAIN** - gives rainfall measured in mm.

    **wd** - indicates the wind direction using compass points.

    **wd_angle** - is the wind direction in degrees.

    **WSPM** - is the wind speed in m/s.

    **station** - shows the station where the readings were taken.
    """)

    st.subheader("Main DataFrame (df) Info")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue(), language='python')

    st.subheader("Daily Aggregated DataFrame (df_daily) Info")
    buffer_daily = io.StringIO()
    df_daily.info(buf=buffer_daily)
    st.code(buffer_daily.getvalue(), language='python')

    st.subheader("ANN Preparation DataFrame (df_ann) Info")
    buffer_ann = io.StringIO()
    df_ann.info(buf=buffer_ann)
    st.code(buffer_ann.getvalue(), language='python')

    st.subheader("Main DataFrame (df) Statistical Summary")
    st.markdown(df.describe().to_html(escape=False), unsafe_allow_html=True)

# page: Data Visualizations
elif page == "Data Visualizations":
    st.header("Data Visualizations")

    plot_type = st.selectbox(
        "Select a visualization type:",
        [
            "Distribution Histograms (Pollutants & Weather)",
            "PM2.5 & Coarse Concentrations Over Time by Station",
            "Daily AQI Over Time by Station",
            "Correlation Heatmaps",
            "Pollution Roses (PM2.5 & Daily AQI)",
            "PM2.5 Boxplots by Month",
            "Daily AQI Distribution by Station"
        ]
    )

    if plot_type == "Distribution Histograms (Pollutants & Weather)":
        st.subheader("Distribution Histograms")
        plot_distribution_histograms(df)
    elif plot_type == "PM2.5 & Coarse Concentrations Over Time by Station":
        st.subheader("PM2.5 and Coarse Concentrations Over Time")
        for station_name in df['station'].unique():
            fig, ax = plt.subplots(figsize=(15, 8))
            station_df = df[df['station'] == station_name].copy()
            # Convert string back to datetime for plotting
            station_df['datetime'] = pd.to_datetime(station_df['datetime'])
            ax.scatter(station_df['datetime'], station_df['PM2.5'], label=f'{station_name} PM2.5', alpha=0.5, s=10)
            ax.scatter(station_df['datetime'], station_df['Coarse'], label=f'{station_name} Coarse', alpha=0.5, s=10)
            ax.set_xlabel('Date')
            ax.set_ylabel('Concentration (ug/m^3)')
            ax.set_title(f'PM2.5 and Coarse Concentrations Over Time for {station_name}')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
    elif plot_type == "Daily AQI Over Time by Station":
        st.subheader("Daily AQI Over Time")
        station_colors = sns.color_palette('viridis', n_colors=len(df_daily['station'].unique()))
        color_map = dict(zip(df_daily['station'].unique(), station_colors))
        for station_name in df_daily['station'].unique():
            fig, ax = plt.subplots(figsize=(15, 8))
            station_df_daily_plot = df_daily[df_daily['station'] == station_name].copy()
            # Convert string back to datetime for plotting
            station_df_daily_plot['date'] = pd.to_datetime(station_df_daily_plot['date'])
            ax.plot(station_df_daily_plot['date'], station_df_daily_plot['Daily_AQI'], label=f'{station_name} Daily AQI', color=color_map[station_name])
            ax.set_xlabel('Date')
            ax.set_ylabel('Daily AQI')
            ax.set_title(f'Daily AQI for {station_name} Over Time')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
    elif plot_type == "Correlation Heatmaps":
        st.subheader("Correlation Heatmap - Hourly Data")
        plot_correlation_heatmap(df, "(Hourly)")
        st.subheader("Correlation Heatmap - Daily Data with AQI")
        plot_correlation_heatmap(df_daily, "(Daily with AQI)")
    elif plot_type == "Pollution Roses (PM2.5 & Daily AQI)":
        st.subheader("Pollution Rose: Wind Frequency and PM2.5 Concentration")
        make_rose_plot(df, 'PM2.5')
        st.subheader("Pollution Rose: Wind Frequency and Daily AQI")
        make_rose_plot(df_daily, 'Daily_AQI')
    elif plot_type == "PM2.5 Boxplots by Month":
        st.subheader("PM2.5 Concentration by Month for Each Station")
        station_df_all = df.copy()
        station_df_all['datetime'] = pd.to_datetime(station_df_all['datetime'])
        station_df_all['month_name'] = station_df_all['datetime'].dt.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for station_name in df['station'].unique():
            fig, ax = plt.subplots(figsize=(15, 8))
            station_df = station_df_all[station_df_all['station'] == station_name]
            sns.boxplot(x='month_name', y='PM2.5', data=station_df, order=month_order, hue='month_name', palette='viridis', legend=False, ax=ax)
            ax.set_title(f'PM2.5 Concentration by Month for {station_name}')
            ax.set_xlabel('Month')
            ax.set_ylabel('PM2.5 Concentration (ug/m^3)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)
    elif plot_type == "Daily AQI Distribution by Station":
        st.subheader("Daily AQI Distribution by Station")
        plot_aqi_boxenplot(df_daily)

# page: Model Outputs
elif page == "Model Outputs":
    st.header("Model Outputs")

    st.subheader("Model Evaluation Metrics")

    # Convert string dates back to datetime for evaluation if needed by original code logic
    # For direct use of y_test_ts_original (already numeric), this conversion is not strictly needed here
    # but important for consistency with how it was created from df_daily.date

    st.write(f'Test Loss (Huber on log(AQI+1)): {model.evaluate(X_test_scaled, np.log1p(y_test_ts_original), verbose=0)[0]:.4f}')
    st.write(f'Test Mean Absolute Error (on AQI scale): {np.mean(np.abs(y_test_ts_original - y_pred_ts)):.4f}')

    st.subheader("Training History (Loss and MAE)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(history_data['loss'])
    ax1.plot(history_data['val_loss'])
    ax1.set_title('Model Loss (Huber on log(AQI+1))')
    ax1.set_ylabel('Loss (Huber on log(AQI+1))')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')

    ax2.plot(history_data['mean_absolute_error'])
    ax2.plot(history_data['val_mean_absolute_error'])
    ax2.set_title('Model Mean Absolute Error (on log(AQI+1))')
    ax2.set_ylabel('MAE (on log(AQI+1))')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Actual vs. Predicted Daily AQI")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_ts_original, y_pred_ts, alpha=0.3)
    ax.plot([y_test_ts_original.min(), y_test_ts_original.max()], [y_test_ts_original.min(), y_test_ts_original.max()], 'r--') # Diagonal line
    ax.set_xlabel('Actual Daily AQI')
    ax.set_ylabel('Predicted Daily AQI')
    ax.set_title('Actual vs. Predicted Daily AQI (Original Scale)')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
