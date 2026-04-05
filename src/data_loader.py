import os
import pandas as pd
from src.random_function import get_first_time_step, get_last_time_step, get_last_day_of_month
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset, random_split
import torch
import holidays
#from sklearn.metrics.pairwise import cosine_similarity


class covariateDataLoader:
    """Loads and processes weather data for training and testing."""
    
    def __init__(self, data_path, start_month, end_month):
        self.data_path = data_path
        self.data_events = None
        self.data_calendar = None
        self.data_weather = None
        self.start = pd.to_datetime(start_month, format="%Y-%m")
        # end is the last day of the month of end_month at 23:59
        self.end = pd.to_datetime(end_month, format="%Y-%m") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
        
        #self.end = pd.to_datetime(end_month, format="%Y-%m") + pd.offsets.MonthEnd(0)
        
        #self.load_data()
        #self.preprocess_holiday_data()
        self.load_events_data()
        self.load_weather_data()
        
        self.data_events.to_csv("data/processed/events_zh_raw.csv", index=True)
        self.data_calendar.to_csv("data/processed/calendar_zh_raw.csv", index=True)
        self.data_weather.to_csv("data/processed/weather_zh_raw.csv", index=True)
        
        self.normalize_standard()
        self.data_events.to_csv("data/processed/events_zh.csv", index=True)
        self.data_calendar.to_csv("data/processed/calendar_zh.csv", index=True)
        self.data_weather.to_csv("data/processed/weather_zh.csv", index=True)
        
        
    def load_events_data(self):
        data_path_suffix = "events"
        df_holiday = pd.read_csv(os.path.join(self.data_path, data_path_suffix, "zurich_school_and_public_holidays_2022_2025.csv"))
        df_events = pd.read_csv(os.path.join(self.data_path, data_path_suffix, "zurich_events.csv"), sep=";")
        
        df_holiday["Start Date"] = pd.to_datetime(df_holiday["Start Date"])
        df_holiday["End Date"] = pd.to_datetime(df_holiday["End Date"])
        
        # get index from self.data
        # Generate pd df with index of hour in the range self.start to self.end
        print(self.start, self.end)
        df_flagged = pd.DataFrame(index=pd.date_range(start=self.start, end=self.end, freq='h'))
        
        # Create one column per event type
        for event in df_holiday['Holiday Type'].unique():
            df_flagged[event] = 0  # initialize as 0

        # Fill flags
        for _, row in df_holiday.iterrows():
            mask = (df_flagged.index >= row['Start Date']) & (df_flagged.index <= row['End Date'])
            df_flagged.loc[mask, row['Holiday Type']] = 1

        # Optional: add a combined flag if needed
        df_flagged['any_holiday'] = (df_flagged.sum(axis=1) > 0).astype(int)
        
        # add column for each of the events
        print(df_events.head())
        # the table shows type of event per row, year in the columns and the date or date range in the cells (e.g., 2022-10-20 or 2024-09-02 to 2024-09-10)
        
        # Transform each table cell to datetime range or data            
        # Iterate over each row in the events table
        # Create a new column for each event type (row) and flag 0
        # iterate over all cells in the row and flag all dates covering or in the range in the flag table
        for _, row in df_events.iterrows():
            event_name = row['Event']
            df_flagged[event_name] = 0
            for year in df_events.columns[1:]:
                date_info = row[year]
                if pd.isna(date_info):
                    continue
                # Check if it's a range
                if 'to' in date_info:
                    start_str, end_str = date_info.split('to')
                    start_date = pd.to_datetime(start_str.strip(), format="%Y_%m_%d")
                    end_date = pd.to_datetime(end_str.strip(), format="%Y_%m_%d")
                else:
                    start_date = pd.to_datetime(date_info.strip(), format="%Y_%m_%d")
                    end_date = start_date
                # Flag the dates in the range
                mask = (df_flagged.index >= start_date) & (df_flagged.index <= end_date)
                df_flagged.loc[mask, event_name] = 1

        
        # concatenate to self.data
        self.data_events = pd.concat([self.data_events, df_flagged], axis=1)
        
                
        # Generate value columns to ste date information
        self.data_calendar = pd.DataFrame(index=self.data_events.index)
        # weekday column
        self.data_calendar['hour'] = self.data_calendar.index.hour
        self.data_calendar['weekday'] = self.data_calendar.index.weekday
        self.data_calendar['month'] = self.data_calendar.index.month
        
        # national holidays
        # Swiss holidays for Zürich
        ch_holidays = holidays.CountryHoliday('CH', prov='ZH')
        # Create a new column: 1 if holiday, 0 otherwise
        #self.data['is_holiday'] = self.data.index.to_series().apply(lambda x: 1 if x in ch_holidays else 0)
        #self.data_events = pd.concat([self.data_events, self.data_calendar], axis=1)
        
        # filter within range
        self.data_events = self.data_events[(self.data_events.index >= self.start) & (self.data_events.index <= self.end)]
        self.data_calendar = self.data_calendar[(self.data_calendar.index >= self.start) & (self.data_calendar.index <= self.end)]

        
    def normalize_min_max(self):
        # normalize per column
        for col in self.data.columns:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
    
    def normalize_standard(self):
        # normalize data_events, data_calendar and data_weather separately
        # normalize per column
        for col in self.data_events.columns:
            # min-max normalization
            min_val = self.data_events[col].min()
            max_val = self.data_events[col].max()
            self.data_events[col] = (self.data_events[col] - min_val) / (max_val - min_val)
        for col in self.data_calendar.columns:
            min_val = self.data_calendar[col].min()
            max_val = self.data_calendar[col].max()
            self.data_calendar[col] = (self.data_calendar[col] - min_val) / (max_val - min_val)
        for col in self.data_weather.columns:
            min_val = self.data_weather[col].min()
            max_val = self.data_weather[col].max()
            self.data_weather[col] = (self.data_weather[col] - min_val) / (max_val - min_val)
        
    def load_weather_data(self):
        """Loads CSV file into a DataFrame."""
        # Load weather data
        
        df_weather = pd.DataFrame()
        # for all csv files in the folder self.data_path containing "ogd"
        data_path_suffix = "weather_zh"
        for file in os.listdir(os.path.join(self.data_path, data_path_suffix)):
            if file.endswith(".csv") and "ogd" in file:
                file_path = os.path.join(self.data_path, os.path.join(data_path_suffix, file))
                print("Loading weather data from file: ", file_path)
                df_temp = pd.read_csv(file_path, low_memory=False, sep=";")
                
                df_temp['reference_timestamp'] = pd.to_datetime(df_temp['reference_timestamp'], format="%d.%m.%Y %H:%M")
                df_temp.set_index('reference_timestamp', inplace=True)
                
                # Delete a few columns: containing tre005
                cols_to_drop = [col for col in df_temp.columns if 'tre005' in col]
                df_temp.drop(columns=cols_to_drop, inplace=True)
                
                # Drop columns with all NaN values
                df_temp.dropna(axis=1, how='all', inplace=True)
                
                # Add station_abbr as suffix to all columns except reference_timestamp
                station_abbr = df_temp['station_abbr'].iloc[2]
                # delete column station_abbr
                df_temp.drop(columns=['station_abbr'], inplace=True)
                df_temp.columns = [f"{col}_{station_abbr}" for col in df_temp.columns]            

                df_weather = pd.merge(df_weather, df_temp, left_index=True, right_index=True, how="outer") if not df_weather.empty else df_temp
                
        self.data_weather = df_weather
        #self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")
        
        # Filter data within the specified date range
        self.data_weather = self.data_weather[(self.data_weather.index >= self.start) & (self.data_weather.index <= self.end)]



class YearlyTrafficProcessor:
    """Handles preprocessing for a single year's traffic data."""
    
    def __init__(self, year, file_path):
        self.year = year
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads CSV file into a DataFrame."""
        self.data = pd.read_csv(self.file_path, low_memory=False)
    
    def clean_data(self):
        """Applies dataset-specific cleaning."""
        self.data.drop(columns=['MSName', 'ZSID', 'ZSName', 'LieferDat', 'Hoehe'], inplace=True)
        self.data.drop(columns=['Achse', 'HNr', 'Knummer', 'Kname'], inplace=True)
        self.data["AnzFahrzeuge"] = pd.to_numeric(self.data["AnzFahrzeuge"], errors='coerce')  # Convert missing values to NaN
        self.data.dropna(inplace=True)  # Remove rows with NaN
        
        # For all duplicates get the average value
        self.data = self.data.groupby(["MSID", "MessungDatZeit"]).agg({
            "AnzFahrzeuge": "mean",
            "Richtung": "first",
            "EKoord": "first",
            "NKoord": "first"
        }).reset_index()

        
    def reshape_data(self):
        """Pivot data so each timestamp becomes a column."""
        self.data = self.data.pivot(index="MSID", columns="MessungDatZeit", values="AnzFahrzeuge").reset_index()
    
    def get_available_sensors(self):
        """Returns a list of sensors available in the dataset."""
        return self.data["MSID"].unique()

    def get_processed_data(self):
        """Returns the processed DataFrame."""
        return self.data
    
    def get_coordinates(self):
        """Returns the coordinates of the sensor and the direction."""
        
        return self.groupby(by="MSID")[["NKoord", "EKoord", "Richtung"]].first()


class TrafficDataManager:
    """Manages multiple years of traffic data and merges them."""
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.all_data = {}  # Stores all processed yearly DataFrames
        self.yearly_available_sensors = []
        # tbd ##########################################################
        # self.coordinates =
    
    def process_all_years(self):
        """Processes all CSV files in the directory, assuming each file is a different year."""
        print("\nLoading all years...")
        for file in os.listdir(self.data_directory):
            if file.endswith(".csv"):
                # Extract year from filename (assuming "traffic_dav_miv_YYYY.csv")
                year = int(file.split("_")[-1].split(".")[0])
                print(f"Loading year {year}...")
                file_path = os.path.join(self.data_directory, file)
                
                processor = YearlyTrafficProcessor(year, file_path)
                processor.load_data()
                processor.clean_data()
                processor.reshape_data()
                
                # Aggregating all data
                processed_df = processor.get_processed_data()
                self.all_data[year] = processed_df
                self.yearly_available_sensors.append(processor.get_available_sensors())

    def merge_data(self):
        """Combines all yearly DataFrames into a single one."""
        print("\n Merging all years...")
        print("Keys in self.all_data:", self.all_data.keys())  # Check available years
        for key, df in self.all_data.items():
            print(key, df.shape)
        # Intialize the final DataFrame with index as MSID
        final_df = pd.DataFrame(index=self.all_data[next(iter(self.all_data))]["MSID"])
        # Iterate over all years and merge the DataFrames
        for key, df in self.all_data.items():
            print(key)
            df.set_index("MSID", inplace=False)
            final_df = pd.merge(final_df, df, on="MSID", how="outer")
            print(final_df.shape)
        
        # Set column msid to index
        final_df.set_index("MSID", inplace=True)
        #final_df["MSID"] = final_df.index
        # Columns to datetime and sort
        final_df.columns = pd.to_datetime(final_df.columns)
        final_df = final_df.reindex(sorted(final_df.columns), axis=1)
        
        print(final_df)
        return final_df
    
    def yearly_availability(self):
        """Return table with available sensors per year."""
        # Get unique values over all years
        all_sensors = set().union(*self.yearly_available_sensors)
        print(len(all_sensors))
        # Create a DataFrame with sensors as rows and years as columns
        #df = pd.DataFrame(index=all_sensors, columns=[year for year in range(2000, 2022)])
        
        
        return {year: len(df) for year, df in self.all_data.items()}


class YearlyActiveModeProcessor:
    """Handles preprocessing for a single year's active mode data."""
    
    def __init__(self, year, file_path):
        self.year = year
        self.file_path = file_path
        self.data = None
        self.bike_data = None
        self.pedestrian_data = None
        
    def load_data(self):
        """Loads CSV file into a DataFrame."""
        self.data = pd.read_csv(self.file_path, low_memory=False)
        
    def drop_empty_rows(self):
        """Drops rows with missing values."""
        # Get columns with "IN" and "OUT" ending
        columns_to_check = [col for col in self.data.columns if col.endswith("IN") or col.endswith("OUT")]
        self.data[columns_to_check] = self.data[columns_to_check].apply(pd.to_numeric, errors='coerce')
        self.data.dropna(subset=columns_to_check, inplace=True)
        
    def clean_data(self):
        """Applies dataset-specific cleaning."""
        self.data.drop(columns=['OST', 'NORD'], inplace=True)

        # For all duplicates get the average value
        self.data = self.data.groupby(["FK_STANDORT", "DATUM"]).agg({
            "VELO_IN": "mean",
            "VELO_OUT": "mean",
            "FUSS_IN": "mean",
            "FUSS_OUT": "mean"
        }).reset_index()

        self.bike_data = self.data[["FK_STANDORT", "DATUM", "VELO_IN", "VELO_OUT"]]
        self.pedestrian_data = self.data[["FK_STANDORT", "DATUM", "FUSS_IN", "FUSS_OUT"]]
        
    def reshape_data(self):
        """Pivot data so each timestamp becomes a column."""
        self.bike_data = self.bike_data.melt(id_vars=["FK_STANDORT", "DATUM"], value_vars=["VELO_IN", "VELO_OUT"], var_name="DIRECTION", value_name="COUNT")
        self.pedestrian_data = self.pedestrian_data.melt(id_vars=["FK_STANDORT", "DATUM"], value_vars=["FUSS_IN", "FUSS_OUT"], var_name="DIRECTION", value_name="COUNT")

        self.bike_data["DIRECTION"] = self.bike_data["DIRECTION"].str.replace("VELO_", "", regex=False).str.lower()
        self.pedestrian_data["DIRECTION"] = self.pedestrian_data["DIRECTION"].str.replace("FUSS_", "", regex=False).str.lower()

        self.bike_data = self.bike_data.pivot(index=["FK_STANDORT", "DIRECTION"], columns="DATUM", values="COUNT")
        self.pedestrian_data = self.pedestrian_data.pivot(index=["FK_STANDORT", "DIRECTION"], columns="DATUM", values="COUNT")
            
        # delete rows with only missing values
        self.bike_data.dropna(how='all', inplace=True)
        self.pedestrian_data.dropna(how='all', inplace=True)

        # Pivot table
        
    def get_available_sensors(self):
        """Returns a list of sensors available in the dataset."""
        return self.data["MSID"].unique()

    def get_processed_data(self):
        """Returns the processed DataFrame."""
        return self.bike_data, self.pedestrian_data
    
    def get_coordinates(self):
        """Returns the coordinates of the sensor and the direction."""
        # Melt to long format
        
        return self.groupby(by="MSID")[["NKoord", "EKoord", "Richtung"]].first()


class ActiveModeDataLoader:
    """Processes activemode data with its specific preprocessing steps."""
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.all_bike_data = {}  # Stores all processed yearly DataFrames
        self.all_pedestrian_data = {}  # Stores all processed yearly DataFrames
        self.yearly_available_sensors = []
        # tbd ##########################################################
        # self.coordinates =
    
    def process_all_years(self):
        """Processes all CSV files in the directory, assuming each file is a different year."""
        print("\nLoading all years...")
        for file in os.listdir(self.data_directory):
            if file.endswith(".csv"):
                # Extract year from filename (assuming "traffic_dav_miv_YYYY.csv")
                year = int(file.split("_")[0])
                print(f"Loading year {year}...")
                file_path = os.path.join(self.data_directory, file)
                
                processor = YearlyActiveModeProcessor(year, file_path)
                processor.load_data()
                processor.clean_data()
                processor.reshape_data()
                
                # Aggregating all data
                processed_bike_df, processed_pedestrian_data = processor.get_processed_data()
                
                self.all_bike_data[year] = processed_bike_df
                self.all_pedestrian_data[year] = processed_pedestrian_data
                #self.yearly_available_sensors.append(processor.get_available_sensors())

    def merge_data(self):
        """Combines all yearly DataFrames into a single one."""
        print("\n Merging all years...")
        all_data = []
        for mode in [self.all_bike_data, self.all_pedestrian_data]:
            print("Keys in self.all_data:", mode.keys())  # Check available years
            # Get the first item of the dictionary
            final_df = pd.DataFrame()
            # Iterate over all years and merge the DataFrames
            for key, df in mode.items():
                print(key)
                if final_df.empty:
                    final_df = df
                else:
                    #df.set_index(["FK_STANDORT", "DIRECTION"], inplace=False)
                    final_df = pd.merge(final_df, df, left_index=True, right_index=True, how="outer")
                print(final_df.shape)

                
            
            # Sort the columns in ascending order
            final_df = final_df.reindex(sorted(final_df.columns), axis=1)
            # Concvert columns to datetimeIndex
            final_df.columns = pd.to_datetime(final_df.columns)
            
            # Group columns from 15min intervals to hour timestamp
            #final_df = final_df.groupby(by=pd.to_datetime(final_df.columns).floor("h"), axis=1).sum(min_count=1)
            final_df = (
                final_df.T
                .groupby(pd.to_datetime(final_df.columns).floor("h"))
                .sum(min_count=1)
                .T
            )

            # set index to columns
            #final_df["ID"] = final_df.index.get_level_values(0)
            #final_df["DIRECTION"] = final_df.index.get_level_values(1)
            all_data.append(final_df)
        return all_data


class YearlyTransitProcessor:
    """Handles preprocessing for a single year's active mode data."""
    
    def __init__(self, year, file_path):
        self.year = year
        self.file_path = file_path
        self.data = None
        self.dates = None
        
    def load_data(self):
        """Loads CSV file into a DataFrame."""
        file_path_data = os.path.join(self.data_directory, folder, "RESISENDE.csv")
        self.data = pd.read_csv(self.file_path, sep=";", low_memory=False)
        file_path_dates = os.path.join(self.data_directory, folder, "TAGTYP.csv")
        self.dates = pd.read_csv(file_path_dates, sep=";", low_memory=False)
    
        
    def get_dates(self):
        """Extracts the dates from the DataFrame."""
        # Only keep row containing "Jahresfahrplan" or "Ferienfahrplan" incolumn "Bemerkung"
        self.dates = self.dates[self.dates["Bemerkung"].str.contains("Jahresfahrplan|Ferienfahrplan")]
        self.dates = self.data.columns[1:]
        
    def drop_empty_rows(self):
        """Drops rows with missing values."""
        # Get columns with "IN" and "OUT" ending
        columns_to_check = [col for col in self.data.columns if col.endswith("IN") or col.endswith("OUT")]
        self.data[columns_to_check] = self.data[columns_to_check].apply(pd.to_numeric, errors='coerce')
        self.data.dropna(subset=columns_to_check, inplace=True)
        
    def clean_data(self):
        """Applies dataset-specific cleaning."""
        # Keep needed data
        self.data = self.data[["Tagtyp_Id", "FZ_AB", "Besetzung", "ID_Abschnitt", "Haltestellen_Id", "Nach_Hst_Id"]]
        
        # Access the Tagtyp_Id and combine with "FZ_AB" for time
        # "Besetzung" is the number of passengers
        # Raum: ID_Abschnitt, "Haltestellen_Id", "Nach_Hst_Id"
        
        
        self.data.drop(columns=['OST', 'NORD'], inplace=True)

        # For all duplicates get the average value
        self.data = self.data.groupby(["FK_STANDORT", "DATUM"]).agg({
            "VELO_IN": "mean",
            "VELO_OUT": "mean",
            "FUSS_IN": "mean",
            "FUSS_OUT": "mean"
        }).reset_index()
        
        self.bike_data = self.data[["FK_STANDORT", "DATUM", "VELO_IN", "VELO_OUT"]]
        self.pedestrian_data = self.data[["FK_STANDORT", "DATUM", "FUSS_IN", "FUSS_OUT"]]

        
    def reshape_data(self):
        """Pivot data so each timestamp becomes a column."""
        self.bike_data = self.bike_data.melt(id_vars=["FK_STANDORT", "DATUM"], value_vars=["VELO_IN", "VELO_OUT"], var_name="DIRECTION", value_name="COUNT")
        self.pedestrian_data = self.pedestrian_data.melt(id_vars=["FK_STANDORT", "DATUM"], value_vars=["FUSS_IN", "FUSS_OUT"], var_name="DIRECTION", value_name="COUNT")
        print("\n self.pedesrian_data: ", self.pedestrian_data.shape)
        
        self.bike_data["DIRECTION"] = self.bike_data["DIRECTION"].str.replace("VELO_", "", regex=False).str.lower()
        self.pedestrian_data["DIRECTION"] = self.pedestrian_data["DIRECTION"].str.replace("FUSS_", "", regex=False).str.lower()
        print("\n self.pedesrian_data: ", self.pedestrian_data.shape)

        self.bike_data = self.bike_data.pivot(index=["FK_STANDORT", "DIRECTION"], columns="DATUM", values="COUNT")
        self.pedestrian_data = self.pedestrian_data.pivot(index=["FK_STANDORT", "DIRECTION"], columns="DATUM", values="COUNT")
        print("Data shape after reshaping:", self.bike_data.shape, self.pedestrian_data.shape)
        print("\n self.pedesrian_data: ", self.pedestrian_data.shape)
        
        self.bike_data.dropna(inplace=True)
        self.pedestrian_data.dropna(inplace=True)
        # Pivot table
        
    def get_available_sensors(self):
        """Returns a list of sensors available in the dataset."""
        return self.data["MSID"].unique()

    def get_processed_data(self):
        """Returns the processed DataFrame."""
        return self.bike_data, self.pedestrian_data
    
    def get_coordinates(self):
        """Returns the coordinates of the sensor and the direction."""
        # Melt to long format
        
        return self.groupby(by="MSID")[["NKoord", "EKoord", "Richtung"]].first()
    
class TransitDataLoader:
    """Processes activemode data with its specific preprocessing steps."""
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.all_bike_data = {}  # Stores all processed yearly DataFrames
        self.all_pedestrian_data = {}  # Stores all processed yearly DataFrames
        self.yearly_available_sensors = []
        # tbd ##########################################################
        # self.coordinates =
    
    def process_all_years(self):
        """Processes all CSV files in the directory, assuming each file is a different year."""
        print("\nLoading all years...")
        for folder in os.listdir(self.data_directory):
            year = int(folder.split("_")[1])
            print(f"Loading year {year}...")
            # Open access the files in the folder
            file_path = os.path.join(self.data_directory, folder)
            print(file_path)
            
            processor = YearlyActiveModeProcessor(year, file_path)
            processor.load_data()
            processor.clean_data()
            processor.reshape_data()
            
            # Aggregating all data
            processed_bike_df, processed_pedestrian_data = processor.get_processed_data()
            
            self.all_bike_data[year] = processed_bike_df
            self.all_pedestrian_data[year] = processed_pedestrian_data
            #self.yearly_available_sensors.append(processor.get_available_sensors())

    def merge_data(self):
        """Combines all yearly DataFrames into a single one."""
        print("\n Merging all years...")
        all_data = []
        for mode in [self.all_bike_data, self.all_pedestrian_data]:
            print("Keys in self.all_data:", mode.keys())  # Check available years
            # Get the first item of the dictionary
            final_df = pd.DataFrame()
            # Iterate over all years and merge the DataFrames
            for key, df in mode.items():
                print(key)
                if final_df.empty:
                    final_df = df
                else:
                    #df.set_index(["FK_STANDORT", "DIRECTION"], inplace=False)
                    final_df = pd.merge(final_df, df, left_index=True, right_index=True, how="outer")
                print(final_df.shape)

            # set index to columns
            final_df["ID"] = final_df.index.get_level_values(0)
            final_df["DIRECTION"] = final_df.index.get_level_values(1)
            all_data.append(final_df)
        return all_data

################################################################################################################################
# Data Loader for NYC data

# Bike data
class NYCBikeDataLoader(Dataset):
    """Loads and processes NYC bike data for training and testing."""
    
    def __init__(self, data_path, range_start, range_end, mode="bike"):
        self.data_path = data_path
        self.mode = mode
        self.range_start = datetime.strptime(f"{range_start}-01-00-00", "%Y-%m-%d-%H-%M")
        self.range_end = datetime.strptime(get_last_day_of_month(range_end), "%Y-%m-%d-%H-%M")
        
        
        # Iterate over all files in the folder
        # Get all values within the time range
        # Store it as one file
        # Load the data
        
        # Prepare the data
        #self.handle_missing_values()
        
        # Split the series
        #self.split_series_to_array_single_sensor(self.data, input_size=5*24, output_size=21)

class NYCBikeYearlyProcessor:
    """Processes yearly NYC bike data."""
    
    def __init__(self, year, file_path):
        self.year = year
        self.file_path = file_path
        self.data = None
        
        self.load_data()
        self.clean_data()
        self.reshape_data()
        
    def load_data(self):
        """Loads CSV file into a DataFrame."""
        self.data = pd.read_csv(self.file_path, low_memory=False)
        
    def transform_trip_to_demand():
        """
        1) get unique stations dict: id, name
        2) Table with ID (in/out), all time steps
        3) Full the table with counts
        """
        
    def clean_data(self):
        """Applies dataset-specific cleaning."""
        # Drop unnecessary columns
        self.data.drop(columns=['MSName', 'ZSID', 'ZSName', 'LieferDat', 'Hoehe'], inplace=True)
        self.data.drop(columns=['Achse', 'HNr', 'Knummer', 'Kname'], inplace=True)
        
        # Convert AnzFahrzeuge to numeric and drop rows with NaN
        self.data["AnzFahrzeuge"] = pd.to_numeric(self.data["AnzFahrzeuge"], errors='coerce')
        self.data.dropna(inplace=True)  # Remove rows with NaN
        
    def reshape_data(self):
        """Pivot data so each timestamp becomes a column."""
        self.data = self.data.pivot(index="ID", columns="MessungDatZeit", values="AnzFahrzeuge").reset_index()




################################################################################################################################


class SplitTimeSeriesSingleMode:
    
    def __init__(self, data, range_start, range_end, mode, transpose=False):
        self.data = data
        self.mode = mode
        self.data_all = data        
        # Get the range in years and convert it to YYYY-MM-DD-HH-mm
        self.range_start = datetime.strptime(f"{range_start}-01-00-00", "%Y-%m-%d-%H-%M")
        self.range_end = datetime.strptime(get_last_day_of_month(range_end), "%Y-%m-%d-%H-%M")
        self.train_X_nw = None
        self.train_Y_mw = None
        self.test_X_nw = None
        self.test_Y_nw = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        
        self.X_train_mask = None
        self.Y_train_mask = None
        self.X_test_mask = None
        self.Y_test_mask = None
        
        """
        if mode == "bike" or mode == "ped":
            # Set columns ID, Direction as index
            self.data.set_index(["ID", "DIRECTION"], inplace=True)
        elif mode == "road":
            # print last 5 elements of the columns
            self.data.set_index(["MSID"], inplace=True)
            print("Columns: ", self.data.columns[-5:])
            # Print first 10 columns
            print("Columns: ", self.data.columns[:10])
        """

        if transpose:
            print("Transposing data...")
            self.data = self.data.T
        
        # Sort the columns of the pd df in ascending order
        self.data.columns = pd.to_datetime(self.data.columns)
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)

        print(f"\n \nPreparing test and training data for {mode}")
        print(f"Shape of the data: {self.data.shape} before splitting")
        
        
    def create_rolling_windows(self, df, n_obs, m_steps, stride=1):
        stride = m_steps # non overlapping windows
        
        df = df.sort_index(axis=1)  # Ensure chronological order
        df = df.apply(pd.to_numeric, errors='coerce')
        total_timesteps = df.shape[1]
        
        # Calculate valid sample count
        num_samples = (total_timesteps - n_obs - m_steps) // stride + 1
        
        # Preallocate arrays
        X = np.full((num_samples, n_obs, df.shape[0]), np.nan)
        Y = np.full((num_samples, m_steps, df.shape[0]), np.nan)
        date = np.full((num_samples, 4), np.nan)
        
        for i in range(num_samples):
            start = i * stride
            X[i] = df.iloc[:, start:start+n_obs].values.T
            Y[i] = df.iloc[:, start+n_obs:start+n_obs+m_steps].values.T
            # Get years of start, month, day, hour
            date[i, 0] = df.columns[start].year
            date[i, 1] = df.columns[start].month
            date[i, 2] = df.columns[start].day
            date[i, 3] = df.columns[start].hour
        
        # Create masks
        X_mask = ~np.isnan(X)
        Y_mask = ~np.isnan(Y)
        
        # Impute NaNs (optional)
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)
        
        return X, X_mask, Y, Y_mask, date
        
    
    def split_series_to_array_single_sensor(self, df, input_size, output_size):
        """
        Splits time series into rolling windows per sensor, with padding to ensure uniform shape.
        
        - Handles each sensor separately.
        - Stores missing sequences as NaNs to maintain alignment.
        
        Returns:
            X_padded (np.array): (max_samples, num_sensors, input_size)
            Y_padded (np.array): (max_samples, num_sensors, output_size)
        """
        
        sensor_data = {}  # Store X, Y per sensor
        valid_sensor = []
        
        for sensor_idx in range(df.shape[0]):  # Iterate sensors (rows)
            X, Y = [], []
            sensor_series = df.iloc[sensor_idx, :]  # Get time series for sensor
            
            for start in range(0, df.shape[1] - input_size - output_size + 1, output_size):
                X_window = sensor_series.iloc[start:start+input_size].values
                Y_window = sensor_series.iloc[start+input_size:start+input_size+output_size].values

                # Skip windows with NaNs
                if np.isnan(X_window).any() or np.isnan(Y_window).any():
                    continue
                
                X.append(X_window)
                Y.append(Y_window)

            # Store valid data for this sensor
            
            # Keep index with
            if X and Y:
                valid_sensor.append(sensor_idx)
            
            sensor_data[sensor_idx] = (np.array(X), np.array(Y))

        # Determine max sample count for padding
        max_samples = max(len(X) for X, _ in sensor_data.values())
        
        # Initialize padded arrays with NaN
        num_sensors = len(sensor_data)
        X_padded = np.full((max_samples, num_sensors, input_size), np.nan)
        Y_padded = np.full((max_samples, num_sensors, output_size), np.nan)

        #for sensor_idx, (X_array, Y_array) in sensor_data.items():
        # Fill in sensor data
        for sensor_idx, (X_array, Y_array) in enumerate(sensor_data.values()):
            try:
                X_padded[:X_array.shape[0], sensor_idx, :] = X_array
                Y_padded[:Y_array.shape[0], sensor_idx, :] = Y_array
            except:
                continue
                print(sensor_idx, X_array.shape, Y_array.shape)
                print(X_padded.shape, Y_padded.shape)
        return X_padded, Y_padded, valid_sensor
    
        
    def handle_missing_values(self):
        """Applies dataset-specific cleaning."""
        # Drop columns that are no in the range
        print(f" Considered time interval {self.range_start} to {self.range_end}")
        # Find nearest existing timestamps
        #range_start = self.data.columns[self.data.columns.get_indexer([self.range_start], method='nearest')[0]]
        #range_end = self.data.columns[self.data.columns.get_indexer([self.range_end], method='nearest')[0]]

        # make sure the columns are in datetime format
        self.data = self.data.loc[:, self.range_start:self.range_end].copy()
        
        # Interpolate values if less than 5 missing values in a row
        self.data.interpolate(method='linear', limit=5, inplace=True)
        # Drop rows if only NaN values are present
        self.data.dropna(axis=0, how='all', inplace=True)
        
        # Get the ratio of missing values for each row
        missing_values = self.data.isna().sum(axis=1) / self.data.shape[1]
        print("Missing data per sensor ", missing_values.value_counts())
    
    def store_data(self, suffix=""):
        
        print("\n Storing train and test tensor...")
        print("Shape of training data: ", self.train_X.shape, self.train_Y.shape)
        print("Shape of testing data: ", self.test_X.shape, self.test_Y.shape)
        if suffix == "":
            folder = "data/train_test"
        else:
            folder = f"data/train_test/{suffix}"
        mode = self.mode
        np.save(f"{folder}/{mode}_train_X.npy", self.train_X)
        np.save(f"{folder}/{mode}_train_Y.npy", self.train_Y)
        np.save(f"{folder}/{mode}_test_X.npy", self.test_X)
        np.save(f"{folder}/{mode}_test_Y.npy", self.test_Y)
        
        # if model not either weather or calendar
        if mode != "weather" and mode != "calendar" and mode!= "weather_reduced":
            np.save(f"{folder}/{mode}_train_X_mask.npy", self.X_train_mask)
            np.save(f"{folder}/{mode}_train_Y_mask.npy", self.Y_train_mask)
            np.save(f"{folder}/{mode}_test_X_mask.npy", self.X_test_mask)
            np.save(f"{folder}/{mode}_test_Y_mask.npy", self.Y_test_mask)
        #np.save(f"{folder}/{mode}_date_train.npy", self.date_train)
        #np.save(f"{folder}/{mode}_date_test.npy", self.date_test)

    
    def remove_sensors_with_no_samples(self):
        """
        Removes sensors that have no valid samples in either X or Y.
        
        Args:
            X (np.array): Shape (max_samples, num_sensors, input_size)
            Y (np.array): Shape (max_samples, num_sensors, output_size)
            
        Returns:
            X_filtered, Y_filtered: Arrays with sensors having valid samples.
        """
        # Check for sensors with no valid samples in X or Y
        valid_sensors = []
        num_sensors = self.train_X.shape[1]

        for sensor_idx in range(num_sensors):
            has_valid_X_train = np.isfinite(self.train_X[:, sensor_idx, :]).any()
            has_valid_Y_train = np.isfinite(self.train_Y[:, sensor_idx, :]).any()
            has_valid_X_test = np.isfinite(self.test_X[:, sensor_idx, :]).any()
            has_valid_Y_test = np.isfinite(self.test_Y[:, sensor_idx, :]).any()

            if has_valid_X_train and has_valid_Y_train and has_valid_X_test and has_valid_Y_test:
                valid_sensors.append(sensor_idx)

        # Convert to NumPy array
        valid_sensors = np.array(valid_sensors)

        # Filter X and Y
        self.train_X = self.train_X[:, valid_sensors, :]
        self.train_Y = self.train_Y[:, valid_sensors, :]
        self.test_X = self.test_X[:, valid_sensors, :]
        self.test_Y = self.test_Y[:, valid_sensors, :]
        
    
    def split_training_testing(self, split_date=False, ratio=0.85, input_size=5*24, output_size=21):
        """Splits the data into training and testing sets."""
        #self.handle_missing_values()
        
        if split_date:
            split_date = datetime.strptime((f"{split_date}-01-00"), "%Y-%m-%d-%H-%M")
            df_train = self.data.loc[:, self.data.columns < split_date]
            df_test = self.data.loc[:, self.data.columns >= split_date]
            
            self.train_X, self.X_train_mask, self.train_Y, self.Y_train_mask, self.date_train = self.create_rolling_windows(df_train, input_size, output_size)
            self.test_X, self.X_test_mask, self.test_Y, self.Y_test_mask, self.date_test = self.create_rolling_windows(df_test, input_size, output_size)
            #self.train_X, self.train_Y, valid_sensor_train = self.split_series_to_array_single_sensor(df_train, input_size, output_size)
            #self.test_X, self.test_Y, valid_sensor_test = self.split_series_to_array_single_sensor(df_test, input_size, output_size)
            
            #self.remove_sensors_with_no_samples() # Not needed as it has been done before
        else:
            df_temp = self.data.copy()
            X, X_mask, Y, Y_mask, date = self.create_rolling_windows(df_temp, input_size, output_size)
            #X, Y, _ = self.split_series_to_array_single_sensor(df_temp, input_size, output_size)
            #X_nw, Y_nw = self.split_series_to_array_nw(df_temp, input_size, output_size)
            
            # Split the data into training and testing
            split = int(ratio * X.shape[0])
            
            self.train_X, self.train_Y = X[:split], Y[:split]
            self.X_train_mask, self.Y_train_mask = X_mask[:split], Y_mask[:split]
            self.test_X, self.test_Y = X[split:], Y[split:]
            self.X_test_mask, self.Y_test_mask = X_mask[split:], Y_mask[split:]
            self.date_train, self.date_test = date[:split], date[split:]
            
            print(f"Shape of training data: {self.train_X.shape}, {self.train_Y.shape}")
            print(f"Shape of testing data: {self.test_X.shape}, {self.test_Y.shape}")
            # print date of split
            split_date = self.data.columns[split + input_size + output_size - 1]  # Last timestamp in the training set
            print(f"Date of split: {split_date}")
            
            #self.train_X_nw, self.train_Y_nw = X_nw[:split], Y_nw[:split]
            #self.test_X_nw, self.test_Y_nw = X_nw[split:], Y_nw[split:]
            
        # Perform a rolling window with X-window size of m and Y-window size of n
    
    # Date range
    # missing values
    # Split training/testing: Split date / ratio
    # Delete nodes with no data for time series
    
    # create set of n observation steps + m pred steps
    
    # return: training, testing network
    # return: training, testing sensors
    
class TimeSeriesDataset(Dataset):
    def __init__(self, mode, path="data/train_test", train=True, exogen_var=False, normalization_type=None, norm_params=None, exo_reduced=True):
        """
        X: (p, s, n) -> input sequences
        y: (p, s, m) -> output sequences
        mask_x: (p, s, n) -> binary mask for X
        mask_y: (p, s, m) -> binary mask for y
        """
        if train==True:
            suffix = "train"
        else:
            suffix = "test"
        
        # Load the data
        try:
            self.X_mask = torch.from_numpy(np.load(f"{path}/{mode}_{suffix}_X_mask.npy")).float()
            self.Y_mask = torch.from_numpy(np.load(f"{path}/{mode}_{suffix}_Y_mask.npy")).float()
        except:
            self.X_mask = None
            self.Y_mask = None
        self.X_raw = torch.from_numpy(np.load(f"{path}/{mode}_{suffix}_X.npy")).float()
        self.Y_raw = torch.from_numpy(np.load(f"{path}/{mode}_{suffix}_Y.npy")).float()
        self.norm_params = norm_params if norm_params is not None else {}
        self.normalization_type = normalization_type
        self.normalized = False
        
        self.exogen_var = exogen_var
        if self.exogen_var:
            self.calendar_X = torch.from_numpy(np.load(f"{path}/calendar_{suffix}_X.npy")).float()
            self.calendar_Y = torch.from_numpy(np.load(f"{path}/calendar_{suffix}_Y.npy")).float()
            self.events_X = torch.from_numpy(np.load(f"{path}/events_{suffix}_X.npy")).float()
            self.events_Y = torch.from_numpy(np.load(f"{path}/events_{suffix}_Y.npy")).float()
        if self.exogen_var and exo_reduced==False:
            self.weather_X = torch.from_numpy(np.load(f"{path}/weather_{suffix}_X.npy")).float()
            self.weather_Y = torch.from_numpy(np.load(f"{path}/weather_{suffix}_Y.npy")).float()
        if self.exogen_var and exo_reduced==True:
            self.weather_X = torch.from_numpy(np.load(f"{path}/weather_reduced_{suffix}_X.npy")).float()
            self.weather_Y = torch.from_numpy(np.load(f"{path}/weather_reduced_{suffix}_Y.npy")).float()
            
        if train==True:
            self.n_samples_train = self.X_raw.shape[0]
            self.n_samples_test = None
        else:
            self.n_samples_train = None
            self.n_samples_test = self.X_raw.shape[0]
        
        self.num_sensors = self.X_raw.shape[2]
        self.input_dim = self.X_raw.shape[1]
        self.output_dim = self.Y_raw.shape[1]
        
        # Normalization
        self.X = self.X_raw
        self.Y = self.Y_raw

        if self.normalization_type is None:
            print("No normalization applied, using raw data.")
        elif self.normalization_type == "rowwise_minmax":
            if train:
                self.normalize_rowwise()  # Compute and apply
            else:
                self.apply_rowwise_minmax_normalization()  # Only apply
        elif self.normalization_type == "minmax":
            if train:
                self.normalize_minmax()  # Compute and apply
            else:
                self.apply_minmax_normalization()  # Only apply
        elif self.normalization_type == "both":
            if train:
                self.normalize_both()  # Compute and apply
            else:
                self.apply_both_normalization()

    def normalize_both(self):
        # minmax normalization for entire ds and rowwise
        self.normalize_minmax()
        self.normalize_rowwise(both=True)
    
    def apply_both_normalization(self):
        # Apply both normalization methods using existing params
        self.apply_minmax_normalization()
        self.apply_rowwise_minmax_normalization(both=True)
        
    def normalize_minmax(self):
        # Compute and store params
        x_min = self.X.min()
        x_max = self.X.max()
        self.norm_params = {"X_min": x_min, "X_max": x_max}
        self.apply_minmax_normalization()

    def apply_minmax_normalization(self):
        # Use existing params to normalize
        X_min = self.norm_params["X_min"]
        X_max = self.norm_params["X_max"]
        self.X = (self.X - X_min) / (X_max - X_min).clamp(min=1e-8)
        self.Y = (self.Y - X_min) / (X_max - X_min).clamp(min=1e-8)
    
    def normalize_rowwise(self, both=False):
        self.norm_params['X_min'] = torch.amin(self.X, dim=(0, 1)).view(1, 1, -1)
        self.norm_params['X_max'] = torch.amax(self.X, dim=(0, 1)).view(1, 1, -1)
        self.norm_params['Y_min'] = torch.amin(self.Y, dim=(0, 1)).view(1, 1, -1)
        self.norm_params['Y_max'] = torch.amax(self.Y, dim=(0, 1)).view(1, 1, -1)
        self.apply_rowwise_minmax_normalization(both)

    def apply_rowwise_minmax_normalization(self, both=False):
        X_min = self.norm_params['X_min']
        X_max = self.norm_params['X_max']
        Y_min = self.norm_params['Y_min']
        Y_max = self.norm_params['Y_max']
        if both:
            self.X_rowwise = (self.X - X_min) / (X_max - X_min).clamp(min=1e-8)
            self.Y_rowwise = (self.Y - Y_min) / (Y_max - Y_min).clamp(min=1e-8)
        else:
            self.X = (self.X - X_min) / (X_max - X_min).clamp(min=1e-8)
            self.Y = (self.Y - Y_min) / (Y_max - Y_min).clamp(min=1e-8)
        
    def denormalize(self):
        if self.normalization_type is None:
            print("No normalization applied, skipping denormalization.")
            return
        if self.normalization_type == "rowwise_minmax":
            self.denormalize_rowwise()
            print("Denormalizing data with rowwise minmax normalization...")
        if self.normalization_type == "minmax":
            print("Denormalizing data with minmax normalization...")
            self.denormalize_minmax()
        if self.normalization_type == "both":
            self.denormalize_rowwise()
            print("Denormalizing data with rowwise normalization...")
    
    def add_input_feature_dim(self):
        self.X = self.X.unsqueeze(-1)
        self.Y = self.Y.unsqueeze(-1)
        self.X_raw = self.X_raw.unsqueeze(-1)
        self.Y_raw = self.Y_raw.unsqueeze(-1)
        self.X_mask = self.X_mask.unsqueeze(-1)
        self.Y_mask = self.Y_mask.unsqueeze(-1)
        
    
    def check_normalized(self):
        if self.norm_params:
            self.normalized = True
            
    def denormalize_minmax(self):
        """Revert normalized data to original form"""
        if self.norm_params:
            print("Denormalizing data...")
            x_min = self.norm_params['X_min']
            x_max = self.norm_params['X_max']
            
            self.X = self.X * (x_max - x_min) + x_min
            self.Y = self.Y * (x_max - x_min) + x_min
    
    def denormalize_rowwise(self):
        """Revert normalized data to original form"""
        if self.norm_params:
            print("Denormalizing data...")
            print(self.X.shape)
            print(self.norm_params['X_max'].shape, self.norm_params['X_min'].shape)
            self.X = self.X * self.norm_params['X_max'].unsqueeze(1) + self.norm_params['X_min'].unsqueeze(1)
            self.Y = self.Y * self.norm_params['X_max'].unsqueeze(1) + self.norm_params['X_min'].unsqueeze(1)
            
            # if first dimension is one, remove the dim
            if self.X.shape[0] == 1:
                self.X = self.X.squeeze(0)
            if self.Y.shape[0] == 1:
                self.Y = self.Y.squeeze(0)
        
    def get_single_sensor_data(self, sensor_idx):
        """
        Returns the data for a single sensor.
        The data are oof shape (samples, observation_len, sensors)
        
        Args:
            sensor_idx (int): Index of the sensor to retrieve.
        
        Returns:
            Tuple: (X, Y, X_mask, Y_mask) for the specified sensor.
        """
        
        temp_X = self.X[:, :, sensor_idx].unsqueeze(-1)
        temp_Y = self.Y[:, :, sensor_idx].unsqueeze(-1)
        temp_X_mask = self.X_mask[:, :, sensor_idx].unsqueeze(-1)
        temp_Y_mask = self.Y_mask[:, :, sensor_idx].unsqueeze(-1)
        
        return SensorDataset(temp_X, temp_Y, temp_X_mask, temp_Y_mask)
    
    
    def get_list_sensor_data(self, sensor_indices):
        temp_X = self.X[:, :, sensor_indices]
        temp_Y = self.Y[:, :, sensor_indices]
        temp_X_mask = self.X_mask[:, :, sensor_indices]
        temp_Y_mask = self.Y_mask[:, :, sensor_indices]
        
        return SensorDataset(temp_X, temp_Y, temp_X_mask, temp_Y_mask)

    
    def transform_to_tensor(self):
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        self.X_mask = torch.tensor(self.X_mask, dtype=torch.float32)
        self.Y_mask = torch.tensor(self.Y_mask, dtype=torch.float32)
        
        print("Data transformed to torch tensors")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        if self.X_mask is None or self.Y_mask is None:
            return self.X[idx], self.Y[idx] if not self.exogen_var else (self.X[idx], self.Y[idx], self.exo_var_x[idx], self.exo_var_y[idx])
        try:
            return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx], self.X_rowwise[idx], self.Y_rowwise[idx] if not self.exogen_var else (self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx], self.X_rowwise[idx], self.Y_rowwise[idx], self.exo_var_x[idx], self.exo_var_y[idx])
        except:
            return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx] if not self.exogen_var else (self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx], self.exo_var_x[idx], self.exo_var_y[idx])
        """
    
    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.Y[idx],

            "x_mask": None if self.X_mask is None else self.X_mask[idx],
            "y_mask": None if self.Y_mask is None else self.Y_mask[idx],

            "x_rowwise": None if not hasattr(self, "X_rowwise") else self.X_rowwise[idx],
            "y_rowwise": None if not hasattr(self, "Y_rowwise") else self.Y_rowwise[idx],
            
            "x_weather": None if not self.exogen_var else self.weather_X[idx],
            "y_weather": None if not self.exogen_var else self.weather_Y[idx],
            "x_calendar": None if not self.exogen_var else self.calendar_X[idx],
            "y_calendar": None if not self.exogen_var else self.calendar_Y[idx],
            "x_events": None if not self.exogen_var else self.events_X[idx],
            "y_events": None if not self.exogen_var else self.events_Y[idx],
        }

    
        
    def shape(self):
        print("Not implemented yet")


class SensorDataset(Dataset):
    def __init__(self, X, Y, X_mask, Y_mask):
        self.X = X
        self.Y = Y
        self.X_norm = None
        self.Y_norm = None 
        self.X_mask = X_mask
        self.Y_mask = Y_mask
        self.norm_params = {}  # Stores normalization parameters

    def normalize(self, do_normalize=True):
        """Normalize data with optional execution control"""
        if do_normalize:
            # Store normalization parameters
            self.norm_params['X_mean'] = self.X.mean()
            self.norm_params['X_std'] = self.X.std()
            self.norm_params['Y_mean'] = self.Y.mean()
            self.norm_params['Y_std'] = self.Y.std()

            # Apply normalization
            self.X_norm = (self.X - self.norm_params['X_mean']) / self.norm_params['X_std']
            self.Y_norm = (self.Y - self.norm_params['Y_mean']) / self.norm_params['Y_std']

    def denormalize(self):
        """Revert normalized data to original form"""
        if self.norm_params:
            self.X = self.X * self.norm_params['X_std'] + self.norm_params['X_mean']
            self.Y = self.Y * self.norm_params['Y_std'] + self.norm_params['Y_mean']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx]
        
    def shape(self):
        print("Not implemented yet")

def identify_missing_time_stamps(df):
    # make sure columns are in datetime format
    df.columns = pd.to_datetime(df.columns)   
    
    # get first and last time stamp in pd columns
    first_time_stamp = min(df.columns)
    last_time_stamp = max(df.columns)
    
    # Create a full datetime index for the period (e.g., hourly)
    full_time_index = pd.date_range(start=first_time_stamp, end=last_time_stamp, freq='h')

    # Get the columns that are currently in the DataFrame (timestamps)
    current_time_index = df.columns

    # Find missing timestamps
    missing_timestamps = full_time_index.difference(current_time_index)
    
    # Create a DataFrame with NaN values for the missing columns
    missing_df = pd.DataFrame(pd.NA, index=df.index, columns=missing_timestamps)

    # Concatenate the missing columns with the original DataFrame
    df = pd.concat([df, missing_df], axis=1)

    # Sort columns to maintain the original order
    df = df[sorted(df.columns)]

    print("Missing timestamps:", len(missing_timestamps))
    
    return df
    

def load_data():
    # Load and preprocess the data
    index = ["ID"]
    # if MIV/traffic_all_years.csv doesn't exist
    if not os.path.exists("data/processed/MIV/traffic_all_years.csv"):
        # Load the traffic data
        data_manager = TrafficDataManager("data/raw/MIV/")
        data_manager.process_all_years()
        data_road = data_manager.merge_data()
        #data_manager.yearly_availability()
        # Save the final dataset
        data_road.to_csv("data/processed/MIV/traffic_all_years.csv", index=True, index_label=index)
    else:
        data_road = pd.read_csv("data/processed/MIV/traffic_all_years.csv", index_col=index)
        #   Set column MSID as index and drop the column
        #data_road.set_index("MSID", inplace=True)
    
    data_road = identify_missing_time_stamps(data_road)
    
    # Load the active mode data
    index_cols = ["FK_STANDORT", "DIRECTION"]
    if not os.path.exists("data/processed/Langsamverkehr/bike_all_years.csv") or not os.path.exists("data/processed/Langsamverkehr/pedestrian_all_years.csv"):
        data_active_mode = ActiveModeDataLoader("data/raw/Langsamverkehr")
        data_active_mode.process_all_years()
        merged_data = data_active_mode.merge_data()
        #data_active_mode.yearly_availability()
        # Save the final dataset
        
        data_bike = merged_data[0].copy()
        data_pedestrian = merged_data[1].copy()
        
        merged_data[0].to_csv("data/processed/Langsamverkehr/bike_all_years.csv", index=True, index_label=index_cols)
        merged_data[1].to_csv("data/processed/Langsamverkehr/pedestrian_all_years.csv", index=True, index_label=index_cols)
       
    else:
        data_bike = pd.read_csv("data/processed/Langsamverkehr/bike_all_years.csv", index_col=index_cols)
        data_pedestrian = pd.read_csv("data/processed/Langsamverkehr/pedestrian_all_years.csv", index_col=index_cols)
    
    data_bike = identify_missing_time_stamps(data_bike)
    data_pedestrian = identify_missing_time_stamps(data_pedestrian)
    """
    if not os.path.exists("data/processed/VBZ/transit_all_years.csv"):
        # Load the transit data
        data_transit = TransitDataLoader("data/raw/VBZ")
        #data_transit.process_all_years()
        #merged_data = data_transit.merge_data()
    """
    
    return data_road, data_bike, data_pedestrian
    


################################################################################################################################

class BaseDataLoader:
    """Base class for handling datasets with different structures but common later steps."""

    def __init__(self, config_path=None, prints=False):
        self.data = None  # Raw data
        self.geo_data = None  # Geo data
        self.cleaned_data = None  # Processed data
        self.config = self._load_config(config_path)
        self.prints = prints

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _load_config(self, config_path):
        """Loads a JSON configuration file for logging-based operations."""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as file:
                return json.load(file)
        return {}

    def common_operations(self):
        """Shared processing steps (e.g., normalization, outlier removal)."""
        self.data.dropna(inplace=True)


    def missing_values(self):
        """Common transformations applied to all datasets after initial preprocessing."""
        # All years are concatenated into a single dataset as columns
        # Sensors are stored as rows
        # Sensor coordinates are stored in a separate dataframe
        if self.config.get("drop_na"):
            self.cleaned_data = self.data.dropna()
            logging.info("Dropped missing values.")

        if self.config.get("normalize_columns"):
            for col in self.config["normalize_columns"]:
                self.cleaned_data[col] = (self.cleaned_data[col] - self.cleaned_data[col].mean()) / self.cleaned_data[col].std()
            logging.info("Normalized columns.")

    def save_data(self, output_path):
        """Saves processed data to CSV."""
        self.cleaned_data.to_csv(output_path, index=False)
        logging.info(f"Data saved to {output_path}")


class RoadDataLoader(BaseDataLoader):
    """Processes road traffic data with its specific preprocessing steps."""

    def preprocess(self, data_path):
        """Loads and preprocesses road dataset."""
        logging.info("Processing road dataset...")
        self.data = pd.read_csv(data_path)

        # Drop unnecessary columns
        drop_cols = ["MSName", "ZSID", "ZSName", "LieferDat", "Hoehe", "Achse", "HNr"]
        self.data.drop(columns=drop_cols, errors="ignore", inplace=True)

        # Convert missing values
        self.data["AnzFahrzeuge"].replace("Fehlend", np.nan, inplace=True)

        # Aggregate per sensor and timestamp
        self.data = self.data.groupby(["MSID", "MessungDatZeit"]).agg({
            "AnzFahrzeuge": "mean",
            "Richtung": "first",
            "EKoord": "first",
            "NKoord": "first"
        }).reset_index()

        logging.info("Road data preprocessed.")


class TransitDataLoader(BaseDataLoader):
    """Processes transit passenger data with its specific preprocessing steps."""

    def preprocess(self, data_path):
        """Loads and preprocesses transit dataset."""
        logging.info("Processing transit dataset...")
        self.data = pd.read_excel(data_path)

        # Example: Convert time format, remove outliers, etc.
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data = self.data[self.data["passenger_count"] > 0]

        logging.info("Transit data preprocessed.")

class ActivemodeDataLoader(BaseDataLoader):
    """Processes activemode data with its specific preprocessing steps."""

    def preprocess(self, data_path):
        """Loads and preprocesses activemode dataset."""
        logging.info("Processing activemode dataset...")
        self.data = pd.read_csv(data_path)

        # Example: Convert time format, remove outliers, etc.
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data = self.data[self.data["passenger_count"] > 0]

        logging.info("Activemode data preprocessed.")
        
################################################################################################################################


# Attributes
# ----------
# input_data: pd.DataFrame
# input_geo_data: gpd.GeoDataFrame

# Functions
# ----------
# load_csv_data(data_path)
# load_gpkg_data(data_path)
# load_geo_data(data_path)
# save_data(data, path)