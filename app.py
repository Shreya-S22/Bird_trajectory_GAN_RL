import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import time
import pickle
import h5py
import sys
import joblib
from scipy.spatial import ConvexHull
import traceback
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import random
from collections import deque

# Initialize session state variables and creating the streamlit page with the title
if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False
if 'adapted' not in st.session_state:
    st.session_state['adapted'] = False
if 'preprocessed' not in st.session_state:
    st.session_state['preprocessed'] = False
st.set_page_config(
    page_title="Bird Trajectory Prediction",
    page_icon="🦅",
    layout="wide"
)

st.sidebar.markdown("---")
st.sidebar.info("Bird Trajectory Prediction and Anti Poaching Patrol Route analysis App using GAN and RL")


# Custom layer definitions to correctly work for our GAN
class GeographicConstraintLayer(tf.keras.layers.Layer):
    """Custom layer to enforce geographic constraints on trajectory predictions"""
    def __init__(self, coord_ranges=None, **kwargs):
        super(GeographicConstraintLayer, self).__init__(**kwargs)
        self.coord_ranges = coord_ranges or {}

    def call(self, inputs):
        # Extract coordinates 
        coords = inputs[:, :, :2]
        features = inputs[:, :, 2:]
        batch_size = tf.shape(coords)[0]
        seq_len = tf.shape(coords)[1]
        
        # Create hemisphere masks based on starting position
        start_lat = coords[:, 0, 1:2] 
        northern_mask = tf.cast(start_lat >= 0, tf.float32)
        southern_mask = 1.0 - northern_mask
        northern_mask = tf.reshape(northern_mask, [batch_size, 1, 1]) 
        northern_mask = tf.tile(northern_mask, [1, seq_len, 1])    
        southern_mask = tf.reshape(southern_mask, [batch_size, 1, 1]) 
        southern_mask = tf.tile(southern_mask, [1, seq_len, 1]) 
        lats = coords[:, :, 1:2]
        #the maximum latitude and longitude for the bird is provided here such that it does not cross this point
        north_constraint = tf.constant(-30.0, dtype=tf.float32) 
        south_constraint = tf.constant(30.0, dtype=tf.float32) 
        constrained_lats = tf.maximum(lats * northern_mask, north_constraint * northern_mask) + \
                           tf.minimum(lats * southern_mask, south_constraint * southern_mask)
        constrained_coords = tf.concat([coords[:, :, 0:1], constrained_lats], axis=2)
        constrained_output = tf.concat([constrained_coords, features], axis=2) 
        return constrained_output
    
    def get_config(self):
        config = super(GeographicConstraintLayer, self).get_config()
        config.update({"coord_ranges": self.coord_ranges})
        return config

class MinibatchDiscrimination(tf.keras.layers.Layer):
    def __init__(self, num_kernels=5, kernel_dim=3, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.num_kernels * self.kernel_dim),
            initializer='random_normal',
            trainable=True,
            name='minibatch_kernel'
        )
        super(MinibatchDiscrimination, self).build(input_shape)
    
    def call(self, inputs):
        m = tf.matmul(inputs, self.w)
        m = tf.reshape(m, [-1, self.num_kernels, self.kernel_dim])
        diffs = tf.expand_dims(m, 3) - tf.expand_dims(tf.transpose(m, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)      
        return tf.concat([inputs, minibatch_features], 1)
    
    def get_config(self):
        config = super(MinibatchDiscrimination, self).get_config()
        config.update({
            "num_kernels": self.num_kernels,
            "kernel_dim": self.kernel_dim
        })
        return config

class BiologicalConstraintsLayer(tf.keras.layers.Layer):
    #list the biological constraints for the birds
    def __init__(self, max_speed_kmh=80.0, **kwargs):
        super(BiologicalConstraintsLayer, self).__init__(**kwargs)
        self.max_speed_kmh = max_speed_kmh
    
    def call(self, inputs):
        coords = inputs[:, :, :2]
        features = inputs[:, :, 2:]
        batch_size = tf.shape(coords)[0]
        seq_len = tf.shape(coords)[1]
        start_lat = coords[:, 0, 1:2] 
        northern_mask = tf.cast(start_lat >= 0, tf.float32)
        southern_mask = 1.0 - northern_mask
        northern_mask = tf.reshape(northern_mask, [batch_size, 1, 1]) 
        northern_mask = tf.tile(northern_mask, [1, seq_len, 1]) 
        
        southern_mask = tf.reshape(southern_mask, [batch_size, 1, 1]) 
        southern_mask = tf.tile(southern_mask, [1, seq_len, 1]) 

        north_constraint = tf.constant(-30.0, dtype=tf.float32)
        south_constraint = tf.constant(30.0, dtype=tf.float32)
        lats = coords[:, :, 1:2]
        constrained_lats = tf.maximum(lats * northern_mask, north_constraint * northern_mask) + \
                           tf.minimum(lats * southern_mask, south_constraint * southern_mask)
        constrained_coords = tf.concat([coords[:, :, 0:1], constrained_lats], axis=2)
        constrained_output = tf.concat([constrained_coords, features], axis=2)
        
        return constrained_output
    
    def get_config(self):
        config = super(BiologicalConstraintsLayer, self).get_config()
        config.update({"max_speed_kmh": self.max_speed_kmh})
        return config

class ContinuityEnforcementLayer(tf.keras.layers.Layer):
    # class to make sure there is continuity between past and future trajectories"""
    def __init__(self, sequence_length=15, **kwargs):
        super(ContinuityEnforcementLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
    
    def call(self, inputs):
        return inputs
    
    def get_config(self):
        config = super(ContinuityEnforcementLayer, self).get_config()
        config.update({"sequence_length": self.sequence_length})
        return config

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def enhanced_continuity_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

custom_objects = {
    'GeographicConstraintLayer': GeographicConstraintLayer,
    'BiologicalConstraintsLayer': BiologicalConstraintsLayer,
    'MinibatchDiscrimination': MinibatchDiscrimination,
    'ContinuityEnforcementLayer': ContinuityEnforcementLayer,
    'wasserstein_loss': wasserstein_loss,
    'enhanced_continuity_loss': enhanced_continuity_loss
}

tf.keras.utils.get_custom_objects().update(custom_objects)

def haversine_distance(lat1, lon1, lat2, lon2):
    #Calculate the great circle distance between two points
    # Convert decimal degrees to radians and use the haversine distance frmula
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371     
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360    
    return bearing

def simple_land_classification(lon, lat):
    # Define simple continent bounding boxes 
    continents = [
        # North America
        {'lon_min': -170, 'lon_max': -50, 'lat_min': 15, 'lat_max': 75},
        # South America
        {'lon_min': -90, 'lon_max': -30, 'lat_min': -60, 'lat_max': 15},
        # Europe and Africa
        {'lon_min': -20, 'lon_max': 60, 'lat_min': -40, 'lat_max': 75},
        # Asia
        {'lon_min': 60, 'lon_max': 180, 'lat_min': 0, 'lat_max': 75},
        # Australia
        {'lon_min': 110, 'lon_max': 155, 'lat_min': -45, 'lat_max': -10},
    ]
    
    is_on_land = np.zeros_like(lon, dtype=bool)
    for continent in continents:
        in_continent = ((lon >= continent['lon_min']) &
                        (lon <= continent['lon_max']) &
                        (lat >= continent['lat_min']) &
                        (lat <= continent['lat_max']))
        is_on_land = is_on_land | in_continent
    
    return is_on_land.astype(int)

def point_in_hull(point, hull, tolerance=1e-12):
    try:
        return all(
            np.dot(eq[:-1], point) + eq[-1] <= tolerance
            for eq in hull.equations
        )
    except:
        return False

def check_required_columns(df):
    #Check if the dataset already has the required columns for preprocessing.
    required_columns = [
        'date', 'timestamp', 'location-long', 'location-lat', 'tag-local-identifier'
    ]
    
    has_required = all(col in df.columns for col in required_columns)
    vegetation_columns = [col for col in df.columns if 'Vegetation' in col]
    has_vegetation = len(vegetation_columns) > 0
    
    return has_required, has_vegetation

def adapt_dataset(df, output_file=None):
    #Adapt dataset to match the format expected byGAN 
    st.write("Adapting dataset format...")
    has_required, has_vegetation = check_required_columns(df)
    
    if has_required:
        st.success("Dataset already has the required format. Proceeding to next step.")
        if output_file:
            df.to_csv(output_file, index=False)
        return df

    columns_mapping = {
        'timestamp': {'timestamp', 'datetime', 'date_time', 'time', 'observation_time'},
        'location-long': {'longitude', 'long', 'lng', 'lon', 'x', 'easting', 'x_coord'},
        'location-lat': {'latitude', 'lat', 'y', 'northing', 'y_coord'},
        'tag-local-identifier': {'tag_id', 'tag', 'bird_id', 'animal_id', 'device_info_serial',
                                'device_id', 'id', 'individual_id', 'bird_name', 'individual-local-identifier'},
    }
    
    # Create a new dataframe for the adapted data
    adapted_df = pd.DataFrame()

    for target_col, possible_names in columns_mapping.items():
        found = False
        for col_name in df.columns:
            if col_name.lower() in possible_names or col_name == target_col:
                if target_col == 'timestamp' and 'date_time' in possible_names:
                    adapted_df[target_col] = df[col_name]
                    found = True
                    break
                else:
                    adapted_df[target_col] = df[col_name]
                    found = True
                    break
        
        if not found:
            st.warning(f"Could not find a column matching {target_col}")
            if target_col == 'tag-local-identifier':
                st.info("Creating sequential tag identifiers")
                adapted_df[target_col] = "tag_" + df.index.astype(str)
    
    # Handle date and timestamp
    if 'timestamp' in adapted_df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(adapted_df['timestamp']):
                date_formats = [
                    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                    '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                    '%Y/%m/%d %H:%M', '%d-%m-%Y %H:%M', '%d/%m/%Y %H:%M'
                ]
                
                for fmt in date_formats:
                    try:
                        adapted_df['timestamp'] = pd.to_datetime(adapted_df['timestamp'], format=fmt)
                        break
                    except:
                        continue
                
                if not pd.api.types.is_datetime64_any_dtype(adapted_df['timestamp']):
                    adapted_df['timestamp'] = pd.to_datetime(adapted_df['timestamp'], errors='coerce')

            adapted_df['date'] = adapted_df['timestamp'].dt.strftime('%Y-%m-%d')
            adapted_df['timestamp'] = adapted_df['timestamp'].dt.strftime('%H:%M:%S')
        
        except Exception as e:
            st.error(f"Error parsing timestamp: {e}")
            adapted_df['date'] = datetime.now().strftime('%Y-%m-%d')
            adapted_df['timestamp'] = datetime.now().strftime('%H:%M:%S')
    else:
        st.warning("No timestamp column found, creating default values")
        adapted_df['date'] = datetime.now().strftime('%Y-%m-%d')
        adapted_df['timestamp'] = datetime.now().strftime('%H:%M:%S')

    vegetation_cols = [
        'ECMWF Interim Full Daily Invariant Low Vegetation Cover',
        'ECMWF Interim Full Daily Invariant High Vegetation Cover'
    ]
    
    for col in vegetation_cols:
        adapted_df[col] = np.nan

    if 'event-id' not in adapted_df.columns:
        adapted_df['event-id'] = adapted_df.index
    
    if 'visible' not in adapted_df.columns:
        adapted_df['visible'] = 1
    
    if 'manually-marked-outlier' not in adapted_df.columns:
        adapted_df['manually-marked-outlier'] = 0
    
    if 'sensor-type' not in adapted_df.columns:
        adapted_df['sensor-type'] = 'GPS'
    
    if 'individual-taxon-canonical-name' not in adapted_df.columns:
        if 'bird_name' in df.columns:
            adapted_df['individual-taxon-canonical-name'] = df['bird_name']
        else:
            adapted_df['individual-taxon-canonical-name'] = 'unknown'
    
    if 'study-name' not in adapted_df.columns:
        adapted_df['study-name'] = 'bird_migration_study'

    if 'speed_2d' in df.columns or 'speed' in df.columns:
        speed_col = 'speed_2d' if 'speed_2d' in df.columns else 'speed'
        adapted_df['speed'] = df[speed_col]
        
    adapted_df = adapted_df.fillna(0)
    
    # Saving the adapted dataset after ensuring all columns are fine
    if output_file:
        adapted_df.to_csv(output_file, index=False)
        st.success(f"Dataset adapted successfully. Saved to {output_file}")
    
    st.info(f"The adapted dataset has {len(adapted_df)} rows and {len(adapted_df.columns)} columns.")
    st.write("Columns in adapted dataset:", adapted_df.columns.tolist())
    
    return adapted_df

def add_geographic_constraints(df):
    df['is_on_land'] = simple_land_classification(df['location-long'], df['location-lat'])
    df['distance_to_equator'] = np.abs(df['location-lat'])
    df['northern_hemisphere'] = (df['location-lat'] > 0).astype(int)
    df['distance_to_poles'] = 90 - df['location-lat'].abs()
    tag_mean_lat = df.groupby('tag-local-identifier')['location-lat'].mean()
    df['main_hemisphere'] = df['tag-local-identifier'].map(tag_mean_lat).apply(lambda x: 1 if x > 0 else -1)
    df['hemisphere_change'] = (np.sign(df['location-lat']) != df['main_hemisphere']).astype(int)
    tag_lat_range = df.groupby('tag-local-identifier')['location-lat'].agg(['min', 'max'])
    df['tag_lat_min'] = df['tag-local-identifier'].map(tag_lat_range['min'])
    df['tag_lat_max'] = df['tag-local-identifier'].map(tag_lat_range['max'])

    df['in_normal_lat_range'] = (
        (df['location-lat'] >= df['tag_lat_min']) &
        (df['location-lat'] <= df['tag_lat_max'])
    ).astype(int)
    
    return df

def add_seasonal_migration_patterns(df):
    """
    Extract and add seasonal migration pattern features based on month and latitude
    """
    if 'month_numeric' not in df.columns:
        if 'date' in df.columns:
            # Convert date to datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                df['date_dt'] = df['date']

            df['month_numeric'] = df['date_dt'].dt.month
        else:

            st.warning("No date column found for seasonal patterns, using current month")
            df['month_numeric'] = datetime.now().month
    
    # Seasonal migration direction: positive means northward, negative means southward
    conditions = [
        (df['location-lat'] > 0) & (df['month_numeric'].between(2, 5)), # Northern spring
        (df['location-lat'] > 0) & (df['month_numeric'].between(8, 11)), # Northern fall
        (df['location-lat'] < 0) & (df['month_numeric'].between(2, 5)), # Southern fall
        (df['location-lat'] < 0) & (df['month_numeric'].between(8, 11)) # Southern spring
    ]
    
    values = [1, -1, -1, 1] 
    df['seasonal_direction'] = np.select(conditions, values, default=0)

    monthly_lat_means = df.groupby(['tag-local-identifier', 'month_numeric'])['location-lat'].mean().reset_index()
    monthly_lat_means.columns = ['tag-local-identifier', 'month_numeric', 'avg_monthly_lat']
    

    df = pd.merge(df, monthly_lat_means, on=['tag-local-identifier', 'month_numeric'], how='left')
    

    df['lat_deviation'] = df['location-lat'] - df['avg_monthly_lat']

    df['breeding_season'] = np.select([
        (df['location-lat'] > 0) & (df['month_numeric'].between(5, 8)), # Northern breeding
        (df['location-lat'] < 0) & ((df['month_numeric'] >= 11) | (df['month_numeric'] <= 2)) # Southern breeding
    ],
    [1, 1],
    default=0)
    
    df['winter_season'] = np.select([
        (df['location-lat'] > 0) & ((df['month_numeric'] >= 11) | (df['month_numeric'] <= 2)), # Northern winter
        (df['location-lat'] < 0) & (df['month_numeric'].between(5, 8)) # Southern winter
    ],
    [1, 1],
    default=0)
    
    return df

def add_migration_corridor_features(df):
    tag_ids = df['tag-local-identifier'].unique()
    df['in_migration_corridor'] = 0
    df['distance_to_corridor'] = 1.0     
    for tag_id in tag_ids:
        tag_data = df[df['tag-local-identifier'] == tag_id]
        if len(tag_data) < 5:
            continue
        points = tag_data[['location-long', 'location-lat']].values
        
        try:

            if len(np.unique(points, axis=0)) >= 3: 
                hull = ConvexHull(points)
                hull_path = points[hull.vertices]

                for idx, row in tag_data.iterrows():
                    point = np.array([row['location-long'], row['location-lat']])

                    if point_in_hull(point, hull):
                        df.loc[idx, 'in_migration_corridor'] = 1
                        df.loc[idx, 'distance_to_corridor'] = 0
                    else:
                        distances = np.sqrt(np.sum((points[hull.vertices] - point)**2, axis=1))
                        min_distance = np.min(distances)
                        df.loc[idx, 'distance_to_corridor'] = min_distance
        
        except Exception as e:
            st.warning(f"Could not calculate corridor for tag {tag_id}: {str(e)}")

    max_dist = df['distance_to_corridor'].max()
    if max_dist > 0:
        df['distance_to_corridor'] = df['distance_to_corridor'] / max_dist
    
    return df

def transform_coordinates(coords, coord_ranges, inverse=False):
    if inverse:
        lon = coords[:, 0] * (coord_ranges['lon_max'] - coord_ranges['lon_min']) + coord_ranges['lon_min']
        lat = coords[:, 1] * (coord_ranges['lat_max'] - coord_ranges['lat_min']) + coord_ranges['lat_min']
    else:
        lon = (coords[:, 0] - coord_ranges['lon_min']) / (coord_ranges['lon_max'] - coord_ranges['lon_min'])
        lat = (coords[:, 1] - coord_ranges['lat_min']) / (coord_ranges['lat_max'] - coord_ranges['lat_min'])
    
    return np.column_stack((lon, lat))

def ensure_trajectory_continuity(past_trajectory, predicted_trajectory, decay_rate=0.9, num_points=10):
    last_past_point = past_trajectory[-1, :2].copy()
    if past_trajectory.shape[0] >= 3:

        direction1 = past_trajectory[-1, :2] - past_trajectory[-2, :2]
        direction2 = past_trajectory[-2, :2] - past_trajectory[-3, :2]
        avg_direction = (direction1 + direction2) / 2
        norm = np.linalg.norm(avg_direction)
        if norm > 0:
            avg_direction = avg_direction / norm
            predicted_trajectory[0, :2] = last_past_point
            for i in range(1, min(num_points, len(predicted_trajectory))):
                decay = decay_rate ** i
                blend_factor = 1 - decay
                current_direction = predicted_trajectory[i, :2] - predicted_trajectory[i-1, :2]
                if np.linalg.norm(current_direction) > 0:
                    current_direction = current_direction / np.linalg.norm(current_direction)
                    blended_direction = avg_direction * decay + current_direction * blend_factor
                    predicted_trajectory[i, :2] = predicted_trajectory[i-1, :2] + blended_direction * 0.05
    else:
        predicted_trajectory[0, :2] = last_past_point
    return predicted_trajectory


def create_or_update_scaler(feature_cols):
    st.info(f"Creating/updating scaler for {len(feature_cols)} features")
    if os.path.exists('preprocessed_data.csv'):
        df = pd.read_csv('preprocessed_data.csv')
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        #create and save scalar
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])
        joblib.dump((scaler, feature_cols), 'feature_scaler.pkl')
        st.success(f"Created and saved new scaler with {len(feature_cols)} features")
        
        return scaler, feature_cols
    else:
        scaler = MinMaxScaler()
        st.warning("No data available to fit the scaler. Will fit during prediction.")
        
        return scaler, feature_cols

def preprocess_for_gan(df, output_file=None, min_points=15):
    #Preprocess data for GAN training with support for all 19 features
    st.write("Preprocessing data for GAN...")
    st.write("Columns before preprocessing:", df.columns.tolist())
    #check the date and time columns
    if 'date' not in df.columns:
        st.error("The 'date' column is missing from the dataset. Please ensure your data has a date column.")
        return None
    if 'timestamp' not in df.columns:
        st.error("The 'timestamp' column is missing from the dataset.")
        return None
    
    try:
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.drop('date', axis=1)
        critical_columns = ['location-long', 'location-lat', 'timestamp', 'tag-local-identifier']
        df = df.dropna(subset=critical_columns)
        vegetation_columns = [col for col in df.columns if 'Vegetation' in col and col != 'NCEP NARR SFC Vegetation at Surface']
        selected_columns = ['timestamp', 'location-long', 'location-lat', 'tag-local-identifier'] + vegetation_columns
        df = df[selected_columns]
        for col in vegetation_columns:
            df[col] = df.groupby('tag-local-identifier')[col].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
            )
        
        tag_counts = df['tag-local-identifier'].value_counts()
        valid_tags = tag_counts[tag_counts >= min_points].index
        df_filtered = df[df['tag-local-identifier'].isin(valid_tags)]
        
        if len(valid_tags) == 0:
            st.error(f"No birds have sufficient data points (at least {min_points}). Please upload a dataset with more observations.")
            return None
        
        st.success(f"Found {len(valid_tags)} birds with sufficient data points.")
        df_filtered = df_filtered.sort_values(['tag-local-identifier', 'timestamp'])
        df_filtered['day_of_year'] = df_filtered['timestamp'].dt.dayofyear / 365
        df_filtered['hour_of_day'] = df_filtered['timestamp'].dt.hour / 24
        df_filtered['day_sin'] = np.sin(2 * np.pi * df_filtered['day_of_year'])
        df_filtered['day_cos'] = np.cos(2 * np.pi * df_filtered['day_of_year'])
        df_filtered['hour_sin'] = np.sin(2 * np.pi * df_filtered['hour_of_day'])
        df_filtered['hour_cos'] = np.cos(2 * np.pi * df_filtered['hour_of_day'])

        df_filtered['month'] = df_filtered['timestamp'].dt.month
        df_filtered['year'] = df_filtered['timestamp'].dt.year
        df_filtered['prev_long'] = df_filtered.groupby('tag-local-identifier')['location-long'].shift(1)
        df_filtered['prev_lat'] = df_filtered.groupby('tag-local-identifier')['location-lat'].shift(1)
        df_filtered['prev_timestamp'] = df_filtered.groupby('tag-local-identifier')['timestamp'].shift(1)
        df_filtered['time_diff_hours'] = df_filtered.groupby('tag-local-identifier')['timestamp'].diff().dt.total_seconds() / 3600
        df_filtered['distance_km'] = haversine_distance(
            df_filtered['prev_lat'], df_filtered['prev_long'],
            df_filtered['location-lat'], df_filtered['location-long']
        )
        
        df_filtered['speed_kmh'] = df_filtered['distance_km'] / df_filtered['time_diff_hours']
        df_filtered['speed_kmh'] = df_filtered['speed_kmh'].replace([np.inf, -np.inf, np.nan], 0)

        df_filtered['orig_location-long'] = df_filtered['location-long']
        df_filtered['orig_location-lat'] = df_filtered['location-lat']
        speed_threshold = df_filtered['speed_kmh'].quantile(0.99)
        df_filtered['speed_outlier'] = df_filtered['speed_kmh'] > speed_threshold
        
        # Calculate sequence lengths for each tag (useful for padding in deep learning)
        sequence_lengths = df_filtered.groupby('tag-local-identifier').size()
        df_filtered['sequence_length'] = df_filtered['tag-local-identifier'].map(sequence_lengths)
        
        # Create a sequence ID for each tag (useful for GAN training)
        df_filtered['sequence_id'] = df_filtered.groupby('tag-local-identifier').ngroup()
        
        # Calculate direction of movement (bearing)
        df_filtered['bearing'] = calculate_bearing(
            df_filtered['prev_lat'], df_filtered['prev_long'],
            df_filtered['location-lat'], df_filtered['location-long']
        )
        
        # Calculate acceleration
        df_filtered['prev_speed'] = df_filtered.groupby('tag-local-identifier')['speed_kmh'].shift(1)
        df_filtered['prev_time_diff'] = df_filtered.groupby('tag-local-identifier')['time_diff_hours'].shift(1)
        df_filtered['acceleration'] = (df_filtered['speed_kmh'] - df_filtered['prev_speed']) / df_filtered['time_diff_hours']
        df_filtered['acceleration'] = df_filtered['acceleration'].replace([np.inf, -np.inf, np.nan], 0)
        
        # Add geographic constraint features
        df_filtered = add_geographic_constraints(df_filtered)
        
        # Add seasonal migration patterns
        df_filtered = add_seasonal_migration_patterns(df_filtered)
        
        # Add species-specific migration corridors
        df_filtered = add_migration_corridor_features(df_filtered)
        
        # Get hemisphere for each tag to create hemisphere-specific constraints
        tag_hemispheres = {}
        for tag in df_filtered['tag-local-identifier'].unique():
            tag_data = df_filtered[df_filtered['tag-local-identifier'] == tag]
            mean_lat = tag_data['location-lat'].mean()
            tag_hemispheres[tag] = 1 if mean_lat > 0 else -1
        
        # Add hemisphere information
        df_filtered['hemisphere'] = df_filtered['tag-local-identifier'].map(tag_hemispheres)
        
        # Create a copy of original values before normalization
        features_to_normalize = ['location-long', 'location-lat', 'distance_km', 'speed_kmh',
                                'bearing', 'acceleration'] + vegetation_columns
        
        for col in features_to_normalize:
            df_filtered[f'orig_{col}'] = df_filtered[col]
        
        # Create biologically constrained coordinate ranges
        coord_ranges = {
            'lon_min': df_filtered['location-long'].min(),
            'lon_max': df_filtered['location-long'].max(),
            'lat_min': df_filtered['location-lat'].min(),
            'lat_max': df_filtered['location-lat'].max(),
            # Add hemispheric constraints
            'north_constraint': max(10, df_filtered[df_filtered['hemisphere'] > 0]['location-lat'].min()),
            'south_constraint': min(-10, df_filtered[df_filtered['hemisphere'] < 0]['location-lat'].max())
        }
        
        # Save these ranges for use during prediction
        joblib.dump(coord_ranges, 'coord_ranges.pkl')
        
        # Define the exact feature columns to match GAN training - UPDATED TO INCLUDE ALL 19 FEATURES
        feature_cols = [
            'location-long',
            'location-lat',
            'ECMWF Interim Full Daily Invariant Low Vegetation Cover',
            'ECMWF Interim Full Daily Invariant High Vegetation Cover',
            'day_of_year',
            'hour_of_day',
            'day_sin',
            'day_cos',
            'hour_sin',
            'hour_cos',
            'time_diff_hours',
            'distance_km',
            'speed_kmh',
            'bearing',
            'is_on_land',
            'distance_to_equator',
            'northern_hemisphere',
            'distance_to_poles',
            'in_normal_lat_range'
        ]
        for col in feature_cols:
            if col not in df_filtered.columns:
                st.warning(f"Feature column {col} not found in data. Adding it with default values.")

        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        features_to_normalize = feature_cols.copy()
        scaler.fit(df_filtered[features_to_normalize])
        df_filtered[features_to_normalize] = scaler.transform(df_filtered[features_to_normalize])
        joblib.dump((scaler, feature_cols), 'feature_scaler.pkl')
        st.success(f"Saved scaler with {len(feature_cols)} features: {feature_cols}")
        columns_to_drop = ['prev_long', 'prev_lat', 'prev_timestamp', 'prev_speed', 'prev_time_diff']
        df_filtered = df_filtered.drop(columns_to_drop, axis=1)
        df_filtered = df_filtered.fillna(0)
        
        if output_file:
            df_filtered.to_csv(output_file, index=False)
            st.success(f"Preprocessing complete. Output saved to {output_file}")
        
        st.info(f"Dataset shape: {df_filtered.shape}")
        st.info(f"Number of unique tags: {df_filtered['tag-local-identifier'].nunique()}")
        
        return df_filtered
    
    except Exception as e:
        st.error(f"An error occurred during preprocessing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def predict_trajectory_for_tag(generator, tag_data, tag_id, scaler=None, feature_cols=None, 
                              latent_dim=64, sequence_length=15):
    #Generate trajectory prediction for a single tag
    tag_data = tag_data.sort_values('timestamp')
    if len(tag_data) < sequence_length:
        st.warning(f"Not enough data points for tag {tag_id}. Need at least {sequence_length}, but got {len(tag_data)}.")
        return None
    past_trajectory = tag_data.iloc[-sequence_length:].copy()
    if feature_cols is not None:
        missing_features = [col for col in feature_cols if col not in past_trajectory.columns]
        for col in missing_features:
            past_trajectory[col] = 0.0
        past_features = past_trajectory[feature_cols].values
    else:
        # Use default features - include all 19 features
        default_features = [
            'location-long', 'location-lat',
            'ECMWF Interim Full Daily Invariant Low Vegetation Cover',
            'ECMWF Interim Full Daily Invariant High Vegetation Cover',
            'day_of_year', 'hour_of_day', 'day_sin', 'day_cos',
            'hour_sin', 'hour_cos', 'time_diff_hours', 'distance_km',
            'speed_kmh', 'bearing', 'is_on_land', 'distance_to_equator',
            'northern_hemisphere', 'distance_to_poles', 'in_normal_lat_range'
        ]
        
        # Add missing columns with default values
        for col in default_features:
            if col not in past_trajectory.columns:
                past_trajectory[col] = 0.0
        
        past_features = past_trajectory[default_features].values
    
    # Handle feature normalization with improved error handling
    if scaler:
        try:
            if past_features.shape[1] != scaler.n_features_in_:
                st.warning(f"Feature dimension mismatch. Creating a new scaler for {past_features.shape[1]} features.")
                temp_scaler = MinMaxScaler()
                past_features = temp_scaler.fit_transform(past_features)
                if feature_cols:
                    joblib.dump((temp_scaler, feature_cols), 'feature_scaler_updated.pkl')
                    st.success(f"Created and saved new scaler with {len(feature_cols)} features")
            else:
                past_features = scaler.transform(past_features)
        except ValueError as ve:
            st.info(f"Creating a new scaler to handle {past_features.shape[1]} features")
            temp_scaler = MinMaxScaler()
            past_features = temp_scaler.fit_transform(past_features)
            if feature_cols:
                joblib.dump((temp_scaler, feature_cols), 'feature_scaler_updated.pkl')
                st.success(f"Created and saved new scaler with {len(feature_cols)} features")
    env_conditions = past_features[0, 2:]
    env_conditions = env_conditions.reshape(1, -1)
    tag_seed = int(str(tag_id).replace('tag_', '').replace('Tag_', '').replace('_', '')) % 10000
    np.random.seed(tag_seed)
    noise = np.random.normal(0, 1, (1, latent_dim))
    tag_id_numeric = tag_seed % 100
    tag_id_input = np.array([[tag_id_numeric]])
    if len(past_features.shape) == 2:
        past_features = np.expand_dims(past_features, axis=0)
    try:
        st.info(f"Generating prediction for tag {tag_id} with feature shape {past_features.shape}")
        pred = generator.predict([noise, past_features, env_conditions, tag_id_input], verbose=0)
        st.success(f"Successfully generated prediction for tag {tag_id}")
        return pred[0]
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        st.error(traceback.format_exc())
        return None
    
def transform_coordinates_with_scaler(normalized_coords, scaler, feature_cols):
    feature_array = np.zeros((normalized_coords.shape[0], len(feature_cols)))

    try:
        lon_idx = feature_cols.index('location-long')
        lat_idx = feature_cols.index('location-lat')
    except ValueError:
        # If exact names not found, try to find columns containing 'lon' and 'lat'
        lon_idx = next((i for i, col in enumerate(feature_cols) if 'lon' in col.lower()), 0)
        lat_idx = next((i for i, col in enumerate(feature_cols) if 'lat' in col.lower()), 1)
    feature_array[:, lon_idx] = normalized_coords[:, 0]
    feature_array[:, lat_idx] = normalized_coords[:, 1]
    
    # Apply inverse transform using the scaler
    try:
        original_features = scaler.inverse_transform(feature_array)
        original_coords = np.column_stack((original_features[:, lon_idx], original_features[:, lat_idx]))
        return original_coords
    except Exception as e:
        print(f"Error in inverse transform: {e}")
        return normalized_coords[:, :2]

def transform_coordinates_back(normalized_coords, coord_ranges):
    """Transform normalized coordinates back to original space with proper scaling"""
    if np.max(normalized_coords[:, 0]) > 1.5 or np.min(normalized_coords[:, 0]) < -1.5:
        # Coordinates might already be in original space
        return normalized_coords[:, :2]
    
    # Apply inverse transformation
    lon = normalized_coords[:, 0] * (coord_ranges['lon_max'] - coord_ranges['lon_min']) + coord_ranges['lon_min']
    lat = normalized_coords[:, 1] * (coord_ranges['lat_max'] - coord_ranges['lat_min']) + coord_ranges['lat_min']
    
    return np.column_stack((lon, lat))


def apply_tag_specific_adjustments(predictions): 
    adjusted_predictions = {}
    for tag_id, pred_data in predictions.items():
        adjusted_predictions[tag_id] = pred_data
    for tag_id, pred_data in adjusted_predictions.items():
        past_traj = pred_data['past']
        pred_traj = pred_data['predicted']
        if past_traj.shape[0] > 1:
            past_direction = past_traj[-1, :2] - past_traj[-2, :2]

            norm = np.linalg.norm(past_direction)
            if norm > 0:
                past_direction = past_direction / norm

            tag_id_numeric = int(str(tag_id).replace('tag_', '').replace('Tag_', '').replace('_', '')) % 10000
            freq1 = 0.1 + (tag_id_numeric % 10) * 0.02
            freq2 = 0.15 + (tag_id_numeric % 7) * 0.03
            amp1 = 0.05 + (tag_id_numeric % 5) * 0.01
            amp2 = 0.04 + (tag_id_numeric % 6) * 0.01
            diversity_vector = np.array([
                np.sin(tag_id_numeric * freq1) * amp1,
                np.cos(tag_id_numeric * freq2) * amp2
            ])
            for i in range(len(pred_traj)):
                oscillation = np.array([
                    np.sin(i * 0.2 + tag_id_numeric * 0.1) * 0.02,
                    np.cos(i * 0.3 + tag_id_numeric * 0.2) * 0.02
                ])
                
                decay = 0.9 ** i
                pred_traj[i, :2] += past_direction * 0.15 * decay + diversity_vector * decay + oscillation * decay

            pred_traj[0, :2] = past_traj[-1, :2]

            adjusted_predictions[tag_id]['predicted'] = pred_traj
    
    return adjusted_predictions



def predict_trajectories(processed_df, model_files):

    st.write("Generating trajectory predictions...")

    predictions = {}
    generator = None
    feature_cols = None
    scaler = None
    
    if model_files['generator']:
        try:
            generator, feature_cols, scaler = load_model_safely(
                model_files['generator'], 
                model_files.get('metadata')
            )
            
            if generator is None:
                st.error("Failed to load generator model")
                return None
                
            st.success("Generator model loaded successfully")
        except Exception as e:
            st.error(f"Error loading generator model: {e}")
            st.error(traceback.format_exc())
            return None
    
    # Load coordinate ranges
    coord_ranges = None
    if os.path.exists('coord_ranges.pkl'):
        try:
            coord_ranges = joblib.load('coord_ranges.pkl')
            st.success("Coordinate ranges loaded successfully")
        except Exception as e:
            st.warning(f"Error loading coordinate ranges: {e}")

    metadata = None
    if model_files.get('metadata') and os.path.exists(model_files['metadata']):
        try:
            with open(model_files['metadata'], 'rb') as f:
                metadata = pickle.load(f)
            st.success("Metadata loaded successfully")
            
            # Extract parameters from metadata
            sequence_length = metadata.get('sequence_length', 15)
            latent_dim = metadata.get('latent_dim', 64)
            
            # Use feature columns from metadata if available
            if feature_cols is None and 'feature_cols' in metadata:
                feature_cols = metadata['feature_cols']
        except Exception as e:
            st.warning(f"Error loading metadata: {e}")
            sequence_length = 15
            latent_dim = 64
    else:
        sequence_length = 15
        latent_dim = 64
    
    # Group by tag_id to process each bird separately
    tag_groups = processed_df.groupby('tag-local-identifier')
    
    # Process each tag
    for tag_id, group in tag_groups:
        st.write(f"Processing tag: {tag_id}")
        if len(group) < sequence_length:
            st.warning(f"Tag {tag_id} has only {len(group)} points, need at least {sequence_length}")
            continue
        
        try:
            group = group.sort_values('timestamp')
            past_trajectory = group.iloc[-sequence_length:].copy()
            # Store original coordinates for visualization BEFORE normalization
            orig_coords = past_trajectory[['location-long', 'location-lat']].values
            if feature_cols is None:
                feature_cols = [
                    'location-long', 'location-lat',
                    'ECMWF Interim Full Daily Invariant Low Vegetation Cover',
                    'ECMWF Interim Full Daily Invariant High Vegetation Cover',
                    'day_of_year', 'hour_of_day', 'day_sin', 'day_cos',
                    'hour_sin', 'hour_cos', 'time_diff_hours', 'distance_km',
                    'speed_kmh', 'bearing', 'is_on_land', 'distance_to_equator',
                    'northern_hemisphere', 'distance_to_poles', 'in_normal_lat_range'
                ]

            for col in feature_cols:
                if col not in past_trajectory.columns:
                    past_trajectory[col] = 0.0
            
            # Extract features in correct order
            past_features = past_trajectory[feature_cols].values
            
            # Normalize features if scaler is available
            if scaler:
                try:
                    # Check if dimensions match
                    if past_features.shape[1] != scaler.n_features_in_:
                        st.warning(f"Feature dimension mismatch. Creating a new scaler for {past_features.shape[1]} features.")
                        temp_scaler = MinMaxScaler()
                        past_features = temp_scaler.fit_transform(past_features)
                    else:
                        past_features = scaler.transform(past_features)
                except ValueError as ve:
                    st.info(f"Creating a new scaler to handle {past_features.shape[1]} features")
                    temp_scaler = MinMaxScaler()
                    past_features = temp_scaler.fit_transform(past_features)
            
            # Extract environmental conditions 
            env_conditions = past_features[0, 2:]
            env_conditions = env_conditions.reshape(1, -1)
            
            # Generate tag-specific seed for consistent predictions
            tag_seed = int(str(tag_id).replace('tag_', '').replace('Tag_', '').replace('_', '')) % 10000
            np.random.seed(tag_seed)
            noise = np.random.normal(0, 1, (1, latent_dim))
            
            # Convert tag_id to numeric for embedding layer
            tag_id_numeric = tag_seed % 100
            tag_id_input = np.array([[tag_id_numeric]])
            
            # Reshape past_features to add batch dimension if needed
            if len(past_features.shape) == 2:
                past_features = np.expand_dims(past_features, axis=0)
            
            # Generate prediction
            pred = generator.predict([noise, past_features, env_conditions, tag_id_input], verbose=0)
            st.success(f"Successfully generated prediction for tag {tag_id}")
            
            # Ensure trajectory continuity
            predicted_trajectory = ensure_trajectory_continuity(
                past_features[0], 
                pred[0],
                decay_rate=0.9,
                num_points=10
            )
            
            # Store predictions with both normalized and original coordinates
            predictions[tag_id] = {
                'past': past_features[0],
                'predicted': predicted_trajectory,
                'orig_past_coords': orig_coords,
                'tag_id': tag_id
            }
            
        except Exception as e:
            st.error(f"Error generating prediction for tag {tag_id}: {e}")
            st.error(traceback.format_exc())
    
    if not predictions:
        st.warning("No predictions were generated. Check the logs for errors.")
        return None
    predictions = apply_tag_specific_adjustments(predictions)
    # Transform coordinates back to original space
    if coord_ranges:
        for tag_id, pred_data in predictions.items():
            if scaler and feature_cols:
                orig_predicted_coords = transform_coordinates_with_scaler(
                    pred_data['predicted'][:, :2], 
                    scaler, 
                    feature_cols
                )
            else:
                orig_predicted_coords = transform_coordinates_back(
                    pred_data['predicted'][:, :2], 
                    coord_ranges
                )
            predictions[tag_id]['orig_predicted_coords'] = orig_predicted_coords
    visualize_predictions_with_orig_coords(predictions, coord_ranges)
    
    return predictions



def visualize_predictions_with_orig_coords(predictions, coord_ranges=None):
    """Visualize the predicted trajectories using original coordinates"""
    if not predictions:
        st.warning("No predictions to visualize")
        return
    
    st.subheader("Predicted Trajectories (Original Coordinates)")
    trajectory_dir = "visualize_trajectory"
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
        st.info(f"Created directory: {trajectory_dir}")
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

        # Create a new figure for each tag
    for i, (tag_id, pred_data) in enumerate(predictions.items()):
        fig, ax = plt.subplots(figsize=(10, 8))
        past_coords = pred_data['orig_past_coords']
        if 'orig_predicted_coords' in pred_data:
            pred_coords = pred_data['orig_predicted_coords']
        elif scaler and feature_cols:
            pred_coords = transform_coordinates_with_scaler(
                pred_data['predicted'][:, :2],
                scaler,
                feature_cols
            )
        elif coord_ranges:
            pred_coords = transform_coordinates_back(pred_data['predicted'][:, :2], coord_ranges)
        else:
            pred_coords = pred_data['predicted'][:, :2]
            st.warning(f"Using normalized coordinates for tag {tag_id} - visualization may not be accurate")
        #Plot the trajectory 
        ax.plot(past_coords[:, 0], past_coords[:, 1], color='blue',
                linewidth=2.5)

        ax.plot(pred_coords[:, 0], pred_coords[:, 1], color='red',
                linewidth=2.5, label='Predicted Trajectory')

        ax.scatter(past_coords[0, 0], past_coords[0, 1], color='green', marker='o', s=100, label='Start')
        ax.scatter(pred_coords[-1, 0], pred_coords[-1, 1], color='purple', marker='*', s=150, label='End')

        if coord_ranges:
            ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=-30, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Equator')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Bird Trajectory Prediction - Tag {tag_id}')
        ax.legend(loc='best')
        filename = os.path.join(trajectory_dir, f"trajectory_tag_{tag_id}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        
        st.success(f"Saved trajectory image for tag {tag_id} to {filename}")
    
    # Also create a combined plot for the streamlit interface
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (tag_id, pred_data) in enumerate(predictions.items()):
        past_coords = pred_data['orig_past_coords']
 
        if 'orig_predicted_coords' in pred_data:
            pred_coords = pred_data['orig_predicted_coords']
        elif coord_ranges:
            pred_coords = transform_coordinates_back(pred_data['predicted'][:, :2], coord_ranges)
        else:
            pred_coords = pred_data['predicted'][:, :2]

        ax.plot(past_coords[:, 0], past_coords[:, 1], color=colors[i], linestyle='--',
                linewidth=2, alpha=0.7)
        ax.plot(pred_coords[:, 0], pred_coords[:, 1], color=colors[i],
                linewidth=2.5, label=f'Tag {tag_id} - Predicted')
        ax.scatter(past_coords[0, 0], past_coords[0, 1], color=colors[i], marker='o', s=100)
        ax.scatter(pred_coords[-1, 0], pred_coords[-1, 1], color=colors[i], marker='*', s=150)

    if coord_ranges:
        ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Geographic Constraints')
        ax.axhline(y=-30, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Equator')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Combined Bird Trajectory Predictions')
    ax.legend(loc='best')
    
    # Save the combined plot 
    combined_filename = os.path.join(trajectory_dir, "combined_trajectories.png")
    plt.savefig(combined_filename, dpi=300)
    plt.close(fig)
    
    st.success(f"Saved combined trajectory image to {combined_filename}")
    st.info(f"All trajectory images have been saved to the '{trajectory_dir}' folder")


def load_model_safely(model_path, metadata_path=None):
    st.write(f"Attempting to load model from {model_path}...")
    
    # Initialize feature_cols to include all 19 features
    feature_cols = [
        'location-long', 'location-lat',
        'ECMWF Interim Full Daily Invariant Low Vegetation Cover',
        'ECMWF Interim Full Daily Invariant High Vegetation Cover',
        'day_of_year', 'hour_of_day', 'day_sin', 'day_cos',
        'hour_sin', 'hour_cos', 'time_diff_hours', 'distance_km',
        'speed_kmh', 'bearing', 'is_on_land', 'distance_to_equator',
        'northern_hemisphere', 'distance_to_poles', 'in_normal_lat_range'
    ]
    
    # First try to load metadata
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            st.success("Metadata loaded successfully")
            
            # Extract feature columns from metadata if available
            if 'feature_cols' in metadata:
                feature_cols = metadata['feature_cols']
                st.success(f"Feature columns loaded from metadata: {len(feature_cols)} features")
        except Exception as e:
            st.warning(f"Could not load metadata: {str(e)}")
    
    # Load feature scaler with feature columns
    scaler = None
    try:
        if os.path.exists('feature_scaler.pkl'):
            scaler_data = joblib.load('feature_scaler.pkl')
            if isinstance(scaler_data, tuple) and len(scaler_data) == 2:
                scaler, loaded_feature_cols = scaler_data
                st.success(f"Feature scaler loaded with {len(loaded_feature_cols)} features")
                
                # Check if feature dimensions match
                if len(loaded_feature_cols) != len(feature_cols):
                    st.warning(f"Feature dimension mismatch. Loaded scaler expects {len(loaded_feature_cols)} features, but we need {len(feature_cols)} features.")
                    # Create a new scaler with the correct dimensions
                    scaler, feature_cols = create_or_update_scaler(feature_cols)
            else:
                st.warning("Feature scaler loaded but doesn't contain feature column information")
                # Create a new scaler with the correct dimensions
                scaler, feature_cols = create_or_update_scaler(feature_cols)
        else:
            st.warning("No feature scaler found. Creating a new one.")
            # Create a new scaler with the correct dimensions
            scaler, feature_cols = create_or_update_scaler(feature_cols)
    except Exception as e:
        st.warning(f"Error loading feature scaler: {e}")
        # Create a new scaler with the correct dimensions
        scaler, feature_cols = create_or_update_scaler(feature_cols)
    
    try:
        # Method 1: Standard loading
        st.write("Trying standard model loading...")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        st.success("Model loaded successfully with standard method")
        return model, feature_cols, scaler
    except Exception as e:
        st.warning(f"Standard loading failed: {str(e)}")
        
        

def apply_minibatch_discrimination(predictions):
    """Apply tag-specific adjustments to all predictions to ensure diversity"""
    adjusted_predictions = {}
    
    # First pass: calculate baseline trajectories
    for tag_id, pred_data in predictions.items():
        adjusted_predictions[tag_id] = pred_data
    
    # Second pass: apply diversity adjustments
    for tag_id, pred_data in adjusted_predictions.items():
        past_traj = pred_data['past']
        pred_traj = pred_data['predicted']
        
        # Extract past movement direction
        if past_traj.shape[0] > 1:
            past_direction = past_traj[-1, :2] - past_traj[-2, :2]
            # Normalize direction vector
            norm = np.linalg.norm(past_direction)
            if norm > 0:
                past_direction = past_direction / norm
                
            # Create a tag-specific diversity vector
            tag_id_numeric = int(str(tag_id).replace('tag_', '').replace('Tag_', '').replace('_', '')) % 10000
            diversity_vector = np.array([
                np.sin(tag_id_numeric * 0.1) * 0.05,  # Small amplitude for subtle diversity
                np.cos(tag_id_numeric * 0.1) * 0.05   # Different direction for different tags
            ])
            
            # Apply the diversity vector with decay
            for i in range(len(pred_traj)):
                decay = 0.9 ** i
                # Add both past direction influence and diversity
                pred_traj[i, :2] += past_direction * 0.1 * decay + diversity_vector * decay
            
            # Ensure the first point still matches the last observed point
            pred_traj[0, :2] = past_traj[-1, :2]
            
            # Update the prediction
            adjusted_predictions[tag_id]['predicted'] = pred_traj
    
    return adjusted_predictions

def identify_hotspots_kde(trajectory_df, num_hotspots=8, bandwidth=0.5):
    st.write("Identifying hotspots using KDE method...")

    # Extract coordinates
    coords = trajectory_df[['location-long', 'location-lat']].values

    # Apply KDE to get density estimates
    kde = gaussian_kde(coords.T, bw_method=bandwidth)
    density = kde(coords.T)

    # Create a DataFrame with coordinates and density
    points_df = pd.DataFrame({
        'longitude': coords[:, 0],
        'latitude': coords[:, 1],
        'density': density
    })

    # Sort by density and take top points as candidates
    candidates = points_df.sort_values('density', ascending=False).head(1000)

    # Use KMeans to find distinct hotspot centers
    if len(candidates) > num_hotspots:
        kmeans = KMeans(n_clusters=num_hotspots, random_state=42)
        kmeans.fit(candidates[['longitude', 'latitude']])

        # Get cluster centers
        centers = kmeans.cluster_centers_

        # Find the point with highest density in each cluster
        hotspots = []
        for i in range(num_hotspots):
            cluster_points = candidates[kmeans.labels_ == i]
            if len(cluster_points) > 0:
                # Get the point with highest density in this cluster
                best_point = cluster_points.sort_values('density', ascending=False).iloc[0]

                # Count unique tags in this cluster
                if 'tag-local-identifier' in trajectory_df.columns:
                    # Find all points close to this hotspot
                    nearby_mask = ((trajectory_df['location-long'] - best_point['longitude'])**2 +
                                  (trajectory_df['location-lat'] - best_point['latitude'])**2 < 0.5**2)
                    nearby_points = trajectory_df[nearby_mask]
                    unique_tags = nearby_points['tag-local-identifier'].nunique()
                else:
                    unique_tags = 0

                hotspots.append({
                    'cluster_id': i+1,
                    'centroid_long': best_point['longitude'],
                    'centroid_lat': best_point['latitude'],
                    'density': best_point['density'],
                })
    else:
        hotspots = []
        for i, row in candidates.iterrows():
            hotspots.append({
                'cluster_id': i+1,
                'centroid_long': row['longitude'],
                'centroid_lat': row['latitude'],
                'density': row['density'],
            })

    hotspots_df = pd.DataFrame(hotspots)

    # Add average prediction day if available
    if 'prediction_day' in trajectory_df.columns:
        for i, row in hotspots_df.iterrows():
            nearby_mask = ((trajectory_df['location-long'] - row['centroid_long'])**2 +
                          (trajectory_df['location-lat'] - row['centroid_lat'])**2 < 0.5**2)
            nearby_points = trajectory_df[nearby_mask]
            if len(nearby_points) > 0:
                hotspots_df.at[i, 'avg_prediction_day'] = nearby_points['prediction_day'].mean()
            else:
                hotspots_df.at[i, 'avg_prediction_day'] = 0

    st.success(f"Identified {len(hotspots_df)} hotspots")
    return hotspots_df

def visualize_hotspots(trajectory_df, hotspots_df, output_dir="hotspot_analysis"):
    st.write("Visualizing hotspots...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

    if 'tag-local-identifier' in trajectory_df.columns:
        for tag, group in trajectory_df.groupby('tag-local-identifier'):
            ax.plot(group['location-long'], group['location-lat'],
                   alpha=0.4, linewidth=0.8)
    else:
        # Just plot all points with connecting lines
        ax.plot(trajectory_df['location-long'], trajectory_df['location-lat'],
               alpha=0.3, linewidth=0.5, c='gray')

    # Check if hotspots were identified
    if len(hotspots_df) > 0:
        # Normalize density for marker size
        max_density = hotspots_df['density'].max()
        min_density = hotspots_df['density'].min()
        norm_density = (hotspots_df['density'] - min_density) / (max_density - min_density)

        # Plot hotspots as distinct points with size based on density
        sizes = 100 + 400 * norm_density

        ax.scatter(
            hotspots_df['centroid_long'],
            hotspots_df['centroid_lat'],
            s=sizes,
            c='red',
            marker='*',
            edgecolors='black',
            linewidths=1,
            alpha=0.9,
            zorder=10
        )

        # Add labels with hotspot IDs
        for i, row in hotspots_df.iterrows():
            ax.text(
                row['centroid_long'],
                row['centroid_lat'],
                f"H{row['cluster_id']}",
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                zorder=11
            )

        # Automatically determine appropriate boundaries
        # First, get the range of hotspots
        hotspot_long_min = hotspots_df['centroid_long'].min()
        hotspot_long_max = hotspots_df['centroid_long'].max()
        hotspot_lat_min = hotspots_df['centroid_lat'].min()
        hotspot_lat_max = hotspots_df['centroid_lat'].max()

        # Calculate the range
        long_range = hotspot_long_max - hotspot_long_min
        lat_range = hotspot_lat_max - hotspot_lat_min

        # Add padding based on the range (20% on each side)
        padding_long = long_range * 0.2
        padding_lat = lat_range * 0.2

        # Ensure minimum padding to avoid too tight plots
        min_padding = 0.5
        padding_long = max(padding_long, min_padding)
        padding_lat = max(padding_lat, min_padding)

        # Set axis limits with padding
        ax.set_xlim([
            hotspot_long_min - padding_long,
            hotspot_long_max + padding_long
        ])
        ax.set_ylim([
            hotspot_lat_min - padding_lat,
            hotspot_lat_max + padding_lat
        ])

        ax.set_title(f'Bird Migration Trajectories with {len(hotspots_df)} Identified Hotspots')
    else:
        ax.set_title('Bird Migration Trajectories - No Hotspots Identified')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    #savw the hotspot image also
    output_file = os.path.join(output_dir, "bird_hotspots_map.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    st.pyplot(fig)
    st.success(f"Visualization saved to {output_file}")

    return output_file

def create_hotspot_report(hotspots_df, output_dir="hotspot_analysis"):
    if len(hotspots_df) == 0:
        st.warning("No hotspots to report")
        return
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "identified_hotspots.csv")
    hotspots_df.to_csv(output_file, index=False)

    st.success(f"Hotspot report saved to {output_file}")
    st.subheader("Identified Hotspots")
    st.dataframe(hotspots_df)
    return output_file

def prepare_trajectory_data_for_hotspots(predictions):
    st.write("Preparing trajectory data for hotspot analysis...")
    all_points = []
    
    for tag_id, pred_data in predictions.items():
        if 'orig_predicted_coords' in pred_data:
            coords = pred_data['orig_predicted_coords']
        elif 'orig_past_coords' in pred_data:
            coords = np.vstack([pred_data['orig_past_coords'], 
                               transform_coordinates_back(pred_data['predicted'][:, :2], coord_ranges)])
        else:
            coords = pred_data['predicted'][:, :2]
            st.warning(f"Using normalized coordinates for tag {tag_id} - hotspot analysis may not be accurate")

        for i, point in enumerate(coords):
            all_points.append({
                'tag-local-identifier': tag_id,
                'location-long': point[0],
                'location-lat': point[1],
                'prediction_day': i,  
                'is_predicted': True  
            })

    trajectory_df = pd.DataFrame(all_points)

    output_file = "all_predictions.csv"
    trajectory_df.to_csv(output_file, index=False)
    st.success(f"Saved all trajectory points to {output_file}")
    
    return trajectory_df



def load_patrol_model(model_type, model_path):
    #Load a pre-trained patrol route planning model
    st.write(f"Loading {model_type} model from {model_path}...")
    
    try:
        if model_type == 'q_learning':
            # For Q-learning, load the Q-table from pickle file
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            st.success(f"Successfully loaded {model_type} model")
            return model_data
            
        elif model_type == 'dqn':
            # Create a model with the same architecture
            input_dim = 42
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(8)
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.Huber()
            )
            return model
        
        elif model_type == 'double_dqn':
            # For Double DQN, use standard loading
            custom_objects = {
                'huber_loss': tf.keras.losses.Huber()
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            st.success(f"Successfully loaded {model_type} model")
            return model
            
        else:
            st.error(f"Unknown model type: {model_type}")
            return None
            
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        st.warning("Could not load weights. Using initialized model.")
        
        return None




def create_dqn_model(input_dim, env_size=8):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(env_size) 
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber()
    )
    
    return model

def create_patrol_environment(hotspots_df, start_idx=0, max_steps=30):
    st.write("Creating patrol environment from hotspots...")

    n_hotspots = len(hotspots_df)
    distance_matrix = np.zeros((n_hotspots, n_hotspots))
    
    for i in range(n_hotspots):
        for j in range(n_hotspots):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                dist = np.sqrt(
                    (hotspots_df.iloc[i]['centroid_long'] - hotspots_df.iloc[j]['centroid_long'])**2 +
                    (hotspots_df.iloc[i]['centroid_lat'] - hotspots_df.iloc[j]['centroid_lat'])**2
                )
                distance_matrix[i, j] = dist

    env = {
        'hotspots': hotspots_df,
        'n_hotspots': n_hotspots,
        'distance_matrix': distance_matrix,
        'max_distance': np.max(distance_matrix),
        'normalized_distance_matrix': distance_matrix / np.max(distance_matrix) if np.max(distance_matrix) > 0 else distance_matrix,
        'start_idx': start_idx,
        'max_steps': max_steps
    }
    
    st.success(f"Created patrol environment with {n_hotspots} hotspots")
    return env

def generate_q_learning_route(model_data, env):
    st.write("Generating optimal patrol route using Q-learning...")
    q_table = model_data['q_table']

    current_state = env['start_idx']
    route = [current_state]
    visited = {current_state}
    total_distance = 0
    total_density = env['hotspots'].iloc[current_state]['density']
    total_reward = env['hotspots'].iloc[current_state]['density']  # Initialize reward with starting hotspot density

    for step in range(env['max_steps']):
        valid_actions = [i for i in range(env['n_hotspots']) if i not in visited]
        
        if not valid_actions:
            break
        valid_q_values = {action: q_table[current_state, action] for action in valid_actions}
        next_state = max(valid_q_values, key=valid_q_values.get)
        action_reward = valid_q_values[next_state]

        route.append(next_state)
        visited.add(next_state)

        distance = env['distance_matrix'][current_state, next_state]
        total_distance += distance

        hotspot_density = env['hotspots'].iloc[next_state]['density']
        total_density += hotspot_density

        step_reward = hotspot_density / (distance + 0.001)  
        total_reward += step_reward

        current_state = next_state
        
    route_details = []
    for i, hotspot_idx in enumerate(route):
        hotspot = env['hotspots'].iloc[hotspot_idx]

        # Initial reward is just the density
        if i > 0:
            prev_idx = route[i-1]
            distance = env['distance_matrix'][prev_idx, hotspot_idx]
            step_reward = hotspot['density'] / (distance + 0.001)
        else:
            distance = 0
            step_reward = hotspot['density'] 
            
        route_details.append({
            'step': i,
            'hotspot_id': hotspot_idx,
            'longitude': hotspot['centroid_long'],
            'latitude': hotspot['centroid_lat'],
            'density': hotspot['density'],
            'distance_from_prev': distance,
            'step_reward': step_reward
        })
    
    route_df = pd.DataFrame(route_details)
    route_df['cumulative_distance'] = route_df['distance_from_prev'].cumsum()
    route_df['cumulative_reward'] = route_df['step_reward'].cumsum()

    efficiency = total_density / total_distance if total_distance > 0 else 0

    route_order = "→".join([str(idx) for idx in route])

    result = {
        'route': route,
        'route_df': route_df,
        'route_order': route_order,
        'total_distance': total_distance,
        'total_density': total_density,
        'total_reward': total_reward,
        'hotspots_visited': len(route),
        'coverage_percentage': len(route) / env['n_hotspots'] * 100,
        'efficiency': efficiency
    }
    
    st.success(f"Generated Q-learning route visiting {len(route)} hotspots with total distance {total_distance:.4f}")
    st.info(f"Route order: {route_order}")
    return result


def generate_dqn_route(model, env, is_double_dqn=False):
    model_type = "Double DQN" if is_double_dqn else "DQN"
    st.write(f"Generating optimal patrol route using {model_type}...")
    current_state = env['start_idx']
    route = [current_state]
    visited = {current_state}
    total_distance = 0
    total_density = env['hotspots'].iloc[current_state]['density']
    total_reward = env['hotspots'].iloc[current_state]['density']  
    expected_input_shape = model.input_shape
    expected_features = expected_input_shape[1] if expected_input_shape and len(expected_input_shape) > 1 else 42
    
    st.info(f"Model expects input shape with {expected_features} features")

    for step in range(env['max_steps']):
        valid_actions = [i for i in range(env['n_hotspots']) if i not in visited]
        
        if not valid_actions:
            break

        if expected_features == 42:  
            state_vector = np.zeros(expected_features)

            if current_state < env['n_hotspots']:
                state_vector[current_state] = 1
            for v in visited:
                if v < env['n_hotspots'] and env['n_hotspots'] + v < expected_features:
                    state_vector[env['n_hotspots'] + v] = 1

            if expected_features - 1 < expected_features:
                state_vector[-1] = (env['max_steps'] - step) / env['max_steps']
        else:
            state_vector = np.zeros(expected_features)

            # 1. One-hot encode current position (scaled to fit)
            position_section = min(env['n_hotspots'], expected_features // 3)
            if current_state < position_section:
                state_vector[current_state] = 1
                
            # 2. Encode visited status in the middle section
            visited_section_start = position_section
            visited_section_end = min(2 * position_section, expected_features)
            for v in visited:
                if v < position_section:
                    idx = visited_section_start + v
                    if idx < visited_section_end:
                        state_vector[idx] = 1
                        
            # 3. Add remaining steps and other info in the last section
            if expected_features - 1 < expected_features:
                state_vector[-1] = (env['max_steps'] - step) / env['max_steps']
        
        # Reshape for model input
        state_input = np.reshape(state_vector, [1, -1])
        
        try:
            q_values = model.predict(state_input, verbose=0)[0]

            masked_q_values = q_values.copy()
            for i in range(min(env['n_hotspots'], len(masked_q_values))):
                if i not in valid_actions:
                    masked_q_values[i] = -1e9

            next_state = np.argmax(masked_q_values[:env['n_hotspots']])

            action_reward = masked_q_values[next_state]
            
        except Exception as e:
            st.warning(f"Error during prediction: {str(e)}. Using random action.")
            next_state = random.choice(valid_actions)
            action_reward = 0
            
        route.append(next_state)
        visited.add(next_state)

        distance = env['distance_matrix'][current_state, next_state]
        total_distance += distance

        hotspot_density = env['hotspots'].iloc[next_state]['density']
        total_density += hotspot_density

        step_reward = hotspot_density / (distance + 0.001)  
        total_reward += step_reward
        current_state = next_state

    route_details = []
    for i, hotspot_idx in enumerate(route):
        hotspot = env['hotspots'].iloc[hotspot_idx]
        
        if i > 0:
            prev_idx = route[i-1]
            distance = env['distance_matrix'][prev_idx, hotspot_idx]
            # Calculate reward for this step
            step_reward = hotspot['density'] / (distance + 0.001)
        else:
            distance = 0
            step_reward = hotspot['density']  # Initial reward is just the density
            
        route_details.append({
            'step': i,
            'hotspot_id': hotspot_idx,
            'longitude': hotspot['centroid_long'],
            'latitude': hotspot['centroid_lat'],
            'density': hotspot['density'],
            'distance_from_prev': distance,
            'step_reward': step_reward
        })
    
    route_df = pd.DataFrame(route_details)
    route_df['cumulative_distance'] = route_df['distance_from_prev'].cumsum()
    route_df['cumulative_reward'] = route_df['step_reward'].cumsum()

    efficiency = total_density / total_distance if total_distance > 0 else 0
    route_order = "→".join([str(idx) for idx in route])
    result = {
        'route': route,
        'route_df': route_df,
        'route_order': route_order,
        'total_distance': total_distance,
        'total_density': total_density,
        'total_reward': total_reward,
        'hotspots_visited': len(route),
        'coverage_percentage': len(set(route)) / env['n_hotspots'] * 100,
        'efficiency': efficiency
    }
    
    st.success(f"Generated {model_type} route visiting {len(route)} hotspots with total distance {total_distance:.4f}")
    st.info(f"Route order: {route_order}")
    return result

def visualize_patrol_route(route_result, env, algorithm_name, output_dir="patrol_route_analysis"):
    st.write(f"Visualizing {algorithm_name} patrol route...")

    os.makedirs(output_dir, exist_ok=True)
    
    # Extract route and hotspots
    route = route_result['route']
    hotspots_df = env['hotspots']

    st.subheader(f"Route Order: {route_result['route_order']}")
    st.subheader("Route Details")
    st.dataframe(route_result['route_df'])
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot hotspots
    scatter = ax.scatter(
        hotspots_df['centroid_long'],
        hotspots_df['centroid_lat'],
        s=200,
        c=hotspots_df['density'],
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidths=1,
        zorder=10
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hotspot Density')
    route_coords = []
    for idx in route:
        route_coords.append((
            hotspots_df.iloc[idx]['centroid_long'],
            hotspots_df.iloc[idx]['centroid_lat']
        ))
    route_coords = np.array(route_coords)

    ax.plot(
        route_coords[:, 0],
        route_coords[:, 1],
        'r-',
        linewidth=2,
        alpha=0.7,
        zorder=5
    )

    ax.plot(
        route_coords[0, 0],
        route_coords[0, 1],
        'go',
        markersize=12,
        label='Start',
        zorder=15
    )
    ax.plot(
        route_coords[-1, 0],
        route_coords[-1, 1],
        'ro',
        markersize=12,
        label='End',
        zorder=15
    )

    for i in range(len(route_coords) - 1):
        x1, y1 = route_coords[i]
        x2, y2 = route_coords[i + 1]
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color='blue',
                lw=1.5,
                alpha=0.7
            ),
            zorder=20
        )

    for i, row in hotspots_df.iterrows():
        if i in route:
            step_num = route.index(i)
            label = f"H{i}\n(Step {step_num})"
        else:
            label = f"H{i}"
            
        ax.annotate(
            label,
            (row['centroid_long'], row['centroid_lat']),
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            zorder=25
        )

    ax.set_title(f'Optimal Patrol Route ({algorithm_name})\nRoute: {route_result["route_order"]}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f"{algorithm_name.lower().replace(' ', '_')}_route.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Save route details
    route_details_file = os.path.join(output_dir, f"{algorithm_name.lower().replace(' ', '_')}_route.csv")
    route_result['route_df'].to_csv(route_details_file, index=False)
    
    st.success(f"Visualization saved to {output_file}")
    return output_file



# Main application code
def main():
    st.title("🦅 Bird Trajectory Prediction")
    st.write("Predict bird migration trajectories using GAN technology")

    # File upload section
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload bird tracking data (CSV format)", type=["csv"])

    if uploaded_file is not None:
        # Load and display the data
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded'] = True
        st.success(f"Data uploaded successfully! {len(df)} records found.")

        # Display a sample of the data
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Adapt the dataset
        if st.button("Adapt Dataset Format"):
            adapted_df = adapt_dataset(df, "adapted_data.csv")
            st.session_state['adapted'] = True
            st.session_state['adapted_df'] = adapted_df

        # Preprocess the data
        if 'adapted' in st.session_state and st.session_state['adapted']:
            if st.button("Preprocess Data for GAN"):
                preprocessed_df = preprocess_for_gan(st.session_state['adapted_df'], "preprocessed_data.csv")
                if preprocessed_df is not None:
                    st.session_state['preprocessed'] = True
                    st.session_state['preprocessed_df'] = preprocessed_df

        # Model loading and prediction section
        if 'preprocessed' in st.session_state and st.session_state['preprocessed']:
            st.header("Generate Trajectory Predictions")

            # Model file upload
            st.subheader("Upload Model Files")
            model_files = {}
            model_files['generator'] = st.file_uploader("Upload Generator Model (.h5)", type=["h5"])
            model_files['metadata'] = st.file_uploader("Upload Model Metadata (.pkl)", type=["pkl"])

            if model_files['generator'] is not None:
                # Save the uploaded model file
                with open("uploaded_generator_model.h5", "wb") as f:
                    f.write(model_files['generator'].getbuffer())
                st.success("Generator model uploaded successfully!")

                if model_files['metadata'] is not None:
                    # Save the uploaded metadata file
                    with open("uploaded_metadata.pkl", "wb") as f:
                        f.write(model_files['metadata'].getbuffer())
                    st.success("Metadata uploaded successfully!")

                # Update model file paths
                model_files['generator'] = "uploaded_generator_model.h5"
                model_files['metadata'] = "uploaded_metadata.pkl" if model_files['metadata'] is not None else None

                # Generate predictions button
                if st.button("Generate Predictions"):
                    # Load the preprocessed data
                    processed_df = st.session_state['preprocessed_df']

                    # Generate predictions
                    predictions = predict_trajectories(processed_df, model_files)
                    
                    if predictions:
                        st.success("Predictions generated successfully!")
                        st.session_state['predictions'] = predictions
                
                # Add hotspot analysis section if predictions exist
                if 'predictions' in st.session_state:
                    st.header("Hotspot Analysis")
                    
                    # Allow user to specify number of hotspots
                    num_hotspots = st.slider("Number of hotspots to identify", min_value=3, max_value=20, value=8)
                    
                    if st.button("Identify Hotspots"):
                        # Prepare trajectory data
                        trajectory_df = prepare_trajectory_data_for_hotspots(st.session_state['predictions'])
                        
                        # Create hotspot analysis directory
                        hotspot_dir = "hotspot_analysis"
                        os.makedirs(hotspot_dir, exist_ok=True)
                        
                        # Identify hotspots
                        hotspots_df = identify_hotspots_kde(trajectory_df, num_hotspots=num_hotspots)
                        
                        # Visualize hotspots
                        visualize_hotspots(trajectory_df, hotspots_df, output_dir=hotspot_dir)
                        
                        # Create hotspot report
                        create_hotspot_report(hotspots_df, output_dir=hotspot_dir)
                        
                        # Store hotspots in session state
                        st.session_state['hotspots_df'] = hotspots_df
                        
                    
                    # Add patrol route planning section if hotspots exist
                    if 'hotspots_df' in st.session_state:
                        st.header("Patrol Route Planning")
                        
                        # Load hotspots
                        hotspots_df = st.session_state['hotspots_df']
                        
                        # Display hotspots
                        st.subheader("Identified Hotspots")
                        st.dataframe(hotspots_df)
                        
                        # Create patrol environment
                        env = create_patrol_environment(hotspots_df)
                        
                        # Allow user to select starting hotspot
                        start_idx = st.selectbox(
                            "Select starting hotspot",
                            options=list(range(len(hotspots_df))),
                            format_func=lambda x: f"Hotspot {x} (Density: {hotspots_df.iloc[x]['density']:.4f})"
                        )
                        
                        # Update environment with selected starting point
                        env['start_idx'] = start_idx
                        
                        # Allow user to set maximum steps
                        max_hotspots = len(hotspots_df)
                        max_steps_allowed = min(30, max_hotspots)
                        min_steps = max(1, min(3, max_hotspots))  # Ensures min_value is always < max_value

                        max_steps = st.slider(
                            "Maximum number of steps in route",
                            min_value=min_steps,
                            max_value=max_steps_allowed,
                            value=min(15, max_steps_allowed)
                        )
                        # Update environment with max steps
                        env['max_steps'] = max_steps
                        
                        # Create tabs for different algorithms
                        tabs = st.tabs(["Q-Learning", "DQN", "Double DQN"])
                        
                        with tabs[0]:
                            st.subheader("Q-Learning Patrol Route")
                            
                            # Check if model file exists in the directory
                            q_learning_model_path = "q_learning_model.h5"
                            if os.path.exists(q_learning_model_path):
                                st.success(f"Found Q-learning model at {q_learning_model_path}")
                                
                                # Load model
                                q_learning_model = load_patrol_model('q_learning', q_learning_model_path)
                                
                                if q_learning_model and st.button("Generate Q-Learning Route"):
                                    # Generate route
                                    q_learning_result = generate_q_learning_route(q_learning_model, env)
                                    
                                    # Visualize route
                                    visualize_patrol_route(q_learning_result, env, "Q-Learning")
                                    
                                    # Store result in session state
                                    st.session_state['q_learning_result'] = q_learning_result
                            else:
                                # Upload Q-learning model file
                                q_learning_file = st.file_uploader("Upload Q-learning model file (.pkl)", type=["pkl"])
                                
                                if q_learning_file is not None:
                                    # Save the uploaded model file
                                    with open("uploaded_q_learning_model.pkl", "wb") as f:
                                        f.write(q_learning_file.getbuffer())
                                    
                                    st.success("Q-learning model uploaded successfully!")
                                    
                                    # Load model
                                    q_learning_model = load_patrol_model('q_learning', "uploaded_q_learning_model.pkl")
                                    
                                    if q_learning_model and st.button("Generate Q-Learning Route"):
                                        # Generate route
                                        q_learning_result = generate_q_learning_route(q_learning_model, env)
                                        
                                        # Visualize route
                                        visualize_patrol_route(q_learning_result, env, "Q-Learning")
                                        
                                        # Store result in session state
                                        st.session_state['q_learning_result'] = q_learning_result
                        
                        with tabs[1]:
                            st.subheader("DQN Patrol Route")
                            
                            # Check if model file exists in the directory
                            dqn_model_path = "dqn_model.h5"
                            if os.path.exists(dqn_model_path):
                                st.success(f"Found DQN model at {dqn_model_path}")
                                
                                # Load model
                                dqn_model = load_patrol_model('dqn', dqn_model_path)
                                
                                if dqn_model and st.button("Generate DQN Route"):
                                    # Generate route
                                    dqn_result = generate_dqn_route(dqn_model, env, is_double_dqn=False)
                                    
                                    # Visualize route
                                    visualize_patrol_route(dqn_result, env, "DQN")
                                    
                                    # Store result in session state
                                    st.session_state['dqn_result'] = dqn_result
                            else:
                                # Upload DQN model file
                                dqn_file = st.file_uploader("Upload DQN model file (.h5)", type=["h5"])
                                
                                if dqn_file is not None:
                                    # Save the uploaded model file
                                    with open("uploaded_dqn_model.h5", "wb") as f:
                                        f.write(dqn_file.getbuffer())
                                    
                                    st.success("DQN model uploaded successfully!")
                                    
                                    # Load model
                                    dqn_model = load_patrol_model('dqn', "uploaded_dqn_model.h5")
                                    
                                    if dqn_model and st.button("Generate DQN Route"):
                                        # Generate route
                                        dqn_result = generate_dqn_route(dqn_model, env, is_double_dqn=False)
                                        
                                        # Visualize route
                                        visualize_patrol_route(dqn_result, env, "DQN")
                                        
                                        # Store result in session state
                                        st.session_state['dqn_result'] = dqn_result
                        
                        with tabs[2]:
                            st.subheader("Double DQN Patrol Route")
                            
                            # Check if model file exists in the directory
                            double_dqn_model_path = "double_dqn_model.h5"
                            if os.path.exists(double_dqn_model_path):
                                st.success(f"Found Double DQN model at {double_dqn_model_path}")
                                
                                # Load model
                                double_dqn_model = load_patrol_model('double_dqn', double_dqn_model_path)
                                
                                if double_dqn_model and st.button("Generate Double DQN Route"):
                                    # Generate route
                                    double_dqn_result = generate_dqn_route(double_dqn_model, env, is_double_dqn=True)
                                    
                                    # Visualize route
                                    visualize_patrol_route(double_dqn_result, env, "Double DQN")
                                    
                                    # Store result in session state
                                    st.session_state['double_dqn_result'] = double_dqn_result
                            else:
                                # Upload Double DQN model file
                                double_dqn_file = st.file_uploader("Upload Double DQN model file (.h5)", type=["h5"])
                                
                                if double_dqn_file is not None:
                                    # Save the uploaded model file
                                    with open("uploaded_double_dqn_model.h5", "wb") as f:
                                        f.write(double_dqn_file.getbuffer())
                                    
                                    st.success("Double DQN model uploaded successfully!")
                                    
                                    # Load model
                                    double_dqn_model = load_patrol_model('double_dqn', "uploaded_double_dqn_model.h5")
                                    
                                    if double_dqn_model and st.button("Generate Double DQN Route"):
                                        # Generate route
                                        double_dqn_result = generate_dqn_route(double_dqn_model, env, is_double_dqn=True)
                                        
                                        # Visualize route
                                        visualize_patrol_route(double_dqn_result, env, "Double DQN")
                                        
                                        # Store result in session state
                                        st.session_state['double_dqn_result'] = double_dqn_result
                        
                        results_available = []
                        if 'q_learning_result' in st.session_state:
                            results_available.append(('Q-Learning', st.session_state['q_learning_result']))
                        if 'dqn_result' in st.session_state:
                            results_available.append(('DQN', st.session_state['dqn_result']))
                        if 'double_dqn_result' in st.session_state:
                            results_available.append(('Double DQN', st.session_state['double_dqn_result']))

                        if len(results_available) > 1:
                            st.subheader("Algorithm Comparison")
                            
                            comparison_data = []
                            for name, result in results_available:
                                data_entry = {
                                    'Algorithm': name,
                                    'Hotspots Visited': result['hotspots_visited'],
                                    'Coverage (%)': result['coverage_percentage'],
                                    'Total Distance': result['total_distance'],
                                    'Total Density': result['total_density'],
                                    'Efficiency': result['efficiency']
                                }
                                
                                # Add route order if available
                                if 'route_order' in result:
                                    data_entry['Route Order'] = result['route_order']
                                else:
                                    # Create route order string from route array
                                    data_entry['Route Order'] = "→".join([str(idx) for idx in result['route']])
                                
                                # Add total reward if available
                                if 'total_reward' in result:
                                    data_entry['Total Reward'] = result['total_reward']
                                else:
                                    # Calculate a basic reward if not available
                                    data_entry['Total Reward'] = result['total_density'] / (result['total_distance'] + 0.001)
                                
                                comparison_data.append(data_entry)

                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df)
                            
                            # Create comparison chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                        
                            
            else:
                st.info("Please upload a generator model file to make predictions.")

    # Add information about the app
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a Generative Adversarial Network (GAN) to predict bird migration trajectories "
        "based on past movement patterns. Upload your bird tracking data, preprocess it, and generate "
        "predictions using the trained GAN model. Then identify migration hotspots and plan optimal "
        "patrol routes using reinforcement learning algorithms."
    )

    # Add instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        "1. Upload CSV data with bird tracking information\n"
        "2. Adapt the dataset to the required format\n"
        "3. Preprocess the data for the GAN model\n"
        "4. Upload the trained GAN generator model\n"
        "5. Generate and visualize trajectory predictions\n"
        "6. Identify migration hotspots\n"
        "7. Plan optimal patrol routes using RL algorithms"
    )

# Run the app
if __name__ == "__main__":
    main()
