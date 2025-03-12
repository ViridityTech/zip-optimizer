import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import math
import io
import zipfile
import streamlit.components.v1 as components

#############################################
# CONSTANT: Allowed Clinics (Only the specified locations)
#############################################
ALLOWED_CLINICS = [
    "Ankeny",
    "Beloit",
    "Bettendorf",
    "Boise",
    "Chicago",
    "Coeur d'Alene",
    "Crystal Lake",
    "Eau Claire",
    "Elgin",
    "Fond du Lac",
    "Geneva",
    "Iowa City",
    "Lake Geneva",
    "Meridian",
    "Moorhead",
    "Nampa",
    "Rolling Meadows",
    "Spokane",
    "Urbandale",
    "Warrenville",
    "Weldon Spring",
    "West Madison"
]

#############################################
# HELPER FUNCTIONS
#############################################

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great circle distance (in miles) between two arrays of lat/lon.
    Returns an array of shape (num_clinics, num_zipcodes).
    """
    R = 3958.8  # Earth radius in miles
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2[:, np.newaxis] - lon1
    dlat = lat2[:, np.newaxis] - lat1
    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1) * np.cos(lat2[:, np.newaxis]) *
         np.sin(dlon / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def min_max_normalize(series: pd.Series) -> pd.Series:
    """Safely apply min-max normalization to a numeric Series, returning values in [0,1]."""
    if series.empty:
        return pd.Series([0.0]*len(series), index=series.index)
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return pd.Series([1.0]*len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)

def compute_metrics(df, geo_distance_w, demo_population_w, demo_bachelors_w,
                    demo_graduate_w, demo_poverty_w, demo_income_w):
    """
    For each ZIP code assignment, compute:
      - distance_score: normalized so that a smaller distance yields a higher score.
      - geo_score = geo_distance_w * distance_score.
      - For each demographic factor, compute a normalized value.
      - demo_score: weighted sum of normalized demographic values.
    """
    max_dist = df['Distance to Clinic'].max()
    df['distance_score'] = (1 - (df['Distance to Clinic'] / max_dist)) if max_dist > 0 else 1.0
    df['geo_score'] = geo_distance_w * df['distance_score']

    if 'population_count' in df.columns:
        df['pop_norm'] = min_max_normalize(df['population_count'])
    else:
        df['pop_norm'] = 0.0
    if 'percent_bachelors_degree' in df.columns:
        df['bach_norm'] = min_max_normalize(df['percent_bachelors_degree'])
    else:
        df['bach_norm'] = 0.0
    if 'percent_graduate_degree' in df.columns:
        df['grad_norm'] = min_max_normalize(df['percent_graduate_degree'])
    else:
        df['grad_norm'] = 0.0
    if 'percent_population_in_poverty' in df.columns:
        df['pov_norm'] = min_max_normalize(df['percent_population_in_poverty'])
    else:
        df['pov_norm'] = 0.0
    if 'median_household_income' in df.columns:
        df['inc_norm'] = min_max_normalize(df['median_household_income'])
    else:
        df['inc_norm'] = 0.0

    df['demo_score'] = (demo_population_w * df['pop_norm'] +
                        demo_bachelors_w * df['bach_norm'] +
                        demo_graduate_w * df['grad_norm'] +
                        demo_poverty_w * df['pov_norm'] +
                        demo_income_w * df['inc_norm'])
    return df

def get_clinic_types():
    return {
        'Ankeny': 'Rural',
        'Beloit': 'Rural',
        'Bettendorf': 'Rural',
        'Boise': 'Rural',
        "Coeur d'Alene": 'Rural',
        'Cedar Rapids': 'Rural',
        'Chicago': 'Urban',
        'Crystal Lake': 'Urban',
        'Davenport': 'Rural',
        'Dubuque': 'Rural',
        'Dyer': 'Rural',
        'Eau Claire': 'Rural',
        'Elgin': 'Urban',
        'Fond du Lac': 'Rural',
        'Geneva': 'Rural',
        'Iowa City': 'Rural',
        'Lake Geneva': 'Rural',
        'Meridian': 'Rural',
        'Moorhead': 'Rural',
        'Munster': 'Urban',
        'Nampa': 'Rural',
        'Rockford': 'Rural',
        'Rolling Meadows': 'Urban',
        'Spokane': 'Rural',
        'Urbandale': 'Urban',
        'Warrenville': 'Urban',
        'Weldon Spring': 'Rural',
        'West Madison': 'Rural',
        'Anoka': 'Rural',
        'Duluth': 'Rural',
        'Lakeville': 'Rural',
        'Mankato': 'Rural',
        'Plymouth': 'Rural',
        'Rochester': 'Rural',
        'Shakopee': 'Rural',
        'White Bear Lake': 'Rural',
        'Woodbury': 'Rural',
        'Kennewick': 'Rural',
        'Tacoma': 'Urban',
        'Appleton': 'Rural',
        'Bellevue': 'Rural',
        'Brillion': 'Rural',
        'Brookfield': 'Urban',
        'De Pere': 'Urban',
        'E. Madison': 'Rural',
        'Franklin': 'Urban',
        'Green Bay CCD': 'Urban',
        'Howard': 'Rural',
        'Janesville': 'Rural',
        'Kenosha': 'Rural',
        'Kimberly': 'Urban',
        'La Crosse': 'Rural',
        'Menomonee': 'Rural',
        'Mequon': 'Rural',
        'Mitchell St.': 'Urban',
        'Pewaukee': 'Rural',
        'Sheboygan': 'Rural',
        'Wausau': 'Rural',
        'Chesterfield': 'Rural',
        'St.Cloud': 'Rural',
        'Oakville': 'Rural',
        'Shawnee': 'Rural'
    }

#############################################
# MAIN APP
#############################################

def main():
    st.title("Caravel Clinic ZIP Code Optimization")
    st.markdown("""
    This application helps optimize the assignment of ZIP codes to Caravel clinics based on:
    1. Geographic proximity (distance from ZIP code to clinic)
    2. Demographic factors (population, education, income, poverty)
    3. Salesforce lead data (from the last 6 months)
    """)

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a tool:", 
                               ["Optimizer: Generate Optimized CSV", 
                                "Visualizer: Generate Maps"])

    if app_mode == "Optimizer: Generate Optimized CSV":
        st.header("Optimizer: Clinic ZIP Code Optimization & CSV Generation")
        st.markdown("""
        **Workflow:**
        1. Load clinic and ZIP code data.
        2. Configure weights and service parameters.
        3. Assign ZIP codes to clinics and compute geographic & demographic metrics.
        4. Query Salesforce for lead counts from the last 6 months (per zipcode).
        5. Merge the Salesforce data (lead count) with the assignments.
        6. Compute the final combined score as a weighted sum of Geographic, Demographic, and Salesforce components.
        7. The output is an **Optimized Assignments** CSV (with all raw and normalized metrics) for use in the Visualizer.
        """)
        st.info("Please run the standalone Optimizer script for this functionality.")

    elif app_mode == "Visualizer: Generate Maps":
        st.header("Visualizer: Generate Maps from Optimized Assignments")
        st.markdown("""
        **Workflow:**
        1. Upload the Optimized Assignments CSV (from the Optimizer).
        2. Generate interactive maps showing:
           - All clinics with their service areas
           - Individual clinic maps with their assigned ZIP codes
           - Optimized vs. non-optimized ZIP codes
        3. Download the maps as HTML files for offline viewing.
        """)
        st.info("Please run the standalone Visualizer script for this functionality.")

    # Display information about the application
    st.sidebar.header("About")
    st.sidebar.info("""
    This application was developed for Caravel Autism Health to optimize the assignment of ZIP codes to clinics.
    
    The optimization takes into account:
    - Geographic proximity
    - Demographic factors
    - Salesforce lead data
    """)

def generate_map(clinics_df, zipcodes_df, rural_radius, urban_radius, selected_clinics):
    """
    Generate a map showing all clinics and their service areas.
    
    Parameters:
    - clinics_df: DataFrame with clinic information (latitude, longitude, Type, Clinic)
    - zipcodes_df: DataFrame with ZIP code information (population_center_latitude, population_center_longitude, zip)
    - rural_radius: Service radius for rural clinics (in miles)
    - urban_radius: Service radius for urban clinics (in miles)
    - selected_clinics: List of clinic names to include in the map
    
    Returns:
    - HTML string of the generated map
    """
    # Filter clinics
    clinics_df = clinics_df[clinics_df['Clinic'].isin(selected_clinics)]
    
    # Calculate distances between clinics and ZIP codes
    zip_lat = zipcodes_df['population_center_latitude'].values
    zip_lon = zipcodes_df['population_center_longitude'].values
    clinic_lat = clinics_df['latitude'].values
    clinic_lon = clinics_df['longitude'].values
    clinic_names = clinics_df['Clinic'].values
    
    # Apply different radius based on clinic type (rural/urban)
    clinic_types = get_clinic_types()
    clinic_radii = np.array([
        rural_radius if clinic_types.get(name, 'Rural') == 'Rural' else urban_radius 
        for name in clinic_names
    ])
    
    distances = haversine_distance(zip_lat, zip_lon, clinic_lat, clinic_lon)
    
    # Determine which ZIP codes are within the radius of each clinic
    within_radius = distances <= clinic_radii[:, np.newaxis]
    zip_in_any = np.any(within_radius, axis=0)
    indices = np.where(zip_in_any)[0]
    zipcodes_within_df = zipcodes_df.iloc[indices].copy()
    distances_within = distances[:, indices]
    mask_within = within_radius[:, indices]
    masked_distances = np.where(mask_within, distances_within, np.inf)
    closest_indices = np.argmin(masked_distances, axis=0)
    assigned_clinics = clinic_names[closest_indices]
    min_distances = masked_distances[closest_indices, np.arange(len(indices))]
    zipcodes_within_df['Assigned Clinic'] = assigned_clinics
    zipcodes_within_df['Distance to Clinic'] = min_distances
    
    # Create a map centered on the average of all clinic locations
    map_center = [clinics_df['latitude'].mean(), clinics_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=6, tiles="cartodbpositron")
    
    # Add markers for each clinic
    for _, clinic in clinics_df.iterrows():
        clinic_type = clinic_types.get(clinic['Clinic'], 'Rural')
        radius = urban_radius if clinic_type == 'Urban' else rural_radius
        folium.Marker(
            location=[clinic['latitude'], clinic['longitude']],
            popup=f"{clinic['Clinic']} ({clinic_type})<br>Radius: {radius} miles",
            icon=folium.Icon(color='blue', icon='hospital', prefix='fa')
        ).add_to(m)
        
        # Add a circle showing the service radius
        folium.Circle(
            location=[clinic['latitude'], clinic['longitude']],
            radius=radius * 1609.34,  # Convert miles to meters
            color='blue',
            fill=True,
            fill_opacity=0.1,
            popup=f"{clinic['Clinic']} Service Area ({radius} miles)"
        ).add_to(m)
    
    # Convert the map to HTML
    html_string = m._repr_html_()
    return html_string

def generate_clinic_map(clinic_name, clinics_df, zip_geo, assignments_df):
    """
    Generate a map for a specific clinic showing its assigned ZIP codes.
    
    Parameters:
    - clinic_name: Name of the clinic
    - clinics_df: DataFrame with clinic information
    - zip_geo: GeoDataFrame with ZIP code boundaries
    - assignments_df: DataFrame with ZIP code assignments
    
    Returns:
    - HTML string of the generated map
    """
    # Filter data for the specific clinic
    clinic_row = clinics_df[clinics_df['Clinic'] == clinic_name].iloc[0]
    clinic_zips = assignments_df[assignments_df['Assigned Clinic'] == clinic_name]
    
    # Merge with geographic data
    clinic_zips['zip'] = clinic_zips['zip'].astype(str)
    zip_geo['ZCTA5CE10'] = zip_geo['ZCTA5CE10'].astype(str)
    within_df = zip_geo.merge(clinic_zips, left_on='ZCTA5CE10', right_on='zip', how='inner')
    
    # Create map centered on the clinic
    m = folium.Map(location=[clinic_row['latitude'], clinic_row['longitude']], 
                   zoom_start=8, tiles="cartodbpositron")
    
    # Add marker for the clinic
    folium.Marker(
        location=[clinic_row['latitude'], clinic_row['longitude']],
        popup=clinic_name,
        icon=folium.Icon(color='blue', icon='hospital', prefix='fa')
    ).add_to(m)
    
    # Add ZIP code polygons
    folium.GeoJson(
        data=within_df[['zip', 'Assigned Clinic', 'Optimized', 'combined_score', 'geometry']].to_json(),
        style_function=lambda x: {
            'fillColor': 'green' if x['properties']['Optimized'] else 'red',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=['zip', 'Assigned Clinic', 'Optimized', 'combined_score'])
    ).add_to(m)
    
    # Convert the map to HTML
    html_string = m._repr_html_()
    return html_string

if __name__ == "__main__":
    main() 