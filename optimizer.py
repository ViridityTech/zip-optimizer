from simple_salesforce import Salesforce
import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import io
import zipfile
import json
import glob
from datetime import datetime
from common_utils import get_allowed_clinics, get_clinic_types  # Import the new functions

#############################################
# Salesforce Query Functions
#############################################

def get_salesforce_auth():
    """
    Get Salesforce authentication credentials from file or user input
    and return authenticated Salesforce instance.
    """
    # First try to load credentials from file
    credentials = load_credentials_from_file()
    
    if credentials:
        try:
            sf = Salesforce(
                username=credentials['username'],
                password=credentials['password'],
                security_token=credentials['security_token'],
                domain='login'
            )
            st.success(f"Successfully authenticated as user: {sf.user_id} using credentials file")
            return sf
        except Exception as e:
            st.warning(f"Authentication failed using credentials file: {str(e)}")
            st.info("Falling back to manual authentication")
            # Fall back to manual authentication
            
    # Create a form for secure credential input
    with st.form("salesforce_auth_form"):
        st.subheader("Salesforce Authentication")
        st.markdown("Please enter your Salesforce credentials:")
        
        username = st.text_input("Username", placeholder="example@caravelautism.com")
        password = st.text_input("Password", type="password")
        security_token = st.text_input("Security Token", type="password", 
                                      help="Your Salesforce security token. If you don't have one, you can reset it in Salesforce under Settings > My Personal Information > Reset Security Token.")
        
        submit_button = st.form_submit_button("Authenticate")
    
    if submit_button:
        if not username or not password:
            st.error("Username and password are required.")
            return None
        
        try:
            sf = Salesforce(
                username=username,
                password=password,
                security_token=security_token,
                domain='login'
            )
            st.success(f"Successfully authenticated as user: {sf.user_id}")
            return sf
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return None
    
    return None

def get_leads_last_6_months(sf):
    soql_query = """
    SELECT PostalCode, Name
    FROM Lead
    WHERE CreatedDate >= LAST_N_MONTHS:6 AND PostalCode != null
    """
    result = sf.query_all(soql_query)
    return result['records']

def get_valid_locations():
    """
    Return the list of allowed clinic locations.
    Uses the get_allowed_clinics() function from common_utils.
    """
    return get_allowed_clinics()

#############################################
# Helper Functions for Geographic & Demographic Metrics
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

#############################################
# MAIN APP – TOOL 1
#############################################

def save_settings(settings):
    """Save the current settings to a text file."""
    os.makedirs('settings', exist_ok=True)
    file_path = os.path.join('settings', 'optimizer_settings.json')
    with open(file_path, 'w') as f:
        json.dump(settings, f)
    return file_path

def load_settings():
    """Load settings from the settings file if it exists."""
    file_path = os.path.join('settings', 'optimizer_settings.json')
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                settings = json.load(f)
            return settings
        return None
    except Exception as e:
        st.error(f"Error loading settings: {e}")
        return None

def load_credentials_from_file(filepath="credentials.txt"):
    """
    Load Salesforce credentials from a credentials file.
    Returns a dictionary containing username, password, and security_token.
    """
    try:
        credentials = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        credentials[key.strip()] = value.strip().strip('"\'')
            
            # Verify all required credentials are present
            required_keys = ['username', 'password', 'security_token']
            if all(key in credentials for key in required_keys):
                return credentials
            else:
                missing = [key for key in required_keys if key not in credentials]
                st.warning(f"Missing credentials in file: {', '.join(missing)}")
                return None
        return None
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return None

def main():
    st.title("Clinic ZIP Code Optimizer & CSV Generation")
    st.markdown("""
    **Workflow:**
    1. Load clinic and ZIP code data.
    2. Configure weights and service parameters.
    3. Assign ZIP codes to clinics and compute geographic & demographic metrics.
    4. Query Salesforce for lead counts from the last 6 months (per zipcode).
    5. Merge the Salesforce data (lead count) with the assignments.
    6. Compute the final combined score as a weighted sum of Geographic, Demographic, and Salesforce components.
    7. The output is an **Optimized Assignments** CSV (with all raw and normalized metrics) for visualization.
    """)

    # Get Salesforce authentication first
    sf = None
    if 'sf_authenticated' not in st.session_state:
        st.session_state.sf_authenticated = False
    
    if not st.session_state.sf_authenticated:
        st.header("Step 1: Salesforce Authentication")
        sf = get_salesforce_auth()
        if sf is not None:
            st.session_state.sf = sf
            st.session_state.sf_authenticated = True
            st.success("Salesforce authentication successful! Please configure optimization parameters.")
        else:
            st.warning("Please authenticate with Salesforce to continue.")
            return
    else:
        sf = st.session_state.sf
        st.success("Using existing Salesforce authentication.")

    # Continue with the rest of the process
    st.header("Step 2: Configure Optimization Parameters")
    
    with st.sidebar:
        st.header("Configuration")
        
        # Load default values or from saved settings
        default_settings = {
            "w_geo": 0.33,
            "w_demo": 0.33,
            "w_sf": 0.34,
            "demo_population_w": 0.2,
            "demo_bachelors_w": 0.2,
            "demo_graduate_w": 0.0,
            "demo_poverty_w": 0.2,
            "demo_income_w": 0.4,
            "keep_percentage": 50,
            "rural_radius": 20.0,
            "urban_radius": 10.0,
            "selected_clinics": get_allowed_clinics()  # Use the new function directly
        }
        
        # Try to load saved settings
        saved_settings = load_settings()
        if saved_settings:
            st.success("Loaded saved settings.")
            settings = saved_settings
        else:
            settings = default_settings
        
        # Weight configuration
        st.subheader("Overall Weights (Must Sum to 1.0)")
        w_geo = st.number_input("Geographic Weight", value=settings["w_geo"], min_value=0.0, max_value=1.0, step=0.01)
        w_demo = st.number_input("Demographic Weight", value=settings["w_demo"], min_value=0.0, max_value=1.0, step=0.01)
        w_sf = st.number_input("Salesforce Weight", value=settings["w_sf"], min_value=0.0, max_value=1.0, step=0.01)
        if abs(w_geo + w_demo + w_sf - 1.0) > 1e-3:
            st.error("The three overall weights must sum to 1.0!")
            st.stop()

        st.subheader("Demographic Sub-Weights (Must Sum to 1.0)")
        demo_population_w = st.slider("Population Weight", 0.0, 1.0, settings["demo_population_w"], 0.05)
        demo_bachelors_w = st.slider("Bachelor's Degree Weight", 0.0, 1.0, settings["demo_bachelors_w"], 0.05)
        demo_graduate_w = st.slider("Graduate Degree Weight", 0.0, 1.0, settings["demo_graduate_w"], 0.05)
        demo_poverty_w = st.slider("Poverty Weight", 0.0, 1.0, settings["demo_poverty_w"], 0.05)
        demo_income_w = st.slider("Income Weight", 0.0, 1.0, settings["demo_income_w"], 0.05)
        demo_sum = demo_population_w + demo_bachelors_w + demo_graduate_w + demo_poverty_w + demo_income_w
        if abs(demo_sum - 1.0) > 1e-3:
            st.error("Demographic sub-weights must sum to 1.0!")
            st.stop()

        st.subheader("Other Parameters")
        keep_percentage = st.slider("Percentage of ZIP codes to optimize", 0, 100, settings["keep_percentage"], 1)
        
        # Add separate radius inputs for rural and urban clinics
        st.subheader("Clinic Service Radius")
        rural_radius = st.number_input("Rural Clinic Radius (miles)", 1.0, 1000.0, settings["rural_radius"], 1.0)
        urban_radius = st.number_input("Urban Clinic Radius (miles)", 1.0, 1000.0, settings["urban_radius"], 1.0)

        # Replace the multiselect with a dropdown and a button
        st.subheader("Clinic Selection")
        all_clinics = get_allowed_clinics()  # Use the new function directly
        clinic_selection_method = st.radio(
            "Selection Method",
            ["Select Single Clinic", "Select All Clinics"],
            index=0 if len(settings["selected_clinics"]) == 1 else 1
        )

        if clinic_selection_method == "Select Single Clinic":
            default_index = 0
            if len(settings["selected_clinics"]) == 1:
                if settings["selected_clinics"][0] in all_clinics:
                    default_index = all_clinics.index(settings["selected_clinics"][0])
            selected_clinics = [st.selectbox("Select Clinic", all_clinics, index=default_index)]
        else:
            selected_clinics = all_clinics
            st.success(f"All {len(all_clinics)} clinics will be processed.")
        
        # Simple save button
        save_button = st.button("Save Current Settings")
        
        if save_button:
            # Create settings dictionary
            current_settings = {
                "w_geo": w_geo,
                "w_demo": w_demo,
                "w_sf": w_sf,
                "demo_population_w": demo_population_w,
                "demo_bachelors_w": demo_bachelors_w,
                "demo_graduate_w": demo_graduate_w,
                "demo_poverty_w": demo_poverty_w,
                "demo_income_w": demo_income_w,
                "keep_percentage": keep_percentage,
                "rural_radius": rural_radius,
                "urban_radius": urban_radius,
                "selected_clinics": selected_clinics
            }
            
            # Save the settings
            file_path = save_settings(current_settings)
            st.success(f"Settings saved to {file_path}")

        run_button = st.button("Run Optimization")

    if not run_button:
        st.info("Adjust settings in the sidebar and click **Run Optimization**.")
        return

    st.info("Loading clinic and ZIP code data...")
    try:
        clinics_df = pd.read_csv("addresses_with_coordinates.csv")
        st.write("Clinic data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading clinic data: {e}")
        return

    try:
        zipcodes_df = pd.read_csv("Zipcodes Info csv.csv", low_memory=False)
        st.write("ZIP code data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading ZIP code data: {e}")
        return

    for col in ['population_center_latitude', 'population_center_longitude', 'zip']:
        if col not in zipcodes_df.columns:
            st.error(f"Missing column in ZIP code data: {col}")
            return
    for col in ['latitude', 'longitude', 'Type', 'Clinic']:
        if col not in clinics_df.columns:
            st.error(f"Missing column in clinic data: {col}")
            return

    zipcodes_df = zipcodes_df.dropna(subset=['population_center_latitude','population_center_longitude','zip'])
    clinics_df = clinics_df.dropna(subset=['latitude','longitude','Type','Clinic'])
    
    try:
        zipcodes_df['population_center_latitude'] = zipcodes_df['population_center_latitude'].astype(float)
        zipcodes_df['population_center_longitude'] = zipcodes_df['population_center_longitude'].astype(float)
        clinics_df['latitude'] = clinics_df['latitude'].astype(float)
        clinics_df['longitude'] = clinics_df['longitude'].astype(float)
    except Exception as e:
        st.error(f"Error converting coordinates: {e}")
        return

    st.write("Calculating distances between clinics and ZIP codes...")
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
    st.write("Distance calculation completed.")

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
    zipcodes_within_df['zip'] = zipcodes_within_df['zip'].astype(str).str.zfill(5)
    
    st.write("Assigning ZIP codes to the closest clinics within their radii...")
    st.dataframe(zipcodes_within_df.head())

    final_df = zipcodes_within_df[zipcodes_within_df['Assigned Clinic'].isin(selected_clinics)].copy()

    optimized_list = []
    for clinic in final_df['Assigned Clinic'].unique():
        clinic_df = final_df[final_df['Assigned Clinic'] == clinic].copy()
        clinic_df = compute_metrics(clinic_df, geo_distance_w=1.0,  # Only one geo factor so weight=1
                                    demo_population_w=demo_population_w,
                                    demo_bachelors_w=demo_bachelors_w,
                                    demo_graduate_w=demo_graduate_w,
                                    demo_poverty_w=demo_poverty_w,
                                    demo_income_w=demo_income_w)
        optimized_list.append(clinic_df)
    final_assignments = pd.concat(optimized_list).reset_index(drop=True)

    st.info("Querying Salesforce for lead counts from the last 6 months (per zipcode)...")
    try:
        sf_records = get_leads_last_6_months(sf)
        sf_df = pd.DataFrame(sf_records)
        if 'attributes' in sf_df.columns:
            sf_df = sf_df.drop(columns='attributes')
        
        sf_group = sf_df.groupby('PostalCode', as_index=False)['Name'].count()
        sf_group.rename(columns={'Name': 'lead_count'}, inplace=True)
        sf_group['zip'] = sf_group['PostalCode'].str.zfill(5)
        
        # Merge Salesforce data (lead_count) into final assignments.
        final_assignments = final_assignments.merge(sf_group[['zip', 'lead_count']], on='zip', how='left')
        final_assignments['lead_count'] = final_assignments['lead_count'].fillna(0)
        final_assignments['sf_leads_norm'] = min_max_normalize(final_assignments['lead_count'])
        
        # Compute final combined score.
        final_assignments['final_combined_score'] = (
            w_geo * final_assignments['geo_score'] +
            w_demo * final_assignments['demo_score'] +
            w_sf * final_assignments['sf_leads_norm']
        )

        final_assignments = final_assignments.sort_values('final_combined_score', ascending=False).reset_index(drop=True)

        # Fix the problem with optimized percentage calculation
        final_assignments['Optimized'] = False
        optimized_count_per_clinic = {}

        # Process each clinic separately to apply the keep percentage correctly
        for clinic in final_assignments['Assigned Clinic'].unique():
            clinic_mask = final_assignments['Assigned Clinic'] == clinic
            clinic_df = final_assignments[clinic_mask]
            
            # Calculate how many zipcodes to keep for this clinic
            clinic_keep_count = int(math.ceil(len(clinic_df) * (keep_percentage / 100)))
            
            # Sort this clinic's zipcodes by score and mark the top ones as optimized
            clinic_indices = clinic_df.sort_values('final_combined_score', ascending=False).index[:clinic_keep_count]
            final_assignments.loc[clinic_indices, 'Optimized'] = True
            
            # Store the count for reporting
            optimized_count_per_clinic[clinic] = clinic_keep_count

        # Report information about the optimization
        total_zipcodes = len(final_assignments)
        total_optimized = final_assignments['Optimized'].sum()
        st.info(f"Optimized {total_optimized} out of {total_zipcodes} ZIP codes ({total_optimized/total_zipcodes:.1%})")

        output_cols = [
            'zip', 'Assigned Clinic', 'Distance to Clinic', 'distance_score', 'geo_score',
            'population_count', 'pop_norm', 'percent_bachelors_degree', 'bach_norm',
            'percent_graduate_degree', 'grad_norm', 'percent_population_in_poverty', 'pov_norm',
            'median_household_income', 'inc_norm', 'demo_score',
            'lead_count', 'sf_leads_norm',
            'final_combined_score', 'Optimized'
        ]
        output_csv = final_assignments[output_cols].to_csv(index=False)

        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "optimized_assignments_with_tooltip.csv")
        with open(output_path, "w") as f:
            f.write(output_csv)

        st.write("Optimized assignments CSV generated and saved (in the 'data' folder).")
        st.download_button("Download Optimized Assignments CSV",
                           data=output_csv,
                           file_name="optimized_assignments_with_tooltip.csv",
                           mime="text/csv")

        st.header("Final Optimized Assignments")
        st.dataframe(final_assignments)
    except Exception as e:
        st.error(f"Error querying Salesforce: {e}")

if __name__ == "__main__":
    main()
