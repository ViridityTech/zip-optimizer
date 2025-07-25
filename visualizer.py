import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import io
import zipfile

#############################################
# HELPER FUNCTION FOR MAP GENERATION
#############################################

def get_feature_color(score, optimized: bool) -> str:
    """
    Return a fixed color based on optimization:
      - Green (#008000) if optimized.
      - Red (#800000) if not optimized.
    """
    return "#008000" if optimized else "#800000"

def generate_clinic_map_html(clinic: str, clinic_lat: float, clinic_lon: float,
                             clinic_geo: gpd.GeoDataFrame) -> str:
    """
    Generate a Folium map for a clinic.
    All ZIP code polygons from the uploaded CSV (merged with the GeoJSON)
    are used—distance is not a filter here.
    Each ZIP's tooltip displays only the CSV fields.
    """
    # Define the desired tooltip fields (adjusted to match the Tool 1 output)
    tooltip_fields = [
        'zip', 'Assigned Clinic', 'population_count', 'pop_norm',
        'percent_bachelors_degree', 'bach_norm',
        'percent_graduate_degree', 'grad_norm',
        'percent_population_in_poverty', 'pov_norm',
        'median_household_income', 'inc_norm',
        'demo_score', 'lead_count', 'sf_leads_norm', 'final_combined_score', 'Optimized'
    ]
    # Subset the GeoDataFrame to only these columns plus geometry.
    map_df = clinic_geo[tooltip_fields + ['geometry']].copy()
    if map_df.empty:
        return ""
    m = folium.Map(location=[clinic_lat, clinic_lon], zoom_start=10)
    def style_function(feature):
        optimized_flag = feature['properties'].get('Optimized', False)
        # Use 'final_combined_score' instead of 'combined_score'
        fill_color = get_feature_color(feature['properties'].get('final_combined_score', 0.0), optimized_flag)
        return {
            'fillColor': fill_color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        }
    highlight_function = lambda feature: {'weight': 3, 'color': 'black', 'fillOpacity': 0.6}
    folium.GeoJson(
        data=map_df.to_json(),
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields)
    ).add_to(m)
    folium.Marker(
        location=[clinic_lat, clinic_lon],
        popup=clinic,
        icon=folium.Icon(color='blue')
    ).add_to(m)
    return m._repr_html_()

#############################################
# MAIN APP – TOOL 2
#############################################

def main():
    st.title("Tool 2: Map Generation from Optimized Assignments CSV")
    st.markdown("""
    **Workflow:**
    1. Upload the **Optimized Assignments CSV** (generated by Tool 1).
    2. The app loads ZIP code boundaries from the project file and merges them with the uploaded data.
    3. For each clinic, a Folium map is generated showing the assigned ZIP codes.
       Tooltips display all individual metric values.
    4. Download a ZIP file containing one HTML map per clinic.
    """)

    with st.sidebar:
        st.header("Upload Files")
        
        # File uploader for optimized assignments only
        uploaded_csv = st.file_uploader(
            "Upload Optimized Assignments CSV", 
            type=["csv"],
            help="CSV file generated by the Optimizer containing optimized ZIP code assignments"
        )
        
        st.subheader("ZIP Code Boundaries")
        st.info("Using: `usa_zip_codes_geo_100m.json` from project directory")
        
        st.header("Map Parameters")
        run_button = st.button("Generate Maps")

    # Check if CSV is uploaded
    if not uploaded_csv:
        st.warning("Please upload the optimized assignments CSV file to continue.")
        st.info("""
        **Required File:**
        
        **Optimized Assignments CSV** - Generated by the Optimizer tool
        
        **Note:** ZIP code boundaries are automatically loaded from the project file.
        """)
        if not run_button:
            st.info("Upload CSV and click **Generate Maps**.")
        return

    if not run_button:
        st.info("Upload CSV and click **Generate Maps**.")
        return

    # Load the optimized assignments CSV
    try:
        optimized_assignments = pd.read_csv(uploaded_csv, dtype={"zip": str})
        st.success("✓ Optimized assignments CSV loaded successfully.")
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        return

    # Load the GeoJSON file for ZIP code boundaries from project directory
    try:
        zip_geo = gpd.read_file('usa_zip_codes_geo_100m.json')
        st.success("✓ ZIP code boundaries loaded from project file.")
    except FileNotFoundError:
        st.error("ZIP code boundaries file not found: `usa_zip_codes_geo_100m.json`")
        st.info("Please ensure the file exists in the project root directory.")
        return
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {e}")
        return

    # Determine the ZIP code column from possible options
    possible_cols = ['ZCTA5CE10', 'GEOID10', 'ZIPCODE', 'postalCode', 'CODE', 'ZIP', 'ZCTA5CE']
    zip_code_col = None
    for c in possible_cols:
        if c in zip_geo.columns:
            zip_code_col = c
            break
    if not zip_code_col:
        st.error("No recognizable ZIP code column found in the GeoJSON file.")
        st.info(f"Available columns: {list(zip_geo.columns)}")
        st.info(f"Expected one of: {possible_cols}")
        return
    
    st.info(f"Using '{zip_code_col}' column for ZIP codes from GeoJSON file.")
    zip_geo['zip'] = zip_geo[zip_code_col].astype(str).str.zfill(5)

    # Merge the GeoJSON with the optimized assignments CSV
    zip_geo_assigned = zip_geo.merge(optimized_assignments, on='zip', how='left')

    st.success("✓ Mapping data prepared!")
    st.header("Generated Maps by Clinic")

    # Generate maps and create a ZIP file with one HTML per clinic.
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        index_lines = [
            "<html><head><meta charset='utf-8'><title>Clinic Maps Index</title></head><body>",
            "<h1>All Clinic Maps</h1><ul>"
        ]
        # Get list of unique clinics from the uploaded CSV.
        clinics = optimized_assignments['Assigned Clinic'].unique()
        for clinic in clinics:
            # For each clinic, filter the merged GeoDataFrame.
            clinic_geo = zip_geo_assigned[zip_geo_assigned['Assigned Clinic'] == clinic].copy()
            if clinic_geo.empty:
                continue
            # Use the first record's geometry centroid as the clinic location.
            centroid = clinic_geo.geometry.unary_union.centroid
            map_html = generate_clinic_map_html(clinic, centroid.y, centroid.x, clinic_geo)
            if not map_html:
                continue
            filename = f"{clinic}_map.html"
            zf.writestr(filename, map_html)
            index_lines.append(f"<li><a href='{filename}' target='_blank'>{clinic}</a></li>")
        index_lines.append("</ul></body></html>")
        zf.writestr("index.html", "\n".join(index_lines))
    
    st.download_button(
        "Download All Maps (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="all_clinic_maps.zip",
        mime="application/x-zip-compressed"
    )
    st.write("ZIP includes one HTML map per clinic plus an index.html to browse them.")

if __name__ == "__main__":
    main()
