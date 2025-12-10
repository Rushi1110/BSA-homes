import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import re

# --- 0. AUTHENTICATION CONFIG ---
# CHANGE THESE TO UPDATE YOUR LOGIN
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

st.set_page_config(layout="wide", page_title="Jumbo Homes - Smart Search")

# Initialize session state for auth
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Login Logic
def check_login():
    user = st.session_state['input_user']
    pwd = st.session_state['input_password']
    if user == ADMIN_USER and pwd == ADMIN_PASS:
        st.session_state.authenticated = True
    else:
        st.error("😕 Incorrect Username or Password")

# If not authenticated, show ONLY the login form and stop
if not st.session_state.authenticated:
    st.markdown("## 🔒 Jumbo Homes Internal Login")
    st.text_input("Username", key="input_user")
    st.text_input("Password", type="password", key="input_password")
    st.button("Login", on_click=check_login)
    st.stop()  # <--- THIS STOPS THE APP HERE IF NOT LOGGED IN

# ==========================================
#  ✅ MAIN APP STARTS HERE (Only runs if logged in)
# ==========================================

# Cache data loading
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Clean Coordinates
    df = df.dropna(subset=['Building/Lat', 'Building/Long'])
    
    # 2. Clean Price
    def clean_price(val):
        try:
            return float(val)
        except:
            return None
    df['Clean_Price'] = df['Home/Ask_Price (lacs)'].apply(clean_price)
    
    # 3. Clean Configuration
    def extract_bhk(val):
        if pd.isna(val): return 0
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else 0
    df['BHK_Num'] = df['Home/Configuration'].apply(extract_bhk)
    
    # 4. Fill missing text
    df['Internal/Status'] = df['Internal/Status'].fillna('Unknown')
    
    return df

try:
    df = load_data("Homes (22).csv")
except FileNotFoundError:
    st.error("CSV file not found. Please ensure 'Homes (22).csv' is in the app directory.")
    st.stop()

# --- SIDEBAR LOGOUT BUTTON ---
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Filters & Search")

# A. Status Filter
all_statuses = df['Internal/Status'].unique().tolist()
# Filter for default visible statuses
default_statuses = [s for s in all_statuses if any(x in s for x in ['Live', 'Inspection Pending', 'Catalogue Pending'])]

selected_statuses = st.sidebar.multiselect(
    "Filter by Status",
    options=all_statuses,
    default=default_statuses
)

filtered_df = df[df['Internal/Status'].isin(selected_statuses)]

# B. Text Search
search_query = st.sidebar.text_input("Search (House ID or Project)", placeholder="e.g. Godrej")

if search_query:
    mask_search = (
        filtered_df['House_ID'].astype(str).str.contains(search_query, case=False, na=False) | 
        filtered_df['Building/Name'].astype(str).str.contains(search_query, case=False, na=False)
    )
    search_matches = filtered_df[mask_search]
    st.sidebar.caption(f"Found {len(search_matches)} matches.")
else:
    search_matches = pd.DataFrame()

# --- SIMILAR HOMES LOGIC ---
st.sidebar.markdown("---")
st.sidebar.header("🏠 Find Similar Homes")

candidate_homes = search_matches if not search_matches.empty else filtered_df
reference_house_id = st.sidebar.selectbox(
    "Select Reference House",
    options=["None"] + candidate_homes['House_ID'].tolist(),
    index=0
)

similar_homes = pd.DataFrame()
ref_house = None

if reference_house_id != "None":
    ref_house = df[df['House_ID'] == reference_house_id].iloc[0]
    
    st.sidebar.markdown(f"**Selected:** {ref_house['House_ID']}")
    st.sidebar.markdown(f"**Config:** {ref_house['Home/Configuration']} | **Price:** {ref_house['Clean_Price']} L")
    
    price_range = st.sidebar.slider("Price Range (± Lakhs)", 5, 100, 20)
    dist_radius = st.sidebar.slider("Distance Radius (km)", 0.5, 10.0, 2.0)
    
    # 1. Status Filter
    valid_status_mask = df['Internal/Status'].isin(default_statuses)
    
    # 2. Config Filter (Same or More BHK)
    config_mask = df['BHK_Num'] >= ref_house['BHK_Num']
    
    # 3. Price Filter
    if pd.notnull(ref_house['Clean_Price']):
        min_p = ref_house['Clean_Price'] - price_range
        max_p = ref_house['Clean_Price'] + price_range
        price_mask = (df['Clean_Price'] >= min_p) & (df['Clean_Price'] <= max_p)
    else:
        price_mask = [True] * len(df)
        
    candidates = df[valid_status_mask & config_mask & price_mask].copy()
    
    # 4. Distance Calculation
    ref_coords = (ref_house['Building/Lat'], ref_house['Building/Long'])
    
    def get_distance(row):
        return geodesic(ref_coords, (row['Building/Lat'], row['Building/Long'])).km
        
    candidates['Distance_km'] = candidates.apply(get_distance, axis=1)
    
    similar_homes = candidates[(candidates['Distance_km'] <= dist_radius) & (candidates['House_ID'] != reference_house_id)]
    
    st.sidebar.success(f"Found {len(similar_homes)} similar homes!")

# --- MAP VISUALIZATION ---

if not search_matches.empty:
    center_lat = search_matches['Building/Lat'].mean()
    center_long = search_matches['Building/Long'].mean()
    zoom = 12
elif reference_house_id != "None":
    center_lat = ref_house['Building/Lat']
    center_long = ref_house['Building/Long']
    zoom = 13
else:
    center_lat = 12.9716
    center_long = 77.5946
    zoom = 11

m = folium.Map(location=[center_lat, center_long], zoom_start=zoom)

def create_tooltip(row):
    return f"""
    <b>ID:</b> {row['House_ID']}<br>
    <b>Price:</b> {row['Home/Ask_Price (lacs)']} L<br>
    <b>Area:</b> {row['Home/Area (super-builtup)']}<br>
    <b>Floor:</b> {row['Home/Floor']}<br>
    <b>Facing:</b> {row['Home/Facing']}<br>
    <b>Status:</b> {row['Internal/Status']}
    """

# Plot Similar Homes (Green)
if not similar_homes.empty:
    for idx, row in similar_homes.iterrows():
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row),
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
            popup=f"SIMILAR: {row['House_ID']}"
        ).add_to(m)

# Plot Search Matches (Red)
highlight_ids = []
if not search_matches.empty:
    highlight_ids = search_matches['House_ID'].tolist()
    for idx, row in search_matches.iterrows():
        if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values:
            continue 
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row),
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
            popup=f"MATCH: {row['House_ID']}"
        ).add_to(m)

# Plot Others (Blue)
for idx, row in filtered_df.iterrows():
    if row['House_ID'] in highlight_ids: continue
    if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values: continue
    if reference_house_id != "None" and row['House_ID'] == reference_house_id:
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip="REFERENCE: " + create_tooltip(row),
            icon=folium.Icon(color="black", icon="user", prefix="fa"),
            popup="REFERENCE HOUSE"
        ).add_to(m)
        continue

    folium.Marker(
        [row['Building/Lat'], row['Building/Long']],
        tooltip=create_tooltip(row),
        icon=folium.Icon(color="blue", icon="info-sign"), 
    ).add_to(m)

# --- RENDER UI ---
st.title("Jumbo Homes Discovery")

st_folium(m, width="100%", height=500)

if not similar_homes.empty:
    st.subheader(f"Similar Homes to {reference_house_id}")
    display_cols = ['House_ID', 'Internal/Status', 'Home/Configuration', 'Home/Ask_Price (lacs)', 'Home/Area (super-builtup)', 'Distance_km']
    st.dataframe(similar_homes[display_cols].style.format({"Distance_km": "{:.2f} km"}))
elif reference_house_id == "None":
    st.info("Select a 'Reference House' from the sidebar to see similar property recommendations.")