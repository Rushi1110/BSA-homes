import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import re

# --- 0. AUTHENTICATION CONFIG ---
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

st.set_page_config(layout="wide", page_title="Jumbo Homes - Discovery Portal")

# --- 1. AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_login():
    user = st.session_state.get('input_user', '')
    pwd = st.session_state.get('input_password', '')
    if user == ADMIN_USER and pwd == ADMIN_PASS:
        st.session_state.authenticated = True
    else:
        st.error("😕 Incorrect Username or Password")

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.image("https://res.cloudinary.com/dewcjgpc7/image/upload/v1763541879/10_jpqpx1.png", width=200)
        st.markdown("### Internal Login")
        st.text_input("Username", key="input_user")
        st.text_input("Password", type="password", key="input_password")
        st.button("Login", on_click=check_login, type="primary")
    st.stop()

# ==========================================
#  ✅ MAIN APP STARTS HERE
# ==========================================

# --- 2. FAST MATH FUNCTIONS ---
def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2_array), np.radians(lon2_array)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- 3. DATA LOADING & CLEANING ---
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    df['Building/Lat'] = pd.to_numeric(df['Building/Lat'], errors='coerce')
    df['Building/Long'] = pd.to_numeric(df['Building/Long'], errors='coerce')
    df = df.dropna(subset=['Building/Lat', 'Building/Long'])
    
    def clean_price(val):
        try:
            return float(val)
        except:
            return None
    df['Clean_Price'] = df['Home/Ask_Price (lacs)'].apply(clean_price)
    
    def extract_bhk(val):
        if pd.isna(val): return 0
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else 0
    df['BHK_Num'] = df['Home/Configuration'].apply(extract_bhk)
    
    df['Internal/Status'] = df['Internal/Status'].fillna('Unknown')
    df['Building/Locality'] = df['Building/Locality'].fillna('')
    
    return df

try:
    df = load_data("Homes.csv")
except FileNotFoundError:
    st.error("❌ Critical Error: 'Homes.csv' not found.")
    st.stop()

# --- 4. SIDEBAR CONTROLS ---

st.sidebar.image("https://res.cloudinary.com/dewcjgpc7/image/upload/v1763541879/10_jpqpx1.png", use_container_width=True)
if st.sidebar.button("Logout", icon="🔒"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.divider()

# Section A: Filter & Search
st.sidebar.subheader("🔍 Search Inventory")
search_query = st.sidebar.text_input("Search (Locality, Project, ID)", placeholder="e.g. Whitefield, Godrej...")

# Status Filter
all_statuses = sorted(df['Internal/Status'].unique().tolist())
default_statuses = [s for s in all_statuses if any(x in s for x in ['Live', 'Inspection Pending', 'Catalogue Pending'])]
selected_statuses = st.sidebar.multiselect("Filter Status", options=all_statuses, default=default_statuses)

filtered_df = df[df['Internal/Status'].isin(selected_statuses)]

# Search Logic
search_matches = pd.DataFrame()
if search_query:
    mask = (
        filtered_df['House_ID'].astype(str).str.contains(search_query, case=False, na=False) | 
        filtered_df['Building/Name'].astype(str).str.contains(search_query, case=False, na=False) |
        filtered_df['Building/Locality'].astype(str).str.contains(search_query, case=False, na=False)
    )
    search_matches = filtered_df[mask]
    if not search_matches.empty:
        st.sidebar.success(f"Found {len(search_matches)} matches")
    else:
        st.sidebar.warning("No matches found")

st.sidebar.divider()

# Section B: Similar Homes
st.sidebar.subheader("🏠 Similar Homes Engine")
candidate_homes = search_matches if not search_matches.empty else filtered_df
reference_house_id = st.sidebar.selectbox("1. Select Reference House", options=["Select a House..."] + candidate_homes['House_ID'].tolist(), index=0)

similar_homes = pd.DataFrame()
ref_house = None

if reference_house_id != "Select a House...":
    ref_house = df[df['House_ID'] == reference_house_id].iloc[0]
    st.sidebar.info(f"**Ref:** {ref_house['Building/Name']}\n\n📍 {ref_house['Building/Locality']} | 💰 {ref_house['Clean_Price']} L")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    price_range = col_s1.slider("± Price (L)", 5, 100, 20)
    dist_radius = col_s2.slider("Radius (km)", 0.5, 10.0, 2.0)
    
    valid_status_mask = df['Internal/Status'].isin(default_statuses)
    config_mask = df['BHK_Num'] >= ref_house['BHK_Num']
    if pd.notnull(ref_house['Clean_Price']):
        min_p = ref_house['Clean_Price'] - price_range
        max_p = ref_house['Clean_Price'] + price_range
        price_mask = (df['Clean_Price'] >= min_p) & (df['Clean_Price'] <= max_p)
    else:
        price_mask = [True] * len(df)
        
    candidates = df[valid_status_mask & config_mask & price_mask].copy()
    if not candidates.empty:
        candidates['Distance_km'] = haversine_vectorized(
            ref_house['Building/Lat'], ref_house['Building/Long'], 
            candidates['Building/Lat'].values, candidates['Building/Long'].values
        )
        similar_homes = candidates[(candidates['Distance_km'] <= dist_radius) & (candidates['House_ID'] != reference_house_id)]
        similar_homes = similar_homes.sort_values(by="Clean_Price", ascending=True)
    st.sidebar.metric("Similar Homes Found", len(similar_homes))

# --- 5. MAIN MAP ---
st.title("Discovery Portal")
with st.expander("ℹ️ **How to use this tool**"):
    st.markdown("""
    1. **Blue Pins:** General inventory.
    2. **Red Stars:** Search results.
    3. **Green Thumbs:** Recommended similar homes.
    """)

# Center Logic
if not search_matches.empty:
    center_lat, center_long, zoom = search_matches['Building/Lat'].mean(), search_matches['Building/Long'].mean(), 12
elif reference_house_id != "Select a House...":
    center_lat, center_long, zoom = ref_house['Building/Lat'], ref_house['Building/Long'], 13
else:
    center_lat, center_long, zoom = 12.9716, 77.5946, 11

m = folium.Map(location=[center_lat, center_long], zoom_start=zoom, prefer_canvas=True)

def create_tooltip(row, label=None):
    status_emoji = {'✅ Live': '✅', '☑️ Sold': '🔴', '⏳On Hold': 'Of', 'Unknown': '❓'}.get(row['Internal/Status'], '🔹')
    title = f"<b>{label}</b><br>" if label else ""
    def safe_val(v): return v if pd.notnull(v) else "N/A"
    return f"""{title}<b>ID:</b> {safe_val(row['House_ID'])}<br><b>Project:</b> {safe_val(row['Building/Name'])}<br><b>Locality:</b> {safe_val(row['Building/Locality'])}<br><b>Price:</b> {safe_val(row['Home/Ask_Price (lacs)'])} L<br><b>Config:</b> {safe_val(row['Home/Configuration'])}<br><b>Status:</b> {status_emoji} {safe_val(row['Internal/Status'])}"""

# Separate the datasets
special_ids = []
if not similar_homes.empty: special_ids.extend(similar_homes['House_ID'].tolist())
if not search_matches.empty: special_ids.extend(search_matches['House_ID'].tolist())
if reference_house_id != "Select a House...": special_ids.append(reference_house_id)

# 1. BLUE LAYER (General Inventory) - NOW USING STANDARD PINS
# We filter out the special IDs so we don't draw double pins
blue_df = filtered_df[~filtered_df['House_ID'].isin(special_ids)]

for _, row in blue_df.iterrows():
    folium.Marker(
        location=[row['Building/Lat'], row['Building/Long']],
        tooltip=create_tooltip(row),
        icon=folium.Icon(color="blue", icon="home", prefix="fa"), # <--- CHANGED TO PIN
    ).add_to(m)

# 2. GREEN LAYER (Similar)
if not similar_homes.empty:
    for _, row in similar_homes.iterrows():
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row, "SIMILAR MATCH"),
            icon=folium.Icon(color="green", icon="thumbs-up", prefix="fa"),
        ).add_to(m)

# 3. RED LAYER (Search)
if not search_matches.empty:
    for _, row in search_matches.iterrows():
        if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values: continue
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row, "SEARCH RESULT"),
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(m)

# 4. BLACK LAYER (Reference)
if reference_house_id != "Select a House...":
    folium.Marker(
        [ref_house['Building/Lat'], ref_house['Building/Long']],
        tooltip=create_tooltip(ref_house, "REFERENCE HOUSE"),
        icon=folium.Icon(color="black", icon="user", prefix="fa"),
    ).add_to(m)

# 🚀 THE FIX: returned_objects=[] prevents the map from sending zoom/pan data back to Python
st_folium(m, width="100%", height=550, returned_objects=[])

# Data Table
if not similar_homes.empty:
    st.subheader(f"✅ Recommended Similar Homes ({len(similar_homes)})")
    display_cols = ['House_ID', 'Building/Name', 'Building/Locality', 'Internal/Status', 'Home/Configuration', 
                   'Home/Ask_Price (lacs)', 'Home/Area (super-builtup)', 'Distance_km']
    display_df = similar_homes[display_cols].copy()
    display_df.columns = ['ID', 'Project', 'Locality', 'Status', 'Config', 'Price (L)', 'Area (sqft)', 'Distance (km)']
    st.dataframe(display_df.style.format({"Distance (km)": "{:.2f}", "Price (L)": "{:.2f}"}), use_container_width=True)

elif reference_house_id == "Select a House..." and not search_matches.empty:
    st.subheader("Search Results")
    st.dataframe(search_matches[['House_ID', 'Building/Name', 'Building/Locality', 'Internal/Status', 'Home/Ask_Price (lacs)']], use_container_width=True)
