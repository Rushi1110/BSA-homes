import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import re

st.set_page_config(layout="wide", page_title="Jumbo Homes - Discovery Portal")

# --- HIDE STREAMLIT DEFAULT BUTTONS ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
#  ✅ MAIN APP LOGIC
# ==========================================

# --- 1. FAST MATH FUNCTIONS ---
def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2_array), np.radians(lon2_array)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- 2. DATA LOADING & CLEANING ---
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Force numeric coords
    df['Building/Lat'] = pd.to_numeric(df['Building/Lat'], errors='coerce')
    df['Building/Long'] = pd.to_numeric(df['Building/Long'], errors='coerce')
    df = df.dropna(subset=['Building/Lat', 'Building/Long'])
    
    # Clean Price
    def clean_price(val):
        try:
            return float(val)
        except:
            return None
    df['Clean_Price'] = df['Home/Ask_Price (lacs)'].apply(clean_price)
    
    # Clean Price Per Sqft (For Sorting)
    def clean_psqft(val):
        try:
            # Remove commas and non-numeric chars
            clean_str = re.sub(r'[^\d.]', '', str(val))
            return float(clean_str)
        except:
            return 999999.0 # Push to bottom if invalid
    
    # Assuming column name is 'Home/Price psqft', adjust if it differs in your CSV
    if 'Home/Price psqft' in df.columns:
        df['Clean_Psqft'] = df['Home/Price psqft'].apply(clean_psqft)
    else:
        # Fallback if column missing
        df['Clean_Psqft'] = 0.0

    # Clean Config
    def extract_bhk(val):
        if pd.isna(val): return 0
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else 0
    df['BHK_Num'] = df['Home/Configuration'].apply(extract_bhk)
    
    # Fill text gaps
    df['Internal/Status'] = df['Internal/Status'].fillna('Unknown')
    df['Building/Locality'] = df['Building/Locality'].fillna('')
    df['Building/Name'] = df['Building/Name'].fillna('Unknown Project')
    
    return df

try:
    df = load_data("Homes.csv")
except FileNotFoundError:
    st.error("❌ Critical Error: 'Homes.csv' not found.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---

st.sidebar.image("https://res.cloudinary.com/dewcjgpc7/image/upload/v1763541879/10_jpqpx1.png", use_container_width=True)
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

# --- 4. MAIN MAP ---
st.title("Discovery Portal")
with st.expander("ℹ️ **How to use this tool**"):
    st.markdown("""
    1. **Blue Pins:** Buildings (Hover to see all units sorted by value).
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

# --- HELPER: Create Multi-Unit Tooltip ---
def create_building_tooltip(group_df, building_name, locality):
    # Sort by Price Per Sqft (Low to High)
    sorted_group = group_df.sort_values('Clean_Psqft', ascending=True)
    
    # Header
    html = f"""
    <div style="font-family: sans-serif; min-width: 250px;">
        <h4 style="margin-bottom: 5px; color: #3186cc;">{building_name}</h4>
        <small style="color: #666;">📍 {locality} ({len(group_df)} units)</small>
        <hr style="margin: 5px 0;">
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr style="text-align: left; background-color: #f0f0f0;">
                <th style="padding: 3px;">Config</th>
                <th style="padding: 3px;">Price</th>
                <th style="padding: 3px;">Status</th>
                <th style="padding: 3px;">ID</th>
            </tr>
    """
    
    # Rows for each house
    for _, row in sorted_group.iterrows():
        status_icon = {'✅ Live': '✅', '☑️ Sold': '🔴', '⏳On Hold': 'Of', 'Unknown': '❓'}.get(row['Internal/Status'], '🔹')
        price_txt = f"{row['Clean_Price']}L" if pd.notnull(row['Clean_Price']) else "N/A"
        
        html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 3px;">{row['Home/Configuration']}</td>
                <td style="padding: 3px; font-weight: bold;">{price_txt}</td>
                <td style="padding: 3px;">{status_icon}</td>
                <td style="padding: 3px; color: #888;">{row['House_ID']}</td>
            </tr>
        """
    
    html += "</table></div>"
    return html

def create_single_tooltip(row, label=None):
    status_emoji = {'✅ Live': '✅', '☑️ Sold': '🔴', '⏳On Hold': 'Of', 'Unknown': '❓'}.get(row['Internal/Status'], '🔹')
    title = f"<b>{label}</b><br>" if label else ""
    def safe_val(v): return v if pd.notnull(v) else "N/A"
    return f"""{title}<b>ID:</b> {safe_val(row['House_ID'])}<br><b>Project:</b> {safe_val(row['Building/Name'])}<br><b>Locality:</b> {safe_val(row['Building/Locality'])}<br><b>Price:</b> {safe_val(row['Home/Ask_Price (lacs)'])} L<br><b>Config:</b> {safe_val(row['Home/Configuration'])}<br><b>Status:</b> {status_emoji} {safe_val(row['Internal/Status'])}"""

# Separate the datasets
special_ids = []
if not similar_homes.empty: special_ids.extend(similar_homes['House_ID'].tolist())
if not search_matches.empty: special_ids.extend(search_matches['House_ID'].tolist())
if reference_house_id != "Select a House...": special_ids.append(reference_house_id)

# 1. BLUE LAYER (General Inventory) - GROUPED BY BUILDING
blue_df = filtered_df[~filtered_df['House_ID'].isin(special_ids)]

# Group by unique location (Name + Lat + Long) to handle phases/blocks if coords differ
grouped = blue_df.groupby(['Building/Name', 'Building/Lat', 'Building/Long', 'Building/Locality'])

for (name, lat, lon, loc), group in grouped:
    # Create one pin per building
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(create_building_tooltip(group, name, loc), max_width=300),
        tooltip=f"{name} ({len(group)} Units)",
        icon=folium.Icon(color="blue", icon="building", prefix="fa"), 
    ).add_to(m)

# 2. GREEN LAYER (Similar) - Individual Pins
if not similar_homes.empty:
    for _, row in similar_homes.iterrows():
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_single_tooltip(row, "SIMILAR MATCH"),
            icon=folium.Icon(color="green", icon="thumbs-up", prefix="fa"),
        ).add_to(m)

# 3. RED LAYER (Search) - Individual Pins
if not search_matches.empty:
    for _, row in search_matches.iterrows():
        if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values: continue
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_single_tooltip(row, "SEARCH RESULT"),
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(m)

# 4. BLACK LAYER (Reference)
if reference_house_id != "Select a House...":
    folium.Marker(
        [ref_house['Building/Lat'], ref_house['Building/Long']],
        tooltip=create_single_tooltip(ref_house, "REFERENCE HOUSE"),
        icon=folium.Icon(color="black", icon="user", prefix="fa"),
    ).add_to(m)

# 🚀 THE FIX: returned_objects=[] prevents the map from sending zoom/pan data back to Python
st_folium(m, width="100%", height=550, returned_objects=[])

# Data Table
if not similar_homes.empty:
    st.subheader(f"✅ Recommended Similar Homes ({len(similar_homes)})")
    display_cols = ['House_ID', 'Building/Name', 'Building/Locality', 'Internal/Status', 'Home/Configuration', 
                   'Home/Ask_Price (lacs)', 'Home/Area (super-builtup)', 'Clean_Psqft']
    display_df = similar_homes[display_cols].copy()
    display_df.columns = ['ID', 'Project', 'Locality', 'Status', 'Config', 'Price (L)', 'Area (sqft)', 'Price/Sqft']
    st.dataframe(display_df.style.format({"Price (L)": "{:.2f}", "Price/Sqft": "{:.0f}"}), use_container_width=True)

elif reference_house_id == "Select a House..." and not search_matches.empty:
    st.subheader("Search Results")
    st.dataframe(search_matches[['House_ID', 'Building/Name', 'Building/Locality', 'Internal/Status', 'Home/Ask_Price (lacs)']], use_container_width=True)
