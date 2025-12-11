import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import re

st.set_page_config(layout="wide", page_title="Jumbo Homes - Discovery Portal")

# --- HIDE STREAMLIT DEFAULT BUTTONS (THE FIX) ---
# We hide the 'stToolbar' (top right) but keep the sidebar toggle (top left)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="stToolbar"] {visibility: hidden !important;}
            [data-testid="stDecoration"] {display: none;}
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
            clean_str = re.sub(r'[^\d.]', '', str(val))
            return float(clean_str)
        except:
            return 999999.0
    
    if 'Home/Price psqft' in df.columns:
        df['Clean_Psqft'] = df['Home/Price psqft'].apply(clean_psqft)
    else:
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
    2. **Red Stars:** Search results (Hover to see specific unit + others in building).
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

# --- TOOLTIP HELPERS ---

def create_table_row(row, bg_color="#fff", bold=False):
    """Helper to create a single row in the HTML table"""
    status_icon = {'✅ Live': '✅', '☑️ Sold': '🔴', '⏳On Hold': 'Of', 'Unknown': '❓'}.get(row['Internal/Status'], '🔹')
    price_txt = f"{row['Clean_Price']}L" if pd.notnull(row['Clean_Price']) else "N/A"
    style_td = "padding: 3px;"
    style_price = "padding: 3px; font-weight: bold;" if not bold else "padding: 3px; font-weight: 800; color: #d32f2f;"
    
    return f"""
        <tr style="border-bottom: 1px solid #eee; background-color: {bg_color};">
            <td style="{style_td}">{row['Home/Configuration']}</td>
            <td style="{style_price}">{price_txt}</td>
            <td style="{style_td}">{status_icon}</td>
            <td style="{style_td} color: #666; font-size: 11px;">{row['House_ID']}</td>
        </tr>
    """

def create_building_tooltip(group_df, building_name, locality):
    """Tooltip for BLUE pins: Shows all units in building"""
    sorted_group = group_df.sort_values('Clean_Psqft', ascending=True)
    
    html = f"""
    <div style="font-family: sans-serif; min-width: 250px;">
        <h4 style="margin-bottom: 2px; color: #3186cc;">{building_name}</h4>
        <small style="color: #666;">📍 {locality}</small>
        <hr style="margin: 5px 0;">
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr style="text-align: left; background-color: #f0f0f0;">
                <th>Config</th><th>Price</th><th>St</th><th>ID</th>
            </tr>
    """
    for _, row in sorted_group.iterrows():
        html += create_table_row(row)
        
    html += "</table></div>"
    return html

def create_context_tooltip(target_row, full_df, label):
    """Tooltip for RED/GREEN pins: Shows target unit first, then siblings"""
    
    # 1. Find siblings in same building
    b_name = target_row['Building/Name']
    siblings = full_df[
        (full_df['Building/Name'] == b_name) & 
        (full_df['House_ID'] != target_row['House_ID'])
    ]
    
    valid_sibs = siblings[siblings['Internal/Status'].isin(default_statuses)].sort_values('Clean_Psqft', ascending=True)

    html = f"""
    <div style="font-family: sans-serif; min-width: 260px;">
        <div style="background-color: #f8f9fa; padding: 5px; border-radius: 4px; border: 1px solid #ddd; margin-bottom: 5px;">
            <strong style="color: #d32f2f;">{label}</strong><br>
            <span style="font-size: 14px; font-weight: bold;">{target_row['Building/Name']}</span>
        </div>
        
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr style="text-align: left; background-color: #f0f0f0;">
                <th>Config</th><th>Price</th><th>St</th><th>ID</th>
            </tr>
            {create_table_row(target_row, bg_color="#fff3cd", bold=True)}
    """
    
    if not valid_sibs.empty:
        html += f"""
            <tr><td colspan="4" style="text-align: center; font-size: 10px; color: #888; padding: 2px;">
                ▼ Other Units in Building ▼
            </td></tr>
        """
        for _, row in valid_sibs.iterrows():
            html += create_table_row(row)
            
    html += "</table></div>"
    return html

# Separate the datasets
special_ids = []
if not similar_homes.empty: special_ids.extend(similar_homes['House_ID'].tolist())
if not search_matches.empty: special_ids.extend(search_matches['House_ID'].tolist())
if reference_house_id != "Select a House...": special_ids.append(reference_house_id)

# 1. BLUE LAYER (General Inventory) - GROUPED BY BUILDING
blue_df = filtered_df[~filtered_df['House_ID'].isin(special_ids)]
grouped = blue_df.groupby(['Building/Name', 'Building/Lat', 'Building/Long', 'Building/Locality'])

for (name, lat, lon, loc), group in grouped:
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(create_building_tooltip(group, name, loc), max_width=300),
        tooltip=f"{name} ({len(group)} Units)",
        icon=folium.Icon(color="blue", icon="building", prefix="fa"), 
    ).add_to(m)

# 2. GREEN LAYER (Similar) - Context Aware
if not similar_homes.empty:
    for _, row in similar_homes.iterrows():
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            popup=folium.Popup(create_context_tooltip(row, df, "SIMILAR MATCH"), max_width=300),
            tooltip=f"Similar: {row['House_ID']}",
            icon=folium.Icon(color="green", icon="thumbs-up", prefix="fa"),
        ).add_to(m)

# 3. RED LAYER (Search) - Context Aware
if not search_matches.empty:
    for _, row in search_matches.iterrows():
        if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values: continue
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            popup=folium.Popup(create_context_tooltip(row, df, "SEARCH RESULT"), max_width=300),
            tooltip=f"Match: {row['House_ID']}",
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(m)

# 4. BLACK LAYER (Reference)
if reference_house_id != "Select a House...":
    folium.Marker(
        [ref_house['Building/Lat'], ref_house['Building/Long']],
        popup=folium.Popup(create_context_tooltip(ref_house, df, "REFERENCE HOUSE"), max_width=300),
        tooltip="REFERENCE HOUSE",
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
