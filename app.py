import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
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
    # Login Screen
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

# --- 2. DATA LOADING & CLEANING ---
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Force convert coordinates to numeric (fixes crashes with bad data)
    df['Building/Lat'] = pd.to_numeric(df['Building/Lat'], errors='coerce')
    df['Building/Long'] = pd.to_numeric(df['Building/Long'], errors='coerce')
    
    # Drop invalid rows
    df = df.dropna(subset=['Building/Lat', 'Building/Long'])
    
    # Clean Price
    def clean_price(val):
        try:
            return float(val)
        except:
            return None
    df['Clean_Price'] = df['Home/Ask_Price (lacs)'].apply(clean_price)
    
    # Clean Configuration (Extract '3' from '3 BHK')
    def extract_bhk(val):
        if pd.isna(val): return 0
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else 0
    df['BHK_Num'] = df['Home/Configuration'].apply(extract_bhk)
    
    # Fill text gaps
    df['Internal/Status'] = df['Internal/Status'].fillna('Unknown')
    
    return df

try:
    df = load_data("Homes.csv")
except FileNotFoundError:
    st.error("❌ Critical Error: 'Homes.csv' not found. Please upload it to the repository.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---

# Logo & Logout
st.sidebar.image("https://res.cloudinary.com/dewcjgpc7/image/upload/v1763541879/10_jpqpx1.png", use_container_width=True)
if st.sidebar.button("Logout", icon="🔒"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.divider()

# Section A: Filter & Search
st.sidebar.subheader("🔍 Search Inventory")

# Search
search_query = st.sidebar.text_input("Find Project / House ID", placeholder="e.g. Godrej, J-100...")

# Status Filter
all_statuses = sorted(df['Internal/Status'].unique().tolist())
default_statuses = [s for s in all_statuses if any(x in s for x in ['Live', 'Inspection Pending', 'Catalogue Pending'])]

selected_statuses = st.sidebar.multiselect(
    "Filter Status",
    options=all_statuses,
    default=default_statuses
)

# Apply Filters
filtered_df = df[df['Internal/Status'].isin(selected_statuses)]

search_matches = pd.DataFrame()
if search_query:
    mask = (
        filtered_df['House_ID'].astype(str).str.contains(search_query, case=False, na=False) | 
        filtered_df['Building/Name'].astype(str).str.contains(search_query, case=False, na=False)
    )
    search_matches = filtered_df[mask]
    if not search_matches.empty:
        st.sidebar.success(f"Found {len(search_matches)} matches")
    else:
        st.sidebar.warning("No matches found")

st.sidebar.divider()

# Section B: Similar Homes Engine
st.sidebar.subheader("🏠 Similar Homes Engine")

# Dropdown options: Prioritize search results, otherwise show all
candidate_homes = search_matches if not search_matches.empty else filtered_df
reference_house_id = st.sidebar.selectbox(
    "1. Select Reference House",
    options=["Select a House..."] + candidate_homes['House_ID'].tolist(),
    index=0
)

similar_homes = pd.DataFrame()
ref_house = None

if reference_house_id != "Select a House...":
    ref_house = df[df['House_ID'] == reference_house_id].iloc[0]
    
    # Display Reference Details
    st.sidebar.info(
        f"**Ref:** {ref_house['Building/Name']}\n\n"
        f"🛏️ {ref_house['Home/Configuration']} | 💰 {ref_house['Clean_Price']} L"
    )
    
    # Compact Sliders
    col_s1, col_s2 = st.sidebar.columns(2)
    price_range = col_s1.slider("± Price (L)", 5, 100, 20)
    dist_radius = col_s2.slider("Radius (km)", 0.5, 10.0, 2.0)
    
    # LOGIC: Find Similar
    # 1. Status Check
    valid_status_mask = df['Internal/Status'].isin(default_statuses)
    # 2. Config Check (Same or bigger)
    config_mask = df['BHK_Num'] >= ref_house['BHK_Num']
    # 3. Price Check
    if pd.notnull(ref_house['Clean_Price']):
        min_p = ref_house['Clean_Price'] - price_range
        max_p = ref_house['Clean_Price'] + price_range
        price_mask = (df['Clean_Price'] >= min_p) & (df['Clean_Price'] <= max_p)
    else:
        price_mask = [True] * len(df)
        
    candidates = df[valid_status_mask & config_mask & price_mask].copy()
    
    # 4. Distance Check
    ref_coords = (ref_house['Building/Lat'], ref_house['Building/Long'])
    
    def get_distance(row):
        return geodesic(ref_coords, (row['Building/Lat'], row['Building/Long'])).km
        
    candidates['Distance_km'] = candidates.apply(get_distance, axis=1)
    
    similar_homes = candidates[
        (candidates['Distance_km'] <= dist_radius) & 
        (candidates['House_ID'] != reference_house_id)
    ]
    
    st.sidebar.metric("Similar Homes Found", len(similar_homes))


# --- 4. MAIN PAGE UI ---

st.title("Discovery Portal")

# Instructions Expander
with st.expander("ℹ️ **How to use this tool**"):
    st.markdown("""
    1. **Explore the Map:** By default, you see all 'Live' and 'Pending' properties (Blue pins).
    2. **Search:** Use the sidebar to find a specific Project or House ID. These will highlight in **Red**.
    3. **Find Comparisons:** * Select a "Reference House" in the sidebar.
        * Adjust the Price and Distance sliders.
        * The app will find similar homes (Green pins) that match the configuration (BHK) and budget.
    """)

# Determine Map Center
if not search_matches.empty:
    center_lat = search_matches['Building/Lat'].mean()
    center_long = search_matches['Building/Long'].mean()
    zoom = 12
elif reference_house_id != "Select a House...":
    center_lat = ref_house['Building/Lat']
    center_long = ref_house['Building/Long']
    zoom = 13
else:
    center_lat = 12.9716 # Bangalore default
    center_long = 77.5946
    zoom = 11

m = folium.Map(location=[center_lat, center_long], zoom_start=zoom)

# Tooltip Helper
def create_tooltip(row, label=None):
    status_emoji = {
        '✅ Live': '✅', '☑️ Sold': '🔴', '⏳On Hold': 'Of', 'Unknown': '❓'
    }.get(row['Internal/Status'], '🔹')
    
    title = f"<b>{label}</b><br>" if label else ""
    return f"""
    {title}
    <b>ID:</b> {row['House_ID']}<br>
    <b>Project:</b> {row['Building/Name']}<br>
    <b>Price:</b> {row['Home/Ask_Price (lacs)']} L<br>
    <b>Config:</b> {row['Home/Configuration']}<br>
    <b>Status:</b> {status_emoji} {row['Internal/Status']}
    """

# LAYER 1: Similar Homes (Green) - Top Priority
if not similar_homes.empty:
    for _, row in similar_homes.iterrows():
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row, "SIMILAR MATCH"),
            icon=folium.Icon(color="green", icon="thumbs-up", prefix="fa"),
        ).add_to(m)

# LAYER 2: Search Matches (Red)
highlight_ids = []
if not search_matches.empty:
    highlight_ids = search_matches['House_ID'].tolist()
    for _, row in search_matches.iterrows():
        # Don't double-plot if it's already green
        if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values:
            continue
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row, "SEARCH RESULT"),
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(m)

# LAYER 3: The Rest (Blue)
for _, row in filtered_df.iterrows():
    # Skip if already highlighted or similar
    if row['House_ID'] in highlight_ids: continue
    if not similar_homes.empty and row['House_ID'] in similar_homes['House_ID'].values: continue
    
    # Reference House (Black)
    if reference_house_id != "Select a House..." and row['House_ID'] == reference_house_id:
        folium.Marker(
            [row['Building/Lat'], row['Building/Long']],
            tooltip=create_tooltip(row, "REFERENCE HOUSE"),
            icon=folium.Icon(color="black", icon="user", prefix="fa"),
        ).add_to(m)
        continue

    # Standard Pin
    folium.Marker(
        [row['Building/Lat'], row['Building/Long']],
        tooltip=create_tooltip(row),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

# Render Map
st_folium(m, width="100%", height=550)

# Data Table below map
if not similar_homes.empty:
    st.subheader(f"✅ Recommended Similar Homes ({len(similar_homes)})")
    
    display_cols = ['House_ID', 'Building/Name', 'Internal/Status', 'Home/Configuration', 
                   'Home/Ask_Price (lacs)', 'Home/Area (super-builtup)', 'Distance_km']
    
    # Clean up column names for display
    display_df = similar_homes[display_cols].copy()
    display_df.columns = ['ID', 'Project', 'Status', 'Config', 'Price (L)', 'Area (sqft)', 'Distance (km)']
    
    st.dataframe(
        display_df.style.format({"Distance (km)": "{:.2f}", "Price (L)": "{:.2f}"}),
        use_container_width=True
    )

elif reference_house_id == "Select a House..." and not search_matches.empty:
    st.subheader("Search Results")
    st.dataframe(search_matches[['House_ID', 'Building/Name', 'Internal/Status', 'Home/Ask_Price (lacs)']], use_container_width=True)
