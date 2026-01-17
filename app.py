import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aadhaar 360 - AI Insight Portal",
    page_icon="🇮🇳",
    layout="wide"
)

# --- 2. SESSION STATE FOR LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- 3. PREMIUM UI STYLING (CSS) ---
st.markdown("""
    <style>
    /* Metric Value (Numbers) Styling */
    [data-testid="stMetricValue"] { 
        font-size: 30px !important; 
        font-weight: bold !important; 
        color: #003366 !important; 
    }
    
    /* Metric Label (Heading) Styling */
    [data-testid="stMetricLabel"] { 
        font-size: 15px !important; 
        font-weight: 700 !important; 
        color: #1a1a1a !important; 
    }

    /* Metric Card Styling */
    .stMetric {
        background-color: #eef6ff !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        border: 1.5px solid #d1e3ff !important;
    }

    .main { background-color: #f9fbfd; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; 
        background-color: #f0f2f6; 
        border-radius: 8px 8px 0 0; 
        padding: 8px 16px; 
        font-weight: 600;
        color: #1a1a1a !important;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #003366 !important; 
        color: white !important; 
    }
    
    /* Insight Box Styling - Enhanced Contrast for Readability */
    .insight-box {
        background-color: #ffffff !important;
        color: #000000 !important; /* High contrast black */
        padding: 20px;
        border-left: 6px solid #003366;
        border: 1px solid #d1e3ff;
        border-radius: 8px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 16px;
        line-height: 1.6;
    }
    
    .insight-box b {
        color: #003366 !important;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LOGIN INTERFACE ---
def login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    cols = st.columns([1, 2, 1])
    with cols[1]:
        # Using a professional layout for the login screen
        st.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=150)
        st.title("Secure Personnel Access")
        st.info("Please enter your authorized UIDAI credentials to proceed.")
        
        with st.form("login_form"):
            # Credentials with professional placeholders
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="uidai@2026")
            submit = st.form_submit_button("Authenticate System Access")
            
            if submit:
                # Security Verification Logic
                if username == "admin" and password == "uidai@2026":
                    st.session_state.logged_in = True
                    st.success("Authentication successful. Initializing analytical modules...")
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid username or password entry.")

# --- 5. DATA CLEANING UTILITIES ---
def deep_clean_text(text):
    """Handles hidden characters, special symbols, and standardizes conjunctions for text entries."""
    if pd.isna(text) or str(text).strip() == "": return "Unknown"
    # Removing non-breaking spaces and hidden unicode
    t = str(text).replace('\xa0', ' ').replace('\u200b', '').strip()
    # Standardizing connectors
    t = t.replace('&', ' and ')
    # Sanitizing symbols while keeping alphanumeric and spaces
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    # Removing redundant whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t.title()

def clean_dataframe(df):
    """Standardizes column headers and sanitizes text/numeric data across the dataframe."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'state' in df.columns: df['state'] = df['state'].apply(deep_clean_text)
    if 'district' in df.columns: df['district'] = df['district'].apply(deep_clean_text)
    # Converting identified numeric columns to standard types
    for col in df.columns:
        if any(x in col for x in ['age', 'demo', 'bio', 'count']):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# --- 6. DATA LOADING ENGINE ---
@st.cache_data
def load_all_data():
    """Initializes and loads the core Aadhaar datasets from the local data directory."""
    files = {'enrol': 'data/enrolment.csv', 'demo': 'data/demographic.csv', 'bio': 'data/biometric.csv'}
    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            data[key] = clean_dataframe(df)
            if 'date' in data[key].columns:
                data[key]['date'] = pd.to_datetime(data[key]['date'], format='mixed', errors='coerce')
        else: return None
    return data

# --- MAIN APPLICATION LOGIC ---
if not st.session_state.logged_in:
    login()
else:
    # --- DASHBOARD INITIALIZATION ---
    datasets = load_all_data()

    if datasets:
        enrol_df, demo_df, bio_df = datasets['enrol'], datasets['demo'], datasets['bio']

        # --- 7. SIDEBAR SYSTEM CONTROLS ---
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=120)
        st.sidebar.title("Operational Controls")
        
        # Establishing Geographical Scoping
        unique_states = sorted(list(set(enrol_df['state'].unique().tolist())))
        sel_state = st.sidebar.selectbox("Geographical Scope", ["All India"] + unique_states)

        if sel_state != "All India":
            state_districts = sorted(list(set(enrol_df[enrol_df['state'] == sel_state]['district'].unique().tolist())))
            sel_dist = st.sidebar.selectbox("District Focus", ["All Districts"] + state_districts)
        else:
            sel_dist = "All Districts"

        # Session Management
        st.sidebar.divider()
        if st.sidebar.button("Terminate Session"):
            st.session_state.logged_in = False
            st.rerun()

        # --- 8. FILTRATION ENGINE ---
        def filter_data(df, state_only=False):
            """Filters datasets based on the selected geographical scope and focus."""
            temp = df.copy()
            if sel_state != "All India":
                temp = temp[temp['state'] == sel_state]
                if not state_only and sel_dist != "All Districts":
                    temp = temp[temp['district'] == sel_dist]
            return temp

        f_enrol = filter_data(enrol_df)
        f_demo = filter_data(demo_df)
        f_bio = filter_data(bio_df)
        s_enrol = filter_data(enrol_df, state_only=True)

        # --- 9. EXECUTIVE SUMMARY & KPIS ---
        st.title("Aadhaar 360 AI Insight Portal")
        st.markdown(f"**Operational Scope:** {sel_state} > {sel_dist}")
        st.divider()

        # Identifying data columns for volumetric calculations
        e_cols = [c for c in enrol_df.columns if 'age_' in c and 'demo' not in c and 'bio' not in c]
        d_cols = [c for c in demo_df.columns if 'demo_age' in c]
        b_cols = [c for c in bio_df.columns if 'bio_age' in c]

        val_enrol = f_enrol[e_cols].sum().sum()
        val_demo = f_demo[d_cols].sum().sum() if not f_demo.empty else 0
        val_bio = f_bio[b_cols].sum().sum() if not f_bio.empty else 0

        # Core Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("New Enrolments", f"{int(val_enrol):,}")
        with m2: st.metric("Demographic Updates", f"{int(val_demo):,}")
        with m3: st.metric("Biometric Updates", f"{int(val_bio):,}")
        with m4: st.metric("Total Service Interactions", f"{int(val_enrol + val_demo + val_bio):,}")

        st.markdown("---")

        # --- 10. ANALYSIS MODULES ---
        tab_overview, tab_ai, tab_geo, tab_explorer = st.tabs(["Overview", "Forecasting", "Geospatial", "Data Explorer"])

        with tab_overview:
            st.subheader("Automated Intelligence Summary")
            peak_day = "Data Unavailable"
            if 'date' in f_enrol.columns and not f_enrol.empty:
                daily_sum = f_enrol.groupby('date')[e_cols].sum().sum(axis=1)
                peak_day = daily_sum.idxmax().strftime('%d %b, %Y')
                avg_load = daily_sum.mean()
                # Generating dynamic English summary
                st.markdown(f"""
                <div class="insight-box">
                    <b>Summary Analysis:</b> Within the region of <b>{sel_state}</b>, the highest activity volume was recorded on <b>{peak_day}</b>. <br>
                    The average daily service interaction volume is approximately <b>{int(avg_load)}</b> units. <br>
                    Information update requests constitute <b>{int((val_demo+val_bio)/(val_enrol+val_demo+val_bio+1)*100)}%</b> of the total operational workload.
                </div>
                """, unsafe_allow_html=True)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Temporal Activity Trend")
                if 'date' in f_enrol.columns:
                    trend = f_enrol.groupby('date')[e_cols].sum().sum(axis=1).reset_index()
                    trend.columns = ['Date', 'Volume']
                    fig = px.area(trend, x='Date', y='Volume', color_discrete_sequence=['#003366'], template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Demographic Mix")
                age_sum = f_enrol[e_cols].sum().reset_index()
                age_sum.columns = ['Age Group', 'Volume']
                fig_pie = px.pie(age_sum, names='Age Group', values='Volume', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab_ai:
            st.subheader("Predictive Workload Analysis")
            if 'date' in f_enrol.columns and not f_enrol.empty:
                ai_data = f_enrol.groupby('date')[e_cols].sum().sum(axis=1).reset_index()
                ai_data.columns = ['Date', 'Y']
                # Normalizing dates for mathematical processing
                ai_data['X'] = (ai_data['Date'] - ai_data['Date'].min()).dt.days
                if len(ai_data) > 2:
                    # Applying Linear Regression for trend extrapolation
                    z = np.polyfit(ai_data['X'], ai_data['Y'], 1)
                    p = np.poly1d(z)
                    # Calculating 30-day projection
                    prediction = max(0, int(p(ai_data['X'].max() + 30)))
                    pc1, pc2 = st.columns(2)
                    pc1.success(f"### 30-Day Forecasted Volume: **{prediction:,}** Service Interactions")
                    velocity = "Positive" if z[0] > 0 else "Negative"
                    pc2.metric("Trend Velocity", f"{velocity} Trend", f"{z[0]:.2f} units/day")
                    
                    # Forecast Visualization
                    f_dates = pd.date_range(start=ai_data['Date'].max(), periods=30)
                    f_vals = [p(ai_data['X'].max() + i) for i in range(30)]
                    fig_ai = go.Figure()
                    fig_ai.add_trace(go.Scatter(x=ai_data['Date'], y=ai_data['Y'], name='Historical Volume'))
                    fig_ai.add_trace(go.Scatter(x=f_dates, y=f_vals, name='AI Projection', line=dict(dash='dash', color='orange')))
                    fig_ai.update_layout(title="Activity Projections (Next 30 Days)", template="plotly_white")
                    st.plotly_chart(fig_ai, use_container_width=True)
                else:
                    st.warning("Prediction engine requires more historical data points to generate an accurate forecast.")

        with tab_geo:
            st.subheader("Geographical Workload Distribution")
            st.info("System Note: Select a specific State or District from the sidebar to refine spatial insights.")
            
            if sel_state == "All India":
                state_data = enrol_df.groupby('state')[e_cols].sum().sum(axis=1).reset_index(name='Total')
                # Standardizing state names for GeoJSON compatibility
                state_data['map_name'] = state_data['state'].replace({
                    'Andaman And Nicobar Islands': 'Andaman & Nicobar',
                    'Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli',
                    'Jammu And Kashmir': 'Jammu & Kashmir'
                })
                # Reliable India States GeoJSON source
                india_geojson_url = "https://raw.githubusercontent.com/jbrobst/56c13bbbf9d97d106f5929d67a182710/raw/8ddc5d1421712952402173f406456079d35bb8/india_states.json"
                
                try:
                    fig_map = px.choropleth(
                        state_data, geojson=india_geojson_url, featureidkey="properties.ST_NM",
                        locations="map_name", color="Total", color_continuous_scale="Viridis",
                        projection="mercator"
                    )
                    fig_map.update_geos(fitbounds="locations", visible=False)
                    fig_map.update_traces(marker_line_width=1, marker_line_color="white")
                    st.plotly_chart(fig_map, use_container_width=True)
                except:
                    st.warning("Spatial engine currently offline. Displaying hierarchical load distribution as a fallback.")
                    fig_fallback = px.treemap(state_data, path=['state'], values='Total', color='Total', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_fallback, use_container_width=True)
            else:
                dist_data = s_enrol.groupby('district')[e_cols].sum().sum(axis=1).reset_index(name='Total')
                fig_tree = px.treemap(dist_data, path=['district'], values='Total', color='Total', color_continuous_scale='Viridis')
                st.plotly_chart(fig_tree, use_container_width=True)

            # --- DISTRICT COMPARISON TOOL ---
            st.divider()
            st.subheader("Comparative District Analysis")
            if sel_state != "All India":
                st.write("Perform a side-by-side volumetric comparison between two districts:")
                geo_data = s_enrol.groupby('district')[e_cols].sum().sum(axis=1).reset_index(name='Total')
                
                col_a, col_b = st.columns(2)
                d_list = sorted(geo_data['district'].unique())
                district_a = col_a.selectbox("Reference District", d_list, key="comp_a")
                district_b = col_b.selectbox("Comparison District", d_list, index=min(1, len(d_list)-1), key="comp_b")
                
                comp_data = geo_data[geo_data['district'].isin([district_a, district_b])]
                fig_comp = px.bar(
                    comp_data, 
                    x='district', 
                    y='Total', 
                    color='district', 
                    text_auto='.2s',
                    title=f"Volumetric Comparison: {district_a} vs {district_b}",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:

                st.warning("Enable the District Comparison Tool by selecting a specific State in the controls.")

        with tab_explorer:
            st.subheader("Raw Data Explorer")
            st.write("Filter and explore the granular service records below.")
            display_df = f_enrol.copy()
            cols_to_show = ['date', 'state', 'district', 'pincode'] + e_cols
            available_cols = [c for c in cols_to_show if c in display_df.columns]
            st.dataframe(display_df[available_cols].head(100), use_container_width=True)
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("Export Dataset to CSV", data=csv, file_name="aadhaar_data_export.csv", mime="text/csv")

        st.divider()
        st.caption("Aadhaar 360 v4.4 | Secure AI Analytical Portal | UIDAI Hackathon 2026 | Developved & Maintained By Team Rapid Innovators ")

    else:
        st.error("Critical System Error: Database files missing. Verify the 'data/' directory contains 'enrolment.csv', 'demographic.csv', and 'biometric.csv'.")
