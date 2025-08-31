import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import psycopg2
from psycopg2 import sql
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import json
import requests
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any, Optional
import hashlib
from dataclasses import dataclass
import io
import base64

# Advanced imports for oceanographic analysis
from scipy import interpolate, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration with custom theme
st.set_page_config(
    page_title="ARGO Floats AI Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f0f8ff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    "host": "ep-still-field-a17hi4xm-pooler.ap-southeast-1.aws.neon.tech",
    "database": "neondb",
    "user": "neondb_owner",
    "password": "npg_qV9a3dQRAeBm",
    "port": "5432",
    "sslmode": "require"
}

# Reference coordinates for major locations
REFERENCE_LOCATIONS = {
    "Chennai": (13.0827, 80.2707),
    "Mumbai": (19.0760, 72.8777),
    "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639),
    "Visakhapatnam": (17.6868, 83.2185),
    "Bengaluru": (12.9716, 77.5946),
    "Arabian Sea": (15.0, 65.0),
    "Bay of Bengal": (15.0, 87.0),
    "Indian Ocean": (-20.0, 80.0),
    "Equator": (0.0, 80.0),
    "Maldives": (3.2028, 73.2207),
    "Sri Lanka": (7.8731, 80.7718)
}

@dataclass
class OceanographicAnalysis:
    """Class to hold oceanographic analysis results"""
    mixed_layer_depth: Optional[float] = None
    thermocline_depth: Optional[float] = None
    halocline_depth: Optional[float] = None
    temperature_gradient: Optional[float] = None
    salinity_gradient: Optional[float] = None
    water_mass_classification: Optional[str] = None

class ArgoAIIntelligence:
    def __init__(self):
        self.db_connection = None
        
        # Enhanced schema with oceanographic context
        self.schema_info = """
        ARGO Floats Database Schema:
        
        Table: argo_floats
        Core Columns:
        - platform_number (integer): Unique ARGO float identifier
        - cycle_number (integer): Profile cycle number
        - measurement_time (timestamp): UTC measurement time
        - latitude (float): Decimal degrees (-90 to 90)
        - longitude (float): Decimal degrees (-180 to 180)
        - pressure (float): Water pressure in decibars (dbar)
        - temperature (float): In-situ temperature in Celsius
        - salinity (float): Practical Salinity Units (PSU)
        - data_quality (text): Quality control flag
        
        Oceanographic Context:
        - Indian Ocean coverage with 85 active floats
        - Depth profiles from surface to ~2000m
        - Temperature range: ~15-35°C
        - Salinity range: ~33-37 PSU
        - Mixed layer depth typically 20-100m
        - Thermocline at 100-500m depth
        """

    def connect_db(self) -> bool:
        """Establish database connection"""
        try:
            if self.db_connection is None or self.db_connection.closed:
                self.db_connection = psycopg2.connect(**DB_CONFIG)
            return True
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return False

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            if self.connect_db():
                df = pd.read_sql_query(query, self.db_connection)
                return df
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
            return pd.DataFrame()

    def advanced_nlp_to_sql(self, user_query: str, api_key: str = None) -> Dict[str, Any]:
        """Advanced natural language to SQL with pattern matching"""
        
        # Enhanced pattern matching with oceanographic domain knowledge
        patterns = {
            # Location-based queries
            r"(?:nearest|closest|near)\s+(?:to\s+)?(.+)": self._handle_location_query,
            r"floats?\s+(?:in|around|near)\s+(.+)": self._handle_area_query,
            
            # Oceanographic analysis
            r"(?:t-s|temperature.salinity)\s+diagram": self._handle_ts_diagram,
            r"(?:mixed\s+layer|mld)\s+depth": self._handle_mixed_layer_depth,
            r"(?:depth\s+profile|vertical\s+profile)": self._handle_depth_profile,
            r"(?:thermocline|halocline)": self._handle_cline_analysis,
            r"water\s+mass": self._handle_water_mass,
            
            # Temporal queries
            r"(?:seasonal|monthly)\s+(?:trend|cycle|variation)": self._handle_seasonal_analysis,
            r"(?:recent|latest|last\s+\d+)\s+(?:days?|weeks?|months?)": self._handle_recent_data,
            r"(?:compare|comparison)\s+.*(?:between|vs)": self._handle_comparison,
            
            # Parameter-based queries
            r"temperature\s+(?:above|greater than|>)\s+([\d.]+)": self._handle_temp_threshold,
            r"temperature\s+(?:below|less than|<)\s+([\d.]+)": self._handle_temp_threshold,
            r"salinity\s+(?:above|greater than|>)\s+([\d.]+)": self._handle_sal_threshold,
            r"salinity\s+(?:below|less than|<)\s+([\d.]+)": self._handle_sal_threshold,
            
            # Statistical queries
            r"(?:statistics|stats|summary)": self._handle_statistics,
            r"(?:correlation|relationship)\s+between": self._handle_correlation,
            
            # Default patterns
            r"(?:show|display|get|find)\s+all\s+floats?": lambda x: self._get_all_floats_query(),
            r"(?:count|how many)\s+floats?": lambda x: "SELECT COUNT(DISTINCT platform_number) as total_floats FROM argo_floats;",
        }
        
        query_lower = user_query.lower().strip()
        sql_query = None
        analysis_type = "general"
        
        # Try pattern matching
        for pattern, handler in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                if callable(handler):
                    result = handler(match.groups()[0] if match.groups() else None)
                    if isinstance(result, dict):
                        sql_query = result.get('sql', '')
                        analysis_type = result.get('type', 'general')
                    else:
                        sql_query = result
                        analysis_type = pattern.split(r"\s+")[0] if r"\s+" in pattern else "general"
                break
        
        # Fallback for API-based generation
        if not sql_query and api_key:
            sql_query = self._generate_with_openai(user_query, api_key)
        
        # Final fallback
        if not sql_query:
            sql_query = "SELECT * FROM argo_floats ORDER BY measurement_time DESC LIMIT 20;"
            analysis_type = "general"
        
        return {
            "sql": sql_query,
            "analysis_type": analysis_type,
            "query_interpretation": self._interpret_query(user_query, analysis_type)
        }

    def _handle_location_query(self, location: str) -> Dict[str, Any]:
        """Handle location-based queries with distance calculation"""
        location_clean = location.strip().title()
        
        if location_clean in REFERENCE_LOCATIONS:
            lat, lon = REFERENCE_LOCATIONS[location_clean]
            sql = f"""
            SELECT 
                platform_number, 
                latitude, 
                longitude, 
                temperature, 
                salinity,
                pressure,
                measurement_time,
                SQRT(POW(latitude - {lat}, 2) + POW(longitude - {lon}, 2)) * 111 as distance_km
            FROM argo_floats 
            ORDER BY distance_km ASC 
            LIMIT 20;
            """
            return {"sql": sql, "type": "location_analysis"}
        
        return {"sql": "SELECT * FROM argo_floats LIMIT 20;", "type": "general"}

    def _handle_area_query(self, location: str) -> Dict[str, Any]:
        """Handle area-based queries"""
        return self._handle_location_query(location)

    def _handle_ts_diagram(self, _) -> Dict[str, Any]:
        """Handle T-S diagram requests"""
        sql = """
        SELECT 
            platform_number,
            temperature,
            salinity,
            pressure,
            latitude,
            longitude,
            measurement_time
        FROM argo_floats 
        WHERE temperature IS NOT NULL 
        AND salinity IS NOT NULL
        ORDER BY platform_number, pressure;
        """
        return {"sql": sql, "type": "ts_diagram"}

    def _handle_mixed_layer_depth(self, _) -> Dict[str, Any]:
        """Handle mixed layer depth analysis"""
        sql = """
        WITH depth_profiles AS (
            SELECT 
                platform_number,
                cycle_number,
                pressure as depth,
                temperature,
                salinity,
                ROW_NUMBER() OVER (PARTITION BY platform_number, cycle_number ORDER BY pressure) as depth_rank
            FROM argo_floats
            WHERE pressure <= 200  -- Focus on upper ocean
            AND temperature IS NOT NULL
        )
        SELECT * FROM depth_profiles
        ORDER BY platform_number, cycle_number, depth;
        """
        return {"sql": sql, "type": "mixed_layer_depth"}

    def _handle_depth_profile(self, _) -> Dict[str, Any]:
        """Handle depth profile visualization"""
        sql = """
        SELECT 
            platform_number,
            cycle_number,
            pressure as depth,
            temperature,
            salinity,
            measurement_time,
            latitude,
            longitude
        FROM argo_floats
        ORDER BY platform_number, cycle_number, pressure;
        """
        return {"sql": sql, "type": "depth_profile"}

    def _handle_cline_analysis(self, _) -> Dict[str, Any]:
        """Handle thermocline/halocline analysis"""
        return self._handle_depth_profile(_)

    def _handle_water_mass(self, _) -> Dict[str, Any]:
        """Handle water mass identification"""
        return self._handle_ts_diagram(_)

    def _handle_seasonal_analysis(self, _) -> Dict[str, Any]:
        """Handle seasonal trend analysis"""
        sql = """
        SELECT 
            EXTRACT(MONTH FROM measurement_time) as month,
            EXTRACT(YEAR FROM measurement_time) as year,
            platform_number,
            AVG(temperature) as avg_temperature,
            AVG(salinity) as avg_salinity,
            COUNT(*) as measurements,
            AVG(latitude) as avg_lat,
            AVG(longitude) as avg_lon
        FROM argo_floats
        GROUP BY year, month, platform_number
        ORDER BY year, month, platform_number;
        """
        return {"sql": sql, "type": "seasonal_analysis"}

    def _handle_recent_data(self, _) -> Dict[str, Any]:
        """Handle recent data queries"""
        sql = """
        SELECT * FROM argo_floats 
        WHERE measurement_time >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY measurement_time DESC
        LIMIT 50;
        """
        return {"sql": sql, "type": "recent_data"}

    def _handle_comparison(self, _) -> Dict[str, Any]:
        """Handle comparison queries"""
        return self._handle_seasonal_analysis(_)

    def _handle_temp_threshold(self, temp: str) -> Dict[str, Any]:
        """Handle temperature threshold queries"""
        try:
            temp_val = float(temp)
            sql = f"SELECT * FROM argo_floats WHERE temperature > {temp_val} ORDER BY temperature DESC LIMIT 50;"
            return {"sql": sql, "type": "temperature_analysis"}
        except:
            return {"sql": "SELECT * FROM argo_floats WHERE temperature > 25 ORDER BY temperature DESC LIMIT 50;", "type": "temperature_analysis"}

    def _handle_sal_threshold(self, sal: str) -> Dict[str, Any]:
        """Handle salinity threshold queries"""
        try:
            sal_val = float(sal)
            sql = f"SELECT * FROM argo_floats WHERE salinity > {sal_val} ORDER BY salinity DESC LIMIT 50;"
            return {"sql": sql, "type": "salinity_analysis"}
        except:
            return {"sql": "SELECT * FROM argo_floats WHERE salinity > 35 ORDER BY salinity DESC LIMIT 50;", "type": "salinity_analysis"}

    def _handle_statistics(self, _) -> Dict[str, Any]:
        """Handle statistical summary requests"""
        sql = """
        SELECT 
            COUNT(DISTINCT platform_number) as unique_floats,
            COUNT(*) as total_measurements,
            AVG(temperature) as mean_temperature,
            STDDEV(temperature) as std_temperature,
            MIN(temperature) as min_temperature,
            MAX(temperature) as max_temperature,
            AVG(salinity) as mean_salinity,
            STDDEV(salinity) as std_salinity,
            MIN(salinity) as min_salinity,
            MAX(salinity) as max_salinity,
            AVG(pressure) as mean_pressure,
            MAX(pressure) as max_pressure,
            MIN(measurement_time) as earliest_measurement,
            MAX(measurement_time) as latest_measurement
        FROM argo_floats;
        """
        return {"sql": sql, "type": "statistics"}

    def _handle_correlation(self, _) -> Dict[str, Any]:
        """Handle correlation analysis"""
        sql = """
        SELECT temperature, salinity, pressure, latitude, longitude 
        FROM argo_floats 
        WHERE temperature IS NOT NULL AND salinity IS NOT NULL;
        """
        return {"sql": sql, "type": "correlation"}

    def _get_all_floats_query(self) -> str:
        """Get all floats query"""
        return "SELECT DISTINCT platform_number, AVG(latitude) as avg_lat, AVG(longitude) as avg_lon FROM argo_floats GROUP BY platform_number ORDER BY platform_number;"

    def _generate_with_openai(self, query: str, api_key: str) -> str:
        """Generate SQL using OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            You are an expert oceanographer and SQL developer. Generate a PostgreSQL query for ARGO float data.

            Database Schema:
            {self.schema_info}

            User Query: "{query}"

            Generate a SQL query that:
            1. Answers the user's question accurately
            2. Uses proper oceanographic terminology
            3. Includes relevant columns for visualization
            4. Handles NULL values appropriately
            5. Uses appropriate LIMIT clauses for performance

            Return only the SQL query, no explanations.
            """
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                st.warning(f"OpenAI API error: {response.status_code}")
                
        except Exception as e:
            st.warning(f"AI query generation failed: {e}")
        
        return "SELECT * FROM argo_floats ORDER BY measurement_time DESC LIMIT 20;"

    def _interpret_query(self, query: str, analysis_type: str) -> str:
        """Provide human-readable interpretation of the query"""
        interpretations = {
            "location_analysis": f"Finding ARGO floats nearest to the specified location with distance calculations.",
            "ts_diagram": "Generating Temperature-Salinity diagram data for water mass analysis.",
            "mixed_layer_depth": "Analyzing mixed layer depth from temperature and salinity profiles.",
            "depth_profile": "Creating vertical ocean profiles showing temperature and salinity vs depth.",
            "seasonal_analysis": "Examining seasonal trends and cycles in oceanographic parameters.",
            "statistics": "Computing comprehensive statistical summary of all ARGO float measurements.",
            "general": f"Processing general query: {query}"
        }
        
        return interpretations.get(analysis_type, f"Analyzing: {query}")

    def create_ts_diagram(self, df: pd.DataFrame) -> go.Figure:
        """Create Temperature-Salinity diagram"""
        if df.empty or 'temperature' not in df.columns or 'salinity' not in df.columns:
            return go.Figure().add_annotation(text="No T-S data available", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        # Color by depth (pressure)
        if 'pressure' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['salinity'],
                y=df['temperature'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df['pressure'],
                    colorscale='Viridis',
                    colorbar=dict(title="Pressure (dbar)"),
                    opacity=0.7
                ),
                text=df.apply(lambda row: f"Float: {row.get('platform_number', 'N/A')}<br>"
                                        f"Depth: {row.get('pressure', 'N/A')} dbar<br>"
                                        f"T: {row['temperature']:.2f}°C<br>"
                                        f"S: {row['salinity']:.2f} PSU", axis=1),
                hovertemplate="%{text}<extra></extra>",
                name="T-S Data"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df['salinity'],
                y=df['temperature'],
                mode='markers',
                marker=dict(size=4, opacity=0.7),
                name="T-S Data"
            ))
        
        fig.update_layout(
            title="Temperature-Salinity Diagram",
            xaxis_title="Salinity (PSU)",
            yaxis_title="Temperature (°C)",
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig

    def create_depth_profile(self, df: pd.DataFrame) -> go.Figure:
        """Create depth profile visualization"""
        if df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature Profile', 'Salinity Profile'),
            shared_yaxes=True
        )
        
        # Group by float and cycle
        colors = px.colors.qualitative.Set3
        color_idx = 0
        
        for (platform, cycle), group in df.groupby(['platform_number', 'cycle_number']):
            if color_idx > 20:  # Limit number of profiles for readability
                break
                
            group = group.sort_values('pressure')
            color = colors[color_idx % len(colors)]
            
            if 'temperature' in group.columns:
                fig.add_trace(
                    go.Scatter(
                        x=group['temperature'],
                        y=-group['pressure'],  # Negative for depth
                        mode='lines+markers',
                        name=f"Float {platform} Cycle {cycle}",
                        line=dict(color=color),
                        showlegend=(color_idx < 10)  # Only show legend for first 10
                    ),
                    row=1, col=1
                )
            
            if 'salinity' in group.columns:
                fig.add_trace(
                    go.Scatter(
                        x=group['salinity'],
                        y=-group['pressure'],
                        mode='lines+markers',
                        name=f"Float {platform} Cycle {cycle}",
                        line=dict(color=color),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            color_idx += 1
        
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
        fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
        
        fig.update_layout(
            title="Ocean Depth Profiles",
            height=600,
            hovermode='closest'
        )
        
        return fig

    def create_advanced_map(self, df: pd.DataFrame, analysis_type: str = "general") -> folium.Map:
        """Create advanced interactive map"""
        if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
            return None
        
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB Positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB Dark_Matter', name='Dark Map').add_to(m)
        
        # Add markers
        for idx, row in df.iterrows():
            popup_content = f"""
            <div style="width: 200px;">
                <h4>ARGO Float {row['platform_number']}</h4>
                <p><b>Location:</b> {row['latitude']:.4f}°N, {row['longitude']:.4f}°E</p>
                <p><b>Temperature:</b> {row.get('temperature', 'N/A')}°C</p>
                <p><b>Salinity:</b> {row.get('salinity', 'N/A')} PSU</p>
                <p><b>Depth:</b> {row.get('pressure', 'N/A')} dbar</p>
                <p><b>Time:</b> {str(row.get('measurement_time', 'N/A'))}</p>
            </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='blue', icon='tint')
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

def create_chat_interface():
    """Create the main chat interface"""
    st.markdown('<h1 class="main-header">🌊 ARGO Floats AI Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced oceanographic analysis powered by AI**")
    
    # Initialize the AI assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ArgoAIIntelligence()
    
    assistant = st.session_state.assistant
    
    # Sidebar with configuration and samples
    with st.sidebar:
        st.header("🤖 AI Configuration")
        
        # API Configuration
        with st.expander("🔑 API Settings", expanded=False):
            openai_key = st.text_input("OpenAI API Key", type="password", help="Optional for advanced AI features")
            st.markdown("**Free Alternatives Available:**")
            st.markdown("- ✅ Built-in NLP patterns")
            st.markdown("- 🤗 HuggingFace models")
            st.markdown("- 🦙 Ollama (local)")
        
        st.divider()
        
        # Sample Queries by Category
        st.header("📊 Query Examples")
        
        sample_categories = {
            "🗺️ Spatial Analysis": [
                "Show floats nearest to Chennai",
                "Floats in Arabian Sea region",
                "Find floats near Maldives"
            ],
            "🌡️ Temperature Analysis": [
                "Temperature above 28 degrees",
                "Show warmest locations",
                "Temperature statistics"
            ],
            "🌊 Oceanographic": [
                "T-S diagram",
                "Mixed layer depth",
                "Depth profiles",
                "Water mass analysis"
            ],
            "📈 Temporal Analysis": [
                "Seasonal temperature trends",
                "Recent measurements",
                "Monthly variations 2024"
            ]
        }
        
        for category, queries in sample_categories.items():
            with st.expander(category):
                for query in queries:
                    if st.button(query, key=f"sample_{query.replace(' ', '_')}", use_container_width=True):
                        st.session_state.user_input = query
                        st.rerun()

    # Main interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "🗣️ Ask about ARGO floats:",
            value=st.session_state.get('user_input', ''),
            placeholder="e.g., 'Show T-S diagram for floats near Chennai' or 'Temperature above 25 degrees'"
        )
    
    with col2:
        search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Process query
    if search_clicked and user_input:
        with st.spinner("🧠 Processing with AI..."):
            # Generate enhanced query
            query_result = assistant.advanced_nlp_to_sql(user_input, openai_key)
            sql_query = query_result['sql']
            analysis_type = query_result['analysis_type']
            interpretation = query_result['query_interpretation']
            
            # Execute query
            df = assistant.execute_query(sql_query)
            
            # Store in session state
            if not df.empty:
                st.session_state.update({
                    'last_query': sql_query,
                    'last_results': df,
                    'last_user_query': user_input,
                    'analysis_type': analysis_type,
                    'interpretation': interpretation
                })
                st.rerun()
    
    # Display results
    if 'last_results' in st.session_state and not st.session_state.last_results.empty:
        df = st.session_state.last_results
        sql_query = st.session_state.last_query
        user_query = st.session_state.last_user_query
        analysis_type = st.session_state.analysis_type
        interpretation = st.session_state.interpretation
        
        # Results header
        st.success(f"✅ Analysis complete! Found {len(df)} records")
        
        # Query interpretation
        st.markdown(f'<div class="chat-message"><b>🧠 AI Interpretation:</b> {interpretation}</div>', 
                   unsafe_allow_html=True)
        
        # SQL query display
        with st.expander("📝 Generated SQL Query"):
            st.code(sql_query, language="sql")
        
        # Create appropriate tabs based on analysis type
        if analysis_type == "ts_diagram":
            tabs = st.tabs(["🌊 T-S Diagram", "🗺️ Map View", "📊 Data Table", "💾 Export"])
            
            with tabs[0]:  # T-S Diagram
                st.subheader("🌊 Temperature-Salinity Diagram")
                ts_fig = assistant.create_ts_diagram(df)
                st.plotly_chart(ts_fig, use_container_width=True)
                
                # T-S Statistics
                if 'temperature' in df.columns and 'salinity' in df.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Temperature Range", f"{df['temperature'].min():.1f} - {df['temperature'].max():.1f}°C")
                    with col2:
                        st.metric("Salinity Range", f"{df['salinity'].min():.2f} - {df['salinity'].max():.2f} PSU")
                    with col3:
                        corr = df['temperature'].corr(df['salinity']) if len(df) > 1 else 0
                        st.metric("T-S Correlation", f"{corr:.3f}")
                        
        elif analysis_type == "depth_profile":
            tabs = st.tabs(["📊 Depth Profiles", "🗺️ Map View", "📋 Data Table", "💾 Export"])
            
            with tabs[0]:  # Depth Profiles
                st.subheader("📊 Ocean Depth Profiles")
                profile_fig = assistant.create_depth_profile(df)
                st.plotly_chart(profile_fig, use_container_width=True)
                
        elif analysis_type == "location_analysis":
            tabs = st.tabs(["🗺️ Location Analysis", "📏 Distance Results", "📊 Data Table", "💾 Export"])
            
            with tabs[1]:  # Distance Results
                if 'distance_km' in df.columns:
                    st.subheader("📏 Distance Analysis")
                    st.dataframe(df[['platform_number', 'latitude', 'longitude', 'distance_km', 'temperature', 'salinity']].head(10))
                    st.metric("Nearest Float Distance", f"{df['distance_km'].min():.1f} km")
                    
        else:
            tabs = st.tabs(["🗺️ Map View", "📊 Data Analysis", "📋 Data Table", "💾 Export"])
        
        # Map View (always present)
        map_tab_idx = 0 if analysis_type not in ["ts_diagram", "depth_profile"] else 1
        
        with tabs[map_tab_idx]:
            st.subheader("🗺️ Geographic Distribution")
            if 'latitude' in df.columns and 'longitude' in df.columns:
                map_obj = assistant.create_advanced_map(df, analysis_type)
                if map_obj:
                    st_folium(map_obj, width=None, height=600)
                
                # Geographic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Latitude Range", f"{df['latitude'].min():.2f}° - {df['latitude'].max():.2f}°")
                with col2:
                    st.metric("Longitude Range", f"{df['longitude'].min():.2f}° - {df['longitude'].max():.2f}°")
                with col3:
                    unique_floats = df['platform_number'].nunique() if 'platform_number' in df.columns else len(df)
                    st.metric("Unique Floats", unique_floats)
                with col4:
                    if 'distance_km' in df.columns:
                        st.metric("Nearest Distance", f"{df['distance_km'].min():.1f} km")
        
        # Data Table
        data_tab_idx = -2
        with tabs[data_tab_idx]:
            st.subheader("📋 Data Table")
            
            # Add basic filtering for large datasets
            if len(df) > 100:
                show_limit = st.selectbox("Show rows:", [100, 500, "All"], index=0)
                display_df = df.head(show_limit) if show_limit != "All" else df
            else:
                display_df = df
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Data summary metrics
            st.subheader("📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                if 'platform_number' in df.columns:
                    st.metric("Unique Floats", df['platform_number'].nunique())
            with col3:
                if 'temperature' in df.columns:
                    st.metric("Avg Temperature", f"{df['temperature'].mean():.2f}°C")
            with col4:
                if 'salinity' in df.columns:
                    st.metric("Avg Salinity", f"{df['salinity'].mean():.2f} PSU")
        
        # Export tab
        with tabs[-1]:
            st.subheader("💾 Export Data & Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"argo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # JSON Export  
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=f"argo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Analysis Report
                temp_range = f"{df['temperature'].min():.2f} - {df['temperature'].max():.2f}°C" if 'temperature' in df.columns else 'N/A'
                sal_range = f"{df['salinity'].min():.2f} - {df['salinity'].max():.2f} PSU" if 'salinity' in df.columns else 'N/A'
                
                report = f"""# ARGO Float Analysis Report

**Query:** {user_query}
**Analysis Type:** {analysis_type}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Records: {len(df)}
- Unique Floats: {df['platform_number'].nunique() if 'platform_number' in df.columns else 'N/A'}
- Temperature Range: {temp_range}
- Salinity Range: {sal_range}

## SQL Query Used
```sql
{sql_query}
```

## Analysis Notes
{interpretation}
"""
                
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=report,
                    file_name=f"argo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    # Footer with system information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 AI Features:**
        - Advanced NLP query processing
        - Pattern-based SQL generation
        - OpenAI integration (optional)
        - Domain-specific oceanographic knowledge
        """)
    
    with col2:
        st.markdown("""
        **🌊 Oceanographic Analysis:**
        - T-S diagrams for water mass ID
        - Depth profile visualization
        - Spatial analysis with distance calc
        - Temporal trend analysis
        """)
    
    with col3:
        st.markdown("""
        **📊 Data & Visualization:**
        - 85 real ARGO floats (Indian Ocean)
        - Interactive maps and charts
        - Professional export options
        - Real-time SQL generation
        """)

def main():
    """Main application entry point"""
    try:
        create_chat_interface()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()