import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Set page configurations
st.set_page_config(
    page_title="App User Behavior Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and premium styling
st.markdown("""
<style>
    /* Gradient Background for headers */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
    }
    .main-header p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metrics Card styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #212529;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Persona Card styling */
    .persona-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .persona-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    if os.path.exists("user_data_with_pca.csv"):
        return pd.read_csv("user_data_with_pca.csv")
    else:
        # Fallback to loading original if clustering not completed
        return pd.read_csv("user_behavior_dataset.csv")

df = load_data()

# Define cluster mapping
cluster_personas = {
    0: {
        "name": "🔇 Silent Users (Non-Raters)",
        "color": "#440154",  # Viridis dark purple
        "desc": "Users who use the app normally but have not provided any ratings (ratings are recorded as 0). They represent a silent group whose satisfaction remains unmeasured.",
        "actions": "Launch targeted push notification or in-app surveys to collect feedback. Offer small rewards or feature unlocks for submitting a rating.",
        "badge": "warning"
    },
    1: {
        "name": "🧭 Active Explorers (High Page Views)",
        "color": "#31688e",  # Viridis blue-grey
        "desc": "Users who click through a very high volume of pages per session (averaging 19.3 pages). They are actively exploring multiple features and search tools.",
        "actions": "Suggest personalized feature shortcuts. Promote related content recommendations. Serve in-app advertisements and contextual premium upgrades.",
        "badge": "info"
    },
    2: {
        "name": "⚡ Focused Browsers (Low Page Views)",
        "color": "#35b779",  # Viridis green
        "desc": "Users who visit the minimum number of pages per session (averaging 7.7 pages). They log in with a specific task in mind and log out immediately.",
        "actions": "Keep the dashboard interface clean and high-performance. Avoid cluttering their path with promotional banners. Offer quick-action widgets.",
        "badge": "success"
    },
    3: {
        "name": "🧘 Immersive Users (Long Session Duration)",
        "color": "#fde725",  # Viridis yellow
        "desc": "Users who spend the longest continuous time in the app per session (averaging 29.1 minutes) while viewing a normal number of pages. They read or interact deeply.",
        "actions": "Promote Premium Subscription tiers. Implement loyalty programs. Send long-form content notifications. Highlight community or social share features.",
        "badge": "error"
    }
}

# App Title Header
st.markdown("""
<div class="main-header">
    <h1>App User Behavior Segmentation Dashboard</h1>
    <p>Unsupervised Machine Learning & PCA Segmentations (k=4)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/activity-feed-value.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Dashboard Overview", "🧬 Segment Visualization (PCA)", "🎯 Targeting & Export"]
)

# Sidebar Info Box
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Configuration")
st.sidebar.markdown("**Algorithm**: K-Means Clustering")
st.sidebar.markdown("**Input Features**: 10 Behavioral Columns")
st.sidebar.markdown("**Optimal Clusters ($k$)**: 4")
st.sidebar.markdown("**Variance Explained (PCA)**: 20.36%")

# PAGE 1: DASHBOARD OVERVIEW
if page == "📊 Dashboard Overview":
    st.header("Overall User Base Stats")
    
    # 4 Column KPI Section
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Users Segmented</div>
            <div class="metric-value">50,000</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Segments</div>
            <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg Engagement Score</div>
            <div class="metric-value">64.94 / 100</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg Churn Risk</div>
            <div class="metric-value">50.08%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("User Distribution by Segment")
        cluster_counts = df['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'count']
        cluster_counts['Segment Name'] = cluster_counts['cluster'].map(lambda x: cluster_personas[x]['name'])
        cluster_counts['Color'] = cluster_counts['cluster'].map(lambda x: cluster_personas[x]['color'])
        
        fig_pie = px.pie(
            cluster_counts, 
            values='count', 
            names='Segment Name',
            color='Segment Name',
            color_discrete_map={cluster_personas[i]['name']: cluster_personas[i]['color'] for i in range(4)},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, width='stretch')
        
    with col2:
        st.subheader("Behavioral Performance Matrix")
        
        # Average metrics per cluster
        means = df.groupby('cluster')[['engagement_score', 'churn_risk_score']].mean().reset_index()
        means['churn_risk_score'] = means['churn_risk_score'] * 100  # convert to %
        means['Segment Name'] = means['cluster'].map(lambda x: cluster_personas[x]['name'])
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=means['Segment Name'],
            y=means['engagement_score'],
            name='Avg Engagement Score',
            marker_color='#2a5298'
        ))
        fig_bar.add_trace(go.Bar(
            x=means['Segment Name'],
            y=means['churn_risk_score'],
            name='Avg Churn Risk (%)',
            marker_color='#d9534f'
        ))
        fig_bar.update_layout(
            barmode='group',
            xaxis_title="Segment",
            yaxis_title="Value / Percentage",
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_bar, width='stretch')

# PAGE 2: SEGMENT VISUALIZATION (PCA)
elif page == "🧬 Segment Visualization (PCA)":
    st.header("Dimensionality Reduction & Cluster Separation")
    
    st.markdown("""
    This scatter plot shows the users projected from **10 behavioral dimensions** down to a 2D space using **Principal Component Analysis (PCA)**. 
    Hover over the points to see details. Note that since we have 50,000 points, we display a random sample of 2,000 points to keep the visualization fast and responsive.
    """)
    
    # Sample data for fast plotting
    df_sample = df.sample(2000, random_state=42)
    df_sample['Segment Name'] = df_sample['cluster'].map(lambda x: cluster_personas[x]['name'])
    
    fig_pca = px.scatter(
        df_sample,
        x='pca_1',
        y='pca_2',
        color='Segment Name',
        color_discrete_map={cluster_personas[i]['name']: cluster_personas[i]['color'] for i in range(4)},
        hover_data={
            'user_id': True,
            'sessions_per_week': True,
            'avg_session_duration_min': ':.1f',
            'pages_viewed_per_session': True,
            'rating_given': True,
            'engagement_score': ':.2f',
            'pca_1': False,
            'pca_2': False,
            'Segment Name': False
        },
        title="2D Projection of User Segments in PCA Space",
        labels={'pca_1': 'Principal Component 1', 'pca_2': 'Principal Component 2'}
    )
    
    fig_pca.update_layout(
        height=600,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_pca, width='stretch')

    # Statistical Comparison Expander
    with st.expander("📊 Compare Statistical Medians per Segment"):
        cols = [
            'sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes', 
            'feature_clicks_per_session', 'pages_viewed_per_session', 
            'days_since_last_login', 'rating_given', 'engagement_score', 'cluster'
        ]
        medians = df[cols].groupby('cluster').median()
        medians.index = medians.index.map(lambda x: cluster_personas[x]['name'])
        st.dataframe(medians, width='stretch')

# PAGE 3: TARGETING & EXPORT
elif page == "🎯 Targeting & Export":
    st.header("Targeted Segment Analysis & Campaign Export")
    
    # 1. Segment Selector Card
    selected_cluster_id = st.selectbox(
        "Select User Segment to Analyze:",
        options=[0, 1, 2, 3],
        format_func=lambda x: cluster_personas[x]['name']
    )
    
    persona = cluster_personas[selected_cluster_id]
    
    # Display Persona details in styled card
    st.markdown(f"""
    <div class="persona-card">
        <div class="persona-title">{persona['name']}</div>
        <p><strong>Persona Description:</strong> {persona['desc']}</p>
        <p style="margin-bottom:0;"><strong>💡 Recommended Business Strategy:</strong></p>
        <div style="background-color:#eef2f7; padding: 10px; border-left: 4px solid #2a5298; border-radius: 4px; margin-top:5px;">
            {persona['actions']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter dataset for download
    df_segment = df[df['cluster'] == selected_cluster_id].drop(columns=['pca_1', 'pca_2'])
    
    # Export options
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader(f"Total Users in Segment: {len(df_segment):,}")
        csv_data = df_segment.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {persona['name']} List (CSV)",
            data=csv_data,
            file_name=f"targeted_segment_{selected_cluster_id}.csv",
            mime="text/csv",
            key='download-csv'
        )
    with col_b:
        st.subheader("Segment Age & Subscription Tier Profiles")
        # Quick summary stats
        avg_age = df_segment['age'].mean()
        avg_acct = df_segment['account_age_days'].mean()
        st.write(f"**Average User Age**: {avg_age:.1f} years old")
        st.write(f"**Average Account Age**: {avg_acct:.0f} days (~{avg_acct/30.4:.1f} months)")
        
    st.markdown("---")
    
    # 2. Individual User Lookup Tool
    st.subheader("🔍 Individual User Lookup Tool")
    user_search = st.text_input("Enter User ID (e.g., 100000 to 149999):", value="")
    
    if user_search:
        try:
            uid = int(user_search)
            user_row = df[df['user_id'] == uid]
            
            if not user_row.empty:
                assigned_cluster = user_row.iloc[0]['cluster']
                st.success(f"User ID **{uid}** is classified in: **{cluster_personas[assigned_cluster]['name']}**")
                
                # Show key metrics in Columns
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Engagement Score", f"{user_row.iloc[0]['engagement_score']:.2f}")
                with stat_col2:
                    st.metric("Sessions/Week", f"{user_row.iloc[0]['sessions_per_week']}")
                with stat_col3:
                    st.metric("Avg Session Duration", f"{user_row.iloc[0]['avg_session_duration_min']:.1f} min")
                with stat_col4:
                    st.metric("Rating Given", f"{user_row.iloc[0]['rating_given']}")
                
                # Expand to show all details
                with st.expander("View Full User Profile"):
                    st.dataframe(user_row.drop(columns=['pca_1', 'pca_2']), width='stretch')
            else:
                st.warning("User ID not found. Please enter a valid ID between 100000 and 149999.")
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
