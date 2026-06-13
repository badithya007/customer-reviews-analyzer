import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from datetime import datetime

# 1. APPLICATION STRUCTURAL CONFIGURATION
st.set_page_config(
    page_title="Feedback Intel Engine Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. THEME-AGNOSTIC HIGH-CONTRAST CSS CUSTOM OVERRIDES (LIGHT & DARK MODE COMPATIBLE)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force high-contrast text rendering on default elements */
    html, body, [data-testid="stAppViewContainer"], .stMarkdown p {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Typography */
    .main-header {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.025em;
    }
    .sub-header {
        font-size: 16px;
        color: #475569 !important;
        margin-bottom: 35px;
        font-weight: 500;
    }
    
    /* Premium Dashboard Metric Cards (Fixed Theme Color Contrast) */
    .metric-card {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 38px;
        font-weight: 700;
        color: #0F172A !important;
        line-height: 1;
    }
    .metric-label {
        font-size: 13px;
        color: #475569 !important;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Premium Themed Alert Containers with Guaranteed Text Contrast */
    .verdict-box {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }
    
    .action-card {
        background-color: #EFF6FF !important;
        border-left: 6px solid #2563EB !important;
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 4px 12px 12px 4px;
        color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .improvement-card {
        background-color: #FEF3C7 !important;
        border-left: 6px solid #D97706 !important;
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 4px 12px 12px 4px;
        color: #1E293B !important; /* Forces dark charcoal font style regardless of application theme */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .red-flag-card {
        background-color: #FEE2E2 !important;
        border-left: 6px solid #DC2626 !important;
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 4px 12px 12px 4px;
        color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Inner Typography Visibility Configuration */
    .action-card p, .improvement-card p, .red-flag-card p {
        color: #1E293B !important;
    }
    </style>
    """)