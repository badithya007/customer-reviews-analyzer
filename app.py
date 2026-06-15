import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random

# ==========================================
# 1. ENTERPRISE SAAS GLOBAL CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="SentimenX | Enterprise Product Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for Dark Mode Executive Theme
st.markdown("""
    <style>
    /* Main app background and typography */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    
    /* Global Card styling */
    div.metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    /* Primary buttons */
    .stButton>button {
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    
    /* Status Badges */
    .badge-leader { background-color: #065F46; color: #34D399; padding: 4px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .badge-monitor { background-color: #78350F; color: #FBBF24; padding: 4px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .badge-pivot { background-color: #7C2D12; color: #FB923C; padding: 4px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .badge-recall { background-color: #991B1B; color: #F87171; padding: 4px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    </style>
""", unsafe_allowed_html=True)

# ==========================================
# 2. CORE BACKEND INTELLIGENCE ENGINES
# ==========================================

def dynamic_categorize_product(text_samples):
    """Dynamically checks keywords to auto-classify product vertical."""
    combined_text = " ".join(text_samples).lower()
    if any(k in combined_text for k in ['battery', 'screen', 'phone', 'laptop', 'charging', 'software', 'app']):
        return "Electronics & Tech", ["Battery Life", "UI/UX", "Hardware Quality", "Customer Support"]
    elif any(k in combined_text for k in ['fit', 'fabric', 'size', 'cloth', 'shirt', 'stitching', 'wash']):
        return "Apparel & Fashion", ["Sizing Accuracy", "Fabric Quality", "Durability", "Comfort"]
    elif any(k in combined_text for k in ['taste', 'flavor', 'ingredient', 'expiry', 'bottle', 'food']):
        return "FMCG & Consumables", ["Taste/Flavor", "Packaging Safety", "Freshness", "Value for Money"]
    else:
        return "General SaaS / Service", ["Feature Set", "System Reliability", "Pricing Structure", "Onboarding Ease"]

def analyze_sentiment_and_aspects(reviews, aspects):
    """Simulates advanced NLP sentiment mapping & Aspect-Based Sentiment Analysis."""
    processed_data = []
    
    # Pre-defined keyword weights for demo realism
    neg_keywords = ['bad', 'broken', 'worst', 'terrible', 'slow', 'expensive', 'hate', 'waste', 'returned', 'poor']
    pos_keywords = ['great', 'awesome', 'love', 'perfect', 'amazing', 'fast', 'good', 'excellent', 'best']
    
    for rev in reviews:
        rev_lower = str(rev).lower()
        # Count semantic hits
        neg_hits = sum(rev_lower.count(k) for k in neg_keywords)
        pos_hits = sum