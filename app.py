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
        pos_hits = sum(rev_lower.count(k) for k in pos_keywords)
        
        # Calculate random context base adjusted by raw keyword hits
        score = random.uniform(0.3, 0.9) if pos_hits >= neg_hits else random.uniform(0.1, 0.5)
        if neg_hits > pos_hits:
            sentiment = "Negative"
        elif pos_hits > neg_hits:
            sentiment = "Positive"
        else:
            sentiment = "Neutral" if random.random() > 0.4 else ("Positive" if random.random() > 0.5 else "Negative")
            
        # Aspect Mapping Generation
        aspect_scores = {}
        for asp in aspects:
            aspect_scores[asp] = random.choice(["Positive", "Negative", "Neutral"])
            
        processed_data.append({
            "Review": rev,
            "Sentiment": sentiment,
            "Score": round(score * 5, 1), # Adjusted to a 5-star standard scale
            "Aspects": aspect_scores
        })
    return pd.DataFrame(processed_data)

# ==========================================
# 3. INTERACTIVE SIDEBAR & CONTROL PANEL
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: #8B5CF6;'>SENTIMENX AI</h2>", unsafe_allowed_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8rem; color: #94A3B8;'>Enterprise Feedback Optimization System</p>", unsafe_allowed_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Navigation Hub", ["📊 Data Deck & Analytics", "🎯 Executive Strategy Center"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Preprocessing Filters")
remove_emojis = st.sidebar.checkbox("Strip Emojis & Special Characters", value=True)
remove_stopwords = st.sidebar.checkbox("Filter Standard Stopwords", value=True)

# Sample Data Archetypes for One-Click Testing
sample_electronics = [
    "The battery life is amazing, easily lasts two days! But the screen interface has a noticeable lag.",
    "Worst purchase ever. Charging brick broke on day three. Do not buy this garbage.",
    "Very clean UI/UX design, performance is blazing fast. Highly recommended product vertical.",
    "Hardware quality feels premium, but customer support was incredibly slow to issue my exchange invoice.",
    "Average phone. Battery drains fast during games but normal use is completely acceptable.",
    "The laptop setup wizard crashed three times. Terrible onboarding software experience."
]

sample_apparel = [
    "Fabric is incredibly soft and comfortable. Sizing accuracy was spot on for medium.",
    "Stitching came loose after the first cold machine wash. Highly disappointed with durability.",
    "Beautiful design but sizing runs way too small. Order at least one size up.",
    "Decent comfort level, but the price does not justify this specific fabric quality.",
    "Excellent durability! Worn it to 10+ hikes and it still looks and feels brand new."
]

# ==========================================
# MAIN ROUTING APP ENGINE
# ==========================================

# Initialize Session States to keep calculations static across layout tab clicks
if 'reviews_df' not in st.session_state:
    # Initialize with default Electronics archetype
    cat, aspects = dynamic_categorize_product(sample_electronics)
    st.session_state.product_category = cat
    st.session_state.extracted_aspects = aspects
    st.session_state.reviews_df = analyze_sentiment_and_aspects(sample_electronics, aspects)
    st.session_state.raw_text_input = "\n".join(sample_electronics)

# Title Block
st.markdown("<h1 style='margin-bottom: 0px;'>📊 Enterprise Product Intelligence Console</h1>", unsafe_allowed_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 1.1rem; margin-bottom: 30px;'>Continuous Natural Language Analysis & Strategic Action Suite</p>", unsafe_allowed_html=True)

if app_mode == "📊 Data Deck & Analytics":
    col_input, col_meta = st.columns([2, 1])
    
    with col_input:
        st.markdown("<div class='metric-card'>", unsafe_allowed_html=True)
        st.markdown("### Data Ingestion Interface")
        
        data_source = st.radio("Select Ingestion Channel", ["Raw Text Matrix Paste", "Batch CSV/XLSX Upload"], horizontal=True)
        
        if data_source == "Raw Text Matrix Paste":
            user_text = st.text_area("Paste unstructured target reviews (one entry per line):", 
                                     value=st.session_state.raw_text_input, height=180)
            if st.button("Execute Pipeline Diagnostics"):
                lines = [line.strip() for line in user_text.split("\n") if line.strip()]
                if lines:
                    st.session_state.raw_text_input = user_text
                    cat, aspects = dynamic_categorize_product(lines)
                    st.session_state.product_category = cat
                    st.session_state.extracted_aspects = aspects
                    st.session_state.reviews_df = analyze_sentiment_and_aspects(lines, aspects)
                    st.rerun()
        else:
            uploaded_file = st.file_uploader("Upload review database file", type=["csv", "xlsx"])
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    # Dynamic identification of text column
                    text_col = [col for col in df_upload.columns if df_upload[col].dtype == 'object'][0]
                    lines = df_upload[text_col].dropna().tolist()
                    cat, aspects = dynamic_categorize_product(lines)
                    st.session_state.product_category = cat
                    st.session_state.extracted_aspects = aspects
                    st.session_state.reviews_df = analyze_sentiment_and_aspects(lines, aspects)
                    st.success(f"Successfully cataloged {len(lines)} structured items.")
                except Exception as e:
                    st.error(f"Ingestion Fault parsed: {str(e)}")
        st.markdown("</div>", unsafe_allowed_html=True)

    with col_meta:
        st.markdown("<div class='metric-card' style='height: 290px;'>", unsafe_allowed_html=True)
        st.markdown("### AI Discovery Metrics")
        st.write("")
        st.markdown(f"**Identified Industrial Category:**")
        st.markdown(f"<span style='color:#8B5CF6; font-size:1.4rem; font-weight:bold;'>{st.session_state.product_category}</span>", unsafe_allowed_html=True)
        st.write("")
        st.markdown("**Dynamically Tracked System Aspects:**")
        for asp in st.session_state.extracted_aspects:
            st.markdown(f"• `{asp}`")
        st.markdown("</div>", unsafe_allowed_html=True)

    st.markdown("---")
    st.markdown("## Distribution Insights Dashboard")
    
    # Calculate Metrics
    df = st.session_state.reviews_df
    total_reviews = len(df)
    pos_count = len(df[df['Sentiment'] == 'Positive'])
    neg_count = len(df[df['Sentiment'] == 'Negative'])
    neu_count = len(df[df['Sentiment'] == 'Neutral'])
    
    # Mathematical Framework for Product Success Rate (PSR)
    pos_ratio = pos_count / total_reviews if total_reviews > 0 else 0
    neg_ratio = neg_count / total_reviews if total_reviews > 0 else 0
    avg_score = df['Score'].mean() if total_reviews > 0 else 0
    
    # Weighted calculation mimicking the custom proposal formula
    raw_psr = ((pos_ratio * 0.6) + ((avg_score / 5) * 0.4)) * 100
    churn_risk_penalty = neg_ratio * 15  # Deduct points penalty for intensive negative clustering
    final_psr = max(min(round(raw_psr - churn_risk_penalty, 1), 100), 0)

    # Layout Metrics Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><p style='color:#94A3B8; margin:0;'>Total Volume</p><h2>{total_reviews} Records</h2></div>", unsafe_allowed_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><p style='color:#34D399; margin:0;'>Positive Feedback</p><h2 style='color:#34D399 !important;'>{pos_count} ({round(pos_ratio*100)}%)</h2></div>", unsafe_allowed_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><p style='color:#F87171; margin:0;'>Negative Vulnerabilities</p><h2 style='color:#F87171 !important;'>{neg_count}</h2></div>", unsafe_allowed_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><p style='color:#60A5FA; margin:0;'>Aggregated CSAT Score</p><h2>{round((avg_score/5)*100, 1)}%</h2></div>", unsafe_allowed_html=True)

    # Charts Section
    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown("<div class='metric-card'>", unsafe_allowed_html=True)
        st.markdown("### Structural Sentiment Breakdown")
        fig_pie = px.pie(
            names=['Positive', 'Neutral', 'Negative'],
            values=[pos_count, neu_count, neg_count],
            color=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={'Positive': '#10B981', 'Neutral': '#64748B', 'Negative': '#EF4444'},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), backgroundcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E2E8F0')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allowed_html=True)
        
    with g2:
        st.markdown("<div class='metric-card'>", unsafe_allowed_html=True)
        st.markdown("### Calculated Product Success Rate (PSR) Gauge")
        
        # Color profile matching PSR thresholds
        if final_psr >= 85: gauge_color = "#10B981"
        elif final_psr >= 60: gauge_color = "#F59E0B"
        elif final_psr >= 40: gauge_color = "#F97316"
        else: gauge_color = "#EF4444"
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_psr,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#E2E8F0"},
                'bar': {'color': gauge_color},
                'bgcolor': "#334155",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [40, 60], 'color': 'rgba(249, 115, 22, 0.1)'},
                    {'range': [60, 85], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
            }
        ))
        fig_gauge.update_layout(margin=dict(t=40, b=20, l=40, r=40), backgroundcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E2E8F0', height=275)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allowed_html=True)

    # Data Table Preview
    st.markdown("### Processed Production Data")
    st.dataframe(df[['Review', 'Sentiment', 'Score']], use_container_width=True)


elif app_mode == "🎯 Executive Strategy Center":
    df = st.session_state.reviews_df
    total_reviews = len(df)
    pos_count = len(df[df['Sentiment'] == 'Positive'])
    neg_count = len(df[df['Sentiment'] == 'Negative'])
    avg_score = df['Score'].mean() if total_reviews > 0 else 0
    
    pos_ratio = pos_count / total_reviews if total_reviews > 0 else 0
    neg_ratio = neg_count / total_reviews if total_reviews > 0 else 0
    final_psr = max(min(round((((pos_ratio * 0.6) + ((avg_score / 5) * 0.4)) * 100) - (neg_ratio * 15), 1), 100), 0)

    st.markdown("## Automated Boardroom Decision Matrix")
    
    # 1. C-Suite Final Verdict Core Generator
    st.markdown("<div class='metric-card'>", unsafe_allowed_html=True)
    st.markdown("### Core Operational Verdict")
    
    if final_psr >= 85:
        verdict_badge = "<span class='badge-leader'>🟢 MARKET LEADER (PSR > 85%)</span>"
        verdict_text = "High relative user satisfaction with minor decoupled technical friction. Recommendation profiles state that corporate structures must aggressively scale production lines, raise target acquisition spend thresholds, and maximize systemic profit margin frameworks."
    elif final_psr >= 60:
        verdict_badge = "<span class='badge-monitor'>🟡 MONITOR & REWORK (PSR 60%–84%)</span>"
        verdict_text = "The product processes a stable customer core validation framework, but overall success velocity is pulled down heavily by highly isolated, fixable friction vectors. Recommended actions: temporarily halt expansion marketing and deploy patch optimizations to the specific defect vectors identified."
    elif final_psr >= 40:
        verdict_badge = "<span class='badge-pivot'>🟠 PIVOT REQUIRED (PSR 40%–59%)</span>"
        verdict_text = "Severe negative sentiment mapping relative to core product utility models. System features run a critical baseline retention risk. R&D must initiate an immediate audience re-alignment or core workflow layout overhaul."
    else:
        verdict_badge = "<span class='badge-recall'>🔴 PRODUCT OBSOLESCENCE / RECALL (PSR < 40%)</span>"
        verdict_text = "Critical system failures, dangerous components, or catastrophic value proposition gaps detected. Corporate risk mandates immediate shipping holds, stock level audits, and systemic replacement updates."

    st.markdown(f"Current Status: {verdict_badge}", unsafe_allowed_html=True)
    st.markdown(f"<p style='font-size:1.1rem; margin-top:15px; color:#CBD5E1;'>{verdict_text}</p>", unsafe_allowed_html=True)
    st.markdown("</div>", unsafe_allowed_html=True)
    
    # 2. Aspect Heatmap matrix
    st.markdown("<div class='metric-card'>", unsafe_allowed_html=True)
    st.markdown("### Aspect Sentiment Heatmap Matrix")
    
    # Gather counts of aspect sentiments
    aspect_matrix_data = []
    for asp in st.session_state.extracted_aspects:
        pos_asp = sum(1 for r in df['Aspects'] if r.get(asp) == 'Positive')
        neg_asp = sum(1 for r in df['Aspects'] if r.get(asp) == 'Negative')
        neu_asp = sum(1 for r in df['Aspects'] if r.get(asp) == 'Neutral')
        aspect_matrix_data.append([asp, 'Positive', pos_asp])
        aspect_matrix_data.append([asp, 'Neutral', neu_asp])
        aspect_matrix_data.append([asp, 'Negative', neg_asp])
        
    m_df = pd.DataFrame(aspect_matrix_data, columns=['Aspect Component', 'Sentiment Context', 'Frequency Score'])
    fig_heat = px.density_heatmap(
        m_df, x="Sentiment Context", y="Aspect Component", z="Frequency Score",
        color_continuous_scale="Purples"
    )
    fig_heat.update_layout(backgroundcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#E2E8F0', height=250)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allowed_html=True)

    # 3. Department-Specific Prescriptive Ticket Generator
    st.markdown("### Departmental Prescriptive Tickets")
    
    # Generating dynamic prescriptive recommendations based on extracted real-time categories
    tickets = []
    if st.session_state.product_category == "Electronics & Tech":
        tickets = [
            {"Dept": "Supply Chain / QA", "Suggestion": "Identify battery cell thermal dissipation rates. Batches indicate localized runtime voltage drops.", "Priority": "🔴 Critical"},
            {"Dept": "Marketing & PR", "Suggestion": "Counter balance negative UI lag complaints by launching campaigns highlighting high-grade materials and customer service speed metrics.", "Priority": "🟡 Medium"},
            {"Dept": "Product Dev (R&D)", "Suggestion": "Refactor foundational initialization blocks in version 2.0 firmware. Users exhibit privacy software bottleneck anxieties.", "Priority": "🟠 High"}
        ]
    elif st.session_state.product_category == "Apparel & Fashion":
        tickets = [
            {"Dept": "Supply Chain / QA", "Suggestion": "Evaluate tensile stitch integrity benchmarks on current manufacturing line weaves to mitigate cold wash structural failure complaints.", "Priority": "🔴 Critical"},
            {"Dept": "Marketing & PR", "Suggestion": "Revise public digital size chart interfaces with interactive dynamic slider dimensions to offset sizing variance reviews.", "Priority": "🟠 High"},
            {"Dept": "Product Dev (R&D)", "Suggestion": "A/B test specialized alternative synthetic blended weaves to optimize fabric soft touch while preserving baseline target retail margins.", "Priority": "🟡 Medium"}
        ]
    else:
        tickets = [
            {"Dept": "Operations / QA", "Suggestion": "Conduct deep audits into entry processing loops to isolate friction patterns recorded in general text records.", "Priority": "🔴 Critical"},
            {"Dept": "Strategic Dev", "Suggestion": "Optimize interface tutorials to increase initial registration and product feature familiarity.", "Priority": "🟡 Medium"}
        ]

    for ticket in tickets:
        with st.expander(f"📥 Ticket Alert for: **{ticket['Dept']}** — {ticket['Priority']}"):
            st.markdown(f"<p style='font-size:1.05rem;'><b>Prescriptive Strategy Formulation:</b> {ticket['Suggestion']}</p>", unsafe_allowed_html=True)