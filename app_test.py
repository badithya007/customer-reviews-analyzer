import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import importlib.util

_sklearn_spec = importlib.util.find_spec("sklearn.feature_extraction.text")
if _sklearn_spec is not None:
    try:
        _sklearn_module = importlib.import_module("sklearn.feature_extraction.text")
        TfidfVectorizer = getattr(_sklearn_module, "TfidfVectorizer", None)
    except Exception:
        TfidfVectorizer = None
else:
    TfidfVectorizer = None

# Flag whether TF-IDF feature extraction is available in the current runtime
tfidf_available = TfidfVectorizer is not None

# Ensure NLTK VADER lexicon is downloaded quietly to prevent deployment logs flooding
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize Core Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# --- Page Layout & Theme Configuration ---
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling Adjustments ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stMetric {background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef;}
    </style>
""", unsafe_allow_html=True)

# --- Title Header ---
st.title("📈 Advanced Customer Feedback Analyzer")
st.markdown("""
*An enterprise-grade system engineered to ingest multi-source product reviews, categorize functional domains, 
extract customer pain-points, deliver executive verdicts, and automate retention communications.*
""")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("📥 Ingestion & Configuration")

product_category = st.sidebar.selectbox(
    "Target Product Category:",
    ["Electronics & Gadgets", "Apparel & Fashion", "Software / SaaS Platforms", "Food & Beverage", "Healthcare & Medical", "Automotive Parts"]
)

input_method = st.sidebar.radio("Data Ingestion Pipeline:", ("Paste Raw Text Reviews", "Upload Dataset (CSV/Excel)"))

reviews_data = []

if input_method == "Paste Raw Text Reviews":
    raw_input = st.sidebar.text_area(
        "Enter reviews (One review per line):",
        height=220,
        placeholder="The device battery drains in under two hours.\nExcellent UI and customer support response!\nShipping took two weeks and the box was slightly damaged."
    )
    if raw_input.strip():
        reviews_data = [line.strip() for line in raw_input.split('\n') if line.strip()]
else:
    uploaded_file = st.sidebar.file_uploader("Upload review spreadsheet:", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            col_target = st.sidebar.selectbox("Select Text Review Column:", df.columns.tolist())
            reviews_data = df[col_target].dropna().astype(str).tolist()
        except Exception as e:
            st.sidebar.error(f"Ingestion Framework Error: {e}")

# --- Core Business Logic Execution ---
if reviews_data:
    total_records = len(reviews_data)
    
    # 1. Processing Sentiment Metrics
    pos_count, neu_count, neg_count = 0, 0, 0
    negative_corpus = []
    
    for review in reviews_data:
        polarity = sia.polarity_scores(review)['compound']
        if polarity >= 0.05:
            pos_count += 1
        elif polarity <= -0.05:
            neg_count += 1
            negative_corpus.append(review)
        else:
            neu_count += 1
            
    # 2. Key Performance Indicator Calculations
    csat_percentage = (pos_count / total_records) * 100 if total_records > 0 else 0
    neg_ratio = (neg_count / total_records) * 100 if total_records > 0 else 0

    # --- KPI Dashboard Row ---
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="Total Ingested Feedback", value=total_records)
    with kpi2:
        st.metric(label="Categorized Product Domain", value=product_category)
    with kpi3:
        st.metric(label="Customer Satisfaction (CSAT)", value=f"{csat_percentage:.1f}%")

    st.markdown("---")

    # --- Charts & Metrics Segmentation ---
    layout_left, layout_right = st.columns([3, 2])
    
    with layout_left:
        st.subheader("📊 Sentiment Vector Distribution")
        chart_df = pd.DataFrame({
            'Sentiment Tier': ['Positive', 'Neutral', 'Negative'],
            'Volume': [pos_count, neu_count, neg_count]
        })
        fig = px.pie(
            chart_df, values='Volume', names='Sentiment Tier',
            color='Sentiment Tier',
            color_discrete_map={'Positive': '#2e7d32', 'Neutral': '#f9a825', 'Negative': '#c62828'},
            hole=0.4
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with layout_right:
        st.subheader("📋 Core Diagnostics")
        st.write(f"🟢 **Positive Sentiment:** {pos_count} ({ (pos_count/total_records)*100:.1f}%)")
        st.write(f"🟡 **Neutral Baseline:** {neu_count} ({ (neu_count/total_records)*100:.1f}%)")
        st.write(f"🔴 **Negative Friction:** {neg_count} ({ (neg_count/total_records)*100:.1f}%)")
        
        if neg_ratio >= 35.0:
            st.error(f"⚠️ **High Operational Risk Flagged:** Negative feedback represents {neg_ratio:.1f}% of current volume. Immediate resource allocation recommended.")
        else:
            st.success("✅ **Stability Bounds Check:** Negative feedback signals remain well within acceptable operational parameters.")

    st.markdown("---")

    # --- Feature Extraction: Most Common Complaints ---
    st.subheader("🔍 Automated Pain-Point Extraction")
    if len(negative_corpus) >= 2:
        if tfidf_available:
            try:
                # Using statistical TF-IDF matrix filtering to identify specific complaint vectors safely
                vectorizer = TfidfVectorizer(stop_words='english', max_features=4, ngram_range=(1,2))
                vectorizer.fit(negative_corpus)
                extracted_features = vectorizer.get_feature_names_out()
                
                st.markdown("Mathematical keyword clustering across negative review nodes indicates core complaints stem around:")
                for feature in extracted_features:
                    st.markdown(f"📍 Critical friction zone detected around: **\"{feature.title()}\"** issues.")
            except Exception:
                st.markdown("ℹ️ *Insufficient vocabulary density across sample nodes to isolate unique keyword vectors.*")
        else:
            st.markdown("ℹ️ *TF-IDF extraction unavailable because scikit-learn is not installed in this environment.*")
    else:
        st.markdown("✅ *Negative sentiment density too low to isolate repeating systemic complaints.*")

    st.markdown("---")

    # --- EXECUTIVE DECISION MATRIX (Final Verdict & Strategic Decisions) ---
    st.subheader("🧠 Executive Summary & Corporate Action Matrix")
    
    # Deterministic analytics evaluation matrix
    if csat_percentage >= 75:
        verdict_status = "STABLE GROWTH / EXCELLENT MARKET POSITIONING"
        verdict_banner = "success"
        verdict_summary = f"The feedback data demonstrates outstanding product resilience inside the **{product_category}** sector. Customer experience metrics show healthy satisfaction levels. Primary retention structures are working well, and product utility remains exceptionally strong."
        strategic_actions = [
            "**Scale Product Visibility:** Turn positive customer feedback highlights into structural marketing assets for new sales funnels.",
            "**Optimize Infrastructure:** Maintain current technical response baselines to avoid support drift as demand spikes.",
            "**Incremental Loyalty Rollouts:** Introduce customer retention milestones to maintain current market dominance."
        ]
        email_tone = "Appreciative & Growth-Oriented"
        email_body = f"Thank you so much for your recent review regarding our **{product_category}** solution! We are absolutely thrilled to hear that your experience has been exceptionally positive. Our product and engineering teams work around the clock to build seamless experiences, and knowing we hit the mark for you inspires us to keep pushing boundaries.\n\nAs a token of our appreciation, your account has been flagged for priority early-access to our upcoming feature pipeline. We value your partnership and look forward to continuing to serve your business operations."
        
    elif 45 <= csat_percentage < 75:
        verdict_status = "MODERATE OPERATIONAL RISK / USER CHURN WARNING"
        verdict_banner = "warning"
        verdict_summary = f"Performance testing reveals clear structural friction points inside the **{product_category}** ecosystem. While the product preserves baseline functional integrity, repeating user friction points endanger critical retention metrics if left unaddressed."
        strategic_actions = [
            "**Targeted Feature Audits:** Cross-verify internal design logs with the specific complaint categories highlighted above.",
            "**Proactive Support Interventions:** Run automated customer success check-ins with users posting neutral or lower scores.",
            "**Patch Deployment Priorities:** Accelerate the release of quality-of-life adjustments to prevent gradual brand erosion."
        ]
        email_tone = "Constructive & Service-Focused"
        email_body = f"Thank you for sharing your candid feedback regarding our **{product_category}** offering. We appreciate your insights, and we hear you loud and clear. We recognize that while parts of your experience met expectations, certain workflows left room for improvement.\n\nOur service engineering groups are already reviewing the specific functional areas you noted to make things right. We want to ensure our platform scales smoothly alongside your business objectives. A support specialist will follow up with you within 24 hours to ensure your immediate operational concerns are completely resolved."
        
    else:
        verdict_status = "CRITICAL BRAND DEFICIT / IMMEDIATE TURNAROUND PROTOCOL"
        verdict_banner = "error"
        verdict_summary = f"Systemic structural performance failure detected inside the **{product_category}** tier. Intense consumer churn is highly probable due to deep performance bugs or broken customer expectations. Immediate product intervention is mandatory."
        strategic_actions = [
            "**Freeze Non-Essential Sprints:** Halt secondary feature pipelines and move developers entirely over to addressing core complaints.",
            "**Executive-Level Account Outreach:** Task customer success leadership with connecting directly with dissatisfied corporate or enterprise accounts.",
            "**Root-Cause Process Re-engineering:** Conduct a complete manufacturing gate or software architecture overhaul to correct systemic quality control escapes."
        ]
        email_tone = "Urgent Mitigation & Resolution-Oriented"
        email_body = f"Thank you for bringing these critical matters to our attention regarding your **{product_category}** experience. We take your feedback incredibly seriously, and we want to apologize sincerely for the operational difficulties this has caused your team.\n\nWe have escalated your feedback directly to our Executive Leadership team. We are currently implementing a comprehensive performance patch to overhaul these specific system vulnerabilities. We want to win back your trust, and a senior engineering manager will contact you directly today to share our precise mitigation roadmap and provide immediate assistance."

    # Render Final Verdict Container
    st.markdown(f"#### **Final Verdict:**")
    if verdict_banner == "success":
        st.success(f"🏆 **{verdict_status}**\n\n{verdict_summary}")
    elif verdict_banner == "warning":
        st.warning(f"⚠️ **{verdict_status}**\n\n{verdict_summary}")
    else:
        st.error(f"🚨 **{verdict_status}**\n\n{verdict_summary}")

    # Render Strategic Decisions
    st.markdown("#### **Prioritized Suggestions for Future Strategic Decisions:**")
    for action in strategic_actions:
        st.markdown(f"- {action}")

    st.markdown("---")

    # --- AUTOMATED THANK YOU / FOLLOW-UP EMAIL GENERATOR ---
    st.subheader("✉️ Automated Customer Retention & Thank You Email Generator")
    st.markdown("*Generate customized corporate email communication based directly on the final sentiment analysis metrics.*")
    
    with st.container():
        st.markdown(f"**Generated Template Strategy:** `{email_tone}`")
        
        # User input fields to populate email parameters dynamically
        email_col1, email_col2 = st.columns(2)
        with email_col1:
            customer_name = st.text_input("Customer Name Placeholder:", value="Valued Partner")
        with email_col2:
            sign_off_name = st.text_input("Corporate Sign-off Name/Title:", value="Customer Operations Experience Team")
            
        customized_email = f"Subject: Regarding your recent feedback on our {product_category} experience\n\nDear {customer_name},\n\n{email_body}\n\nWarm regards,\n\n{sign_off_name}"
        
        st.text_area("Copy/Paste Ready Email Content:", value=customized_email, height=260)
        st.info("💡 **How to utilize this resource:** This communication copy automatically shifts its tone and action map to match your computed CSAT ranges, allowing automated customer outreach at scale.")

else:
    st.info("👋 Welcome! Please choose a product category and paste reviews or upload a data file in the sidebar to run the application analytics engine.")