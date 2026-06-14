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

# 2. FIXED THEME-AGNOSTIC HIGH-CONTRAST CSS CUSTOM OVERRIDES
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .stMarkdown p {
        font-family: 'Inter', sans-serif;
    }
    
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
    
    .verdict-box {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }
    
    .improvement-card {
        background-color: #FEF3C7 !important;
        border-left: 6px solid #D97706 !important;
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 4px 12px 12px 4px;
        color: #1E293B !important;
    }
    
    .red-flag-card {
        background-color: #FEE2E2 !important;
        border-left: 6px solid #DC2626 !important;
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 4px 12px 12px 4px;
        color: #1E293B !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. ROBUST EXCEPTION-SAFE NLP ENGINE
def analyze_review_sentiment(text):
    if not isinstance(text, str):
        return 'Neutral'
    text_lower = text.lower()
    pos_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'satisfied', 'awesome', 'fast', 'smooth', 'impressed', 'wonderful', 'efficient', 'reliable']
    neg_words = ['bad', 'worst', 'terrible', 'horrible', 'hate', 'broken', 'disappointed', 'slow', 'expensive', 'waste', 'useless', 'fail', 'defect', 'poor', 'annoying', 'damaged']
    
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)
    
    if pos_count > neg_count:
        return 'Positive'
    elif neg_count > pos_count:
        return 'Negative'
    else:
        return 'Neutral'

def detect_emotion(text, sentiment):
    if not isinstance(text, str):
        return 'Neutral / Indifferent'
    text_lower = text.lower()
    if sentiment == 'Negative':
        if any(w in text_lower for w in ['broken', 'defect', 'fail', 'waste', 'poor', 'damaged']): return 'Frustration'
        if any(w in text_lower for w in ['slow', 'delay', 'wait', 'annoying']): return 'Impatience'
        if any(w in text_lower for w in ['terrible', 'worst', 'hate']): return 'Anger'
        return 'Disappointment'
    elif sentiment == 'Positive':
        if any(w in text_lower for w in ['amazing', 'excellent', 'perfect', 'best', 'wonderful', 'awesome']): return 'Delight'
        return 'Satisfaction'
    return 'Neutral / Indifferent'

def assign_aspects_and_scores(text):
    if not isinstance(text, str):
        return {'General': 'Neutral'}
    text_lower = text.lower()
    aspects = {}
    rules = {
        'Product Quality': ['quality', 'broken', 'screen', 'battery', 'build', 'material', 'durable', 'defect', 'hardware', 'performs', 'damaged'],
        'Customer Service': ['service', 'support', 'agent', 'help', 'representative', 'call', 'chat', 'response', 'team'],
        'Shipping & Delivery': ['shipping', 'delivery', 'arrived', 'package', 'fast', 'slow', 'late', 'shipment'],
        'Pricing & Value': ['price', 'expensive', 'cost', 'worth', 'cheap', 'value', 'waste', 'money', 'budget']
    }
    for aspect, keywords in rules.items():
        if any(w in text_lower for w in keywords):
            aspects[aspect] = analyze_review_sentiment(text)
    if not aspects:
        aspects['General'] = analyze_review_sentiment(text)
    return aspects

def extract_ngrams(texts, n=2):
    words_list = []
    stop_words = {'the', 'a', 'and', 'is', 'i', 'to', 'this', 'it', 'of', 'for', 'in', 'with', 'was', 'but', 'on', 'that', 'my', 'you', 'have', 'as', 'at', 'be'}
    for text in texts:
        if not isinstance(text, str): continue
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in clean_text.split() if w not in stop_words and len(w) > 2]
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                words_list.append(" ".join(words[i:i+n]))
    return Counter(words_list).most_common(10)

# 4. MEMORY LAYER STABILIZATION
if 'reviews_df' not in st.session_state:
    st.session_state['reviews_df'] = pd.DataFrame({
        'Product_Name': ["AlphaPhone X", "AlphaPhone X", "BetaBuds Pro", "BetaBuds Pro", "AlphaPhone X", "Quantum Watch"],
        'Review_Text': [
            "The battery life is amazing and the product quality is top notch. Love it!",
            "Customer service agent was extremely rude and unhelpful.",
            "Delivery was incredibly slow, package arrived damaged.",
            "Great value for money, highly recommend this budget option.",
            "The screen broke within two days of casual use. Waste of money.",
            "It performs decently but the pricing is just too expensive."
        ]
    })

df_all = st.session_state['reviews_df'].copy()
df_all['Sentiment'] = df_all['Review_Text'].apply(analyze_review_sentiment)
df_all['Emotion'] = df_all.apply(lambda row: detect_emotion(row['Review_Text'], row['Sentiment']), axis=1)

# 5. SIDEBAR CONTROLS
st.sidebar.title("Intel Engine Filters")
unique_products = sorted(list(df_all['Product_Name'].dropna().unique())) if not df_all.empty else []

if unique_products:
    selected_product = st.sidebar.selectbox("🎯 Target Product Lookup:", unique_products)
else:
    selected_product = "No Products Found"
    st.sidebar.info("Go to Ingestion workspace to load a product.")

df_processed = df_all[df_all['Product_Name'] == selected_product].copy() if unique_products else pd.DataFrame(columns=['Product_Name','Review_Text','Sentiment','Emotion'])

# Aspect Mapping
aspect_rows = []
for idx, row in df_processed.iterrows():
    for asp, asp_sent in assign_aspects_and_scores(row['Review_Text']).items():
        aspect_rows.append({'Aspect': asp, 'Aspect_Sentiment': asp_sent})
df_aspects = pd.DataFrame(aspect_rows) if aspect_rows else pd.DataFrame(columns=['Aspect','Aspect_Sentiment'])

app_mode = st.sidebar.radio("Navigate Workspaces:", ["📊 Core Dashboard", "🔍 Deep-Dive Diagnostics", "🔮 Executive Verdict", "📥 Data Ingestion"])

# Live Status Indicator
total_reviews = len(df_processed)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("Live Status Metric")
if total_reviews > 0:
    pos_c = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    csat_score = round((pos_c / total_reviews) * 100)
    st.sidebar.metric(label="Isolated CSAT Score", value=f"{csat_score}%")
else:
    st.sidebar.info("0 Records Found")
    csat_score = 0

min_freq_filter = st.sidebar.slider("Min Phrase Occurrence Filter", 1, 5, 1)

# Main Dashboard View Header
st.markdown('<div class="main-header">Customer Feedback Intel Engine Pro</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Analyzing Target Data Matrix for: <strong>{selected_product}</strong></div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 📊 CORE DASHBOARD WORKSPACE
# ----------------------------------------------------
if app_mode == "📊 Core Dashboard":
    if total_reviews > 0:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{total_reviews}</div><div class="metric-label">Total Reviews</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #10B981;">{len(df_processed[df_processed["Sentiment"]=="Positive"])}</div><div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #EF4444;">{len(df_processed[df_processed["Sentiment"]=="Negative"])}</div><div class="metric-label">Negative</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Sentiment Share")
            fig_pie = px.pie(df_processed, names='Sentiment', color='Sentiment',
                             color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'},
                             hole=0.4, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
        with g2:
            st.subheader("Emotion Spectrum Profile")
            fig_bar = px.bar(df_processed['Emotion'].value_counts().reset_index(), x='count', y='Emotion', orientation='h', template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No logs matched. Initialize data strings via the 'Data Ingestion' pipeline.")

# ----------------------------------------------------
# 🔍 DEEP-DIVE DIAGNOSTICS WORKSPACE
# ----------------------------------------------------
elif app_mode == "🔍 Deep-Dive Diagnostics":
    if total_reviews > 0:
        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Aspect-Based Sentiment Breakdown")
            if not df_aspects.empty:
                fig_absa = px.bar(df_aspects.groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count'), 
                                  x='Aspect', y='Count', color='Aspect_Sentiment', barmode='stack',
                                  color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'}, template="plotly_white")
                st.plotly_chart(fig_absa, use_container_width=True)
            else:
                st.info("No aspects located.")
        with d2:
            st.subheader("Contextual Common Phrase N-Grams")
            ngrams = extract_ngrams(df_processed['Review_Text'].tolist(), n=2)
            filtered_ngrams = [ng for ng in ngrams if ng[1] >= min_freq_filter]
            if filtered_ngrams:
                ng_df = pd.DataFrame(filtered_ngrams, columns=['Phrase N-Gram', 'Density Count'])
                fig_ng = px.bar(ng_df, x='Density Count', y='Phrase N-Gram', orientation='h', template="plotly_white")
                st.plotly_chart(fig_ng, use_container_width=True)
            else:
                st.info("No recurring phrase groupings matched selection parameters.")
        
        st.subheader("Localized Dynamic Table Query Filter")
        search = st.text_input("Type phrase keywords to parse this product's database:")
        df_search = df_processed.copy()
        if search:
            df_search = df_search[df_search['Review_Text'].str.contains(search, case=False)]
        st.dataframe(df_search[['Review_Text', 'Sentiment', 'Emotion']], use_container_width=True)
    else:
        st.info("Empty database matrix tier.")

# ----------------------------------------------------
# 🔮 EXECUTIVE VERDICT WORKSPACE
# ----------------------------------------------------
elif app_mode == "🔮 Executive Verdict":
    if total_reviews > 0:
        st.subheader("Product Performance Scorecard")
        if csat_score >= 75:
            st.markdown(f'<div class="verdict-box" style="background-color: #D1FAE5; border-left: 6px solid #10B981;"><h4>🟢 STRONGLY APPROVED ({csat_score}% CSAT)</h4><p>Value propositions verified. Scale marketing allocations.</p></div>', unsafe_allow_html=True)
        elif csat_score >= 45:
            st.markdown(f'<div class="verdict-box" style="background-color: #FEF3C7; border-left: 6px solid #F59E0B;"><h4>🟡 CONDITIONALLY APPROVED ({csat_score}% CSAT)</h4><p>Volatility noted. Resolve component or service bottlenecks before scaling.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-box" style="background-color: #FEE2E2; border-left: 6px solid #EF4444;"><h4>🔴 CRITICAL SYSTEM WARNING ({csat_score}% CSAT)</h4><p>High churn vectors logged. Audit manufacturing tolerances immediately.</p></div>', unsafe_allow_html=True)
            
        report_data = f"Executive Summary Report: {selected_product}\nCSAT Score: {csat_score}%\nTotal Records: {total_reviews}"
        st.download_button(label="📥 Download Executive Blueprint Text Report", data=report_data, file_name=f"Report_{selected_product}.txt")
    else:
        st.info("No active records available to synthesize projection roadmap cards.")

# ----------------------------------------------------
# 📥 DATA INGESTION WORKSPACE
# ----------------------------------------------------
elif app_mode == "📥 Data Ingestion":
    st.header("Data Intake Pipeline")
    target_prod = st.text_input("Enter product target label identifier:", value=selected_product if selected_product != "No Products Found" else "New Product")
    
    t1, t2 = st.tabs(["📄 Sheet Document Loader (CSV/Excel)", "✍️ Real-Time Stream Entry"])
    with t1:
        file = st.file_uploader("Upload review file", type=["csv", "xlsx"])
        col_header = st.text_input("Review Text Column Header Name Match String", value="Review_Text")
        if file and st.button("Ingest Uploaded File Matrix"):
            df_in = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            if col_header in df_in.columns:
                new_df = pd.DataFrame({'Product_Name': target_prod.strip(), 'Review_Text': df_in[col_header].dropna().astype(str)})
                st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_df], ignore_index=True)
                st.success(f"Injected {len(new_df)} lines safely!"); st.rerun()
            else:
                st.error(f"Header '{col_header}' not found.")
    with t2:
        text_feed = st.text_area("Paste user feedback strings (Separate using line breaks)")
        if st.button("Commit Raw Streams to Repository Pool"):
            if text_feed.strip():
                lines = [l.strip() for l in text_feed.split('\n') if l.strip()]
                new_df = pd.DataFrame({'Product_Name': target_prod.strip(), 'Review_Text': lines})
                st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_df], ignore_index=True)
                st.success(f"Committed {len(lines)} records successfully!"); st.rerun()