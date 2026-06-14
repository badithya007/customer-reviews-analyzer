import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from datetime import datetime

# ==========================================
# 1. APPLICATION ARCHITECTURE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Feedback Intel Engine Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force-inject strict, theme-isolated CSS to maintain high text contrast on light/dark modes
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .stMarkdown p {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.025em;
    }
    .sub-header {
        font-size: 15px;
        color: #475569 !important;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #0F172A !important;
        line-height: 1;
    }
    .metric-label {
        font-size: 12px;
        color: #64748B !important;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }
    .improvement-card {
        background-color: #FEF3C7 !important;
        border-left: 5px solid #D97706 !important;
        padding: 16px;
        margin-bottom: 14px;
        border-radius: 4px 8px 8px 4px;
        color: #1E293B !important;
    }
    .red-flag-card {
        background-color: #FEE2E2 !important;
        border-left: 5px solid #DC2626 !important;
        padding: 16px;
        margin-bottom: 14px;
        border-radius: 4px 8px 8px 4px;
        color: #1E293B !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FAILSAFE RULE-BASED NLP CORAL ENGINES
# ==========================================
def run_sentiment_inference(text):
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
    txt = text.lower()
    pos = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'satisfied', 'awesome', 'fast', 'smooth', 'impressed', 'wonderful']
    neg = ['bad', 'worst', 'terrible', 'horrible', 'hate', 'broken', 'disappointed', 'slow', 'expensive', 'waste', 'useless', 'fail', 'defect', 'poor', 'damaged']
    p_score = sum(1 for w in pos if w in txt)
    n_score = sum(1 for w in neg if w in txt)
    if p_score > n_score: return 'Positive'
    if n_score > p_score: return 'Negative'
    return 'Neutral'

def run_emotion_inference(text, sentiment):
    if not isinstance(text, str) or not text.strip():
        return 'Neutral / Indifferent'
    txt = text.lower()
    if sentiment == 'Negative':
        if any(w in txt for w in ['broken', 'defect', 'fail', 'waste', 'poor', 'damaged']): return 'Frustration'
        if any(w in txt for w in ['slow', 'delay', 'wait', 'annoying']): return 'Impatience'
        if any(w in txt for w in ['terrible', 'worst', 'hate']): return 'Anger'
        return 'Disappointment'
    if sentiment == 'Positive':
        if any(w in txt for w in ['amazing', 'excellent', 'perfect', 'best', 'wonderful', 'awesome']): return 'Delight'
        return 'Satisfaction'
    return 'Neutral / Indifferent'

def run_aspect_inference(text):
    if not isinstance(text, str) or not text.strip():
        return {'General': 'Neutral'}
    txt = text.lower()
    aspects = {}
    rules = {
        'Product Quality': ['quality', 'broken', 'screen', 'battery', 'build', 'material', 'durable', 'defect', 'hardware', 'performs', 'damaged'],
        'Customer Service': ['service', 'support', 'agent', 'help', 'representative', 'call', 'chat', 'response', 'team'],
        'Shipping & Delivery': ['shipping', 'delivery', 'arrived', 'package', 'fast', 'slow', 'late', 'shipment'],
        'Pricing & Value': ['price', 'expensive', 'cost', 'worth', 'cheap', 'value', 'waste', 'money', 'budget']
    }
    for aspect, keywords in rules.items():
        if any(w in txt for w in keywords):
            aspects[aspect] = run_sentiment_inference(text)
    if not aspects:
        aspects['General'] = run_sentiment_inference(text)
    return aspects

def run_ngram_extraction(texts, n=2):
    phrases = []
    stops = {'the', 'a', 'and', 'is', 'i', 'to', 'this', 'it', 'of', 'for', 'in', 'with', 'was', 'but', 'on', 'that', 'my', 'you', 'have', 'as', 'at', 'be'}
    for t in texts:
        if not isinstance(t, str): continue
        clean = re.sub(r'[^\w\s]', '', t.lower())
        tokens = [w for w in clean.split() if w not in stops and len(w) > 2]
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                phrases.append(" ".join(tokens[i:i+n]))
    return Counter(phrases).most_common(10)

# ==========================================
# 3. COMPREHENSIVE REPOSITORY INITIALIZATION
# ==========================================
if 'reviews_df' not in st.session_state:
    st.session_state['reviews_df'] = pd.DataFrame({
        'Product_Name': [
            "AlphaPhone X", "AlphaPhone X", "BetaBuds Pro", "BetaBuds Pro", 
            "AlphaPhone X", "Quantum Watch", "Quantum Watch", "BetaBuds Pro"
        ],
        'Review_Text': [
            "The battery life is amazing and the product quality is top notch. Love it!",
            "Customer service agent was extremely rude and unhelpful. Took 3 days to get a response.",
            "Delivery was incredibly slow, package arrived damaged. Very disappointed.",
            "Great value for money, highly recommend this to anyone looking for a budget option.",
            "The screen broke within two days of casual use. Absolute waste of money.",
            "It performs decently but the pricing is just too expensive for what it offers.",
            "Extremely fast shipping! Product matches description perfectly.",
            "The customer support resolved my issue instantly. Great team!"
        ]
    })

# Process structural data copies
df_master = st.session_state['reviews_df'].copy()
df_master['Sentiment'] = df_master['Review_Text'].apply(run_sentiment_inference)
df_master['Emotion'] = df_master.apply(lambda r: run_emotion_inference(r['Review_Text'], r['Sentiment']), axis=1)

# ==========================================
# 4. SIDEBAR FILTER LOGIC MATRIX
# ==========================================
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 15px;'><img src='https://cdn-icons-png.flaticon.com/512/1486/1486433.png' width='60'></div>", unsafe_allow_html=True)
st.sidebar.title("Intel System Navigator")

st.sidebar.subheader("🔍 Product Portfolio Lookup")
all_active_products = sorted(list(df_master['Product_Name'].dropna().unique())) if not df_master.empty else []

if all_active_products:
    target_product = st.sidebar.selectbox("Choose Product Target File:", all_active_products)
else:
    target_product = "No Data Available"
    st.sidebar.warning("Load datasets using the Ingestion tab.")

# Narrow core processing arrays down to chosen target product
df_isolated = df_master[df_master['Product_Name'] == target_product].copy() if all_active_products else pd.DataFrame(columns=['Product_Name','Review_Text','Sentiment','Emotion'])

# Unpack isolated aspect metrics
aspect_list = []
for idx, r in df_isolated.iterrows():
    for asp, asp_s in run_aspect_inference(r['Review_Text']).items():
        aspect_list.append({'Aspect': asp, 'Aspect_Sentiment': asp_s})
df_aspect_isolated = pd.DataFrame(aspect_list) if aspect_list else pd.DataFrame(columns=['Aspect','Aspect_Sentiment'])

app_mode = st.sidebar.radio("Active Workspace:", ["📊 Analytics Panel", "🔍 Deep Diagnostics", "🔮 Executive Verdict", "📥 Data Ingestion Panel"])

# Status Dashboard Metric Calculation
st.sidebar.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
st.sidebar.subheader("📈 Operational Standing")
total_rows = len(df_isolated)
if total_rows > 0:
    positive_count = len(df_isolated[df_isolated['Sentiment'] == 'Positive'])
    current_csat = round((positive_count / total_rows) * 100)
    st.sidebar.metric(label=f"CSAT Profile ({target_product})", value=f"{current_csat}%")
else:
    st.sidebar.info("0 Logs Tracked")
    current_csat = 0

min_freq_filter = st.sidebar.slider("Min Density Occurrence Cutoff", 1, 5, 1)

# App View Headings
st.markdown('<div class="main-header">Customer Feedback Intel Engine Pro</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Processing data telemetry matrix for target asset: <strong>{target_product}</strong></div>', unsafe_allow_html=True)

# ==========================================
# WORKSPACE 1: CORE GRAPHICAL ANALYTICS
# ==========================================
if app_mode == "📊 Analytics Panel":
    st.header(f"📊 Dashboard Summary Profile: {target_product}")
    
    if total_rows > 0:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="metric-value">{total_rows}</div><div class="metric-label">Evaluated Ingestions</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #10B981;">{len(df_isolated[df_isolated["Sentiment"]=="Positive"])}</div><div class="metric-label">Positive Logs</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #EF4444;">{len(df_isolated[df_isolated["Sentiment"]=="Negative"])}</div><div class="metric-label">Negative Flags</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        
        with g1:
            st.subheader("System Sentiment Mix Ratio")
            fig_pie = px.pie(df_isolated, names='Sentiment', color='Sentiment',
                             color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'},
                             hole=0.45, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with g2:
            st.subheader("Linguistic Emotion Breakdown")
            fig_bar = px.bar(df_isolated['Emotion'].value_counts().reset_index(), x='count', y='Emotion', 
                             orientation='h', template="plotly_white", color='Emotion',
                             color_discrete_sequence=px.colors.qualitative.Safe)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No system records loaded for this product tier. Jump to Data Ingestion Panel.")

# ==========================================
# WORKSPACE 2: DEEP DIAGNOSTICS & LOOKUPS
# ==========================================
elif app_mode == "🔍 Deep Diagnostics":
    st.header(f"🔍 Feature Friction Identifiers: {target_product}")
    
    if total_rows > 0:
        d1, d2 = st.columns(2)
        
        with d1:
            st.subheader("Aspect-Based Structural Classifications (ABSA)")
            if not df_aspect_isolated.empty:
                absa_grouped = df_aspect_isolated.groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count')
                fig_absa = px.bar(absa_grouped, x='Aspect', y='Count', color='Aspect_Sentiment', barmode='stack',
                                  color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'}, template="plotly_white")
                st.plotly_chart(fig_absa, use_container_width=True)
            else:
                st.info("Insufficient diagnostic keywords found.")
                
        with d2:
            st.subheader("Density Phrase Tracker (N-Grams)")
            tokens_pool = df_isolated['Review_Text'].tolist()
            extracted_ngrams = run_ngram_extraction(tokens_pool, n=2)
            filtered_ngrams = [ng for ng in extracted_ngrams if ng[1] >= min_freq_filter]
            
            if filtered_ngrams:
                ng_df = pd.DataFrame(filtered_ngrams, columns=['Phrase Tuple', 'Frequency'])
                fig_ng = px.bar(ng_df, x='Frequency', y='Phrase Tuple', orientation='h', template="plotly_white")
                st.plotly_chart(fig_ng, use_container_width=True)
            else:
                st.info(f"No double-word chains matched the target cutoff threshold (>= {min_freq_filter}).")
                
        st.subheader("Sub-String Real-Time Database Query Engine")
        term_query = st.text_input("Type characters or custom criteria to query text parameters directly:")
        df_queried = df_isolated.copy()
        if term_query:
            df_queried = df_queried[df_queried['Review_Text'].str.contains(term_query, case=False)]
        st.dataframe(df_queried[['Review_Text', 'Sentiment', 'Emotion']], use_container_width=True)
    else:
        st.info("No tracing logs mapped.")

# ==========================================
# WORKSPACE 3: EXECUTIVE VERDICT ROADMAPS
# ==========================================
elif app_mode == "🔮 Executive Verdict":
    st.header(f"🔮 Operational Roadmap & Projections: {target_product}")
    
    if total_rows > 0:
        st.subheader("📋 Board Performance Scorecard")
        if current_csat >= 75:
            st.markdown(f'<div class="verdict-box" style="background-color: #D1FAE5; border-left: 6px solid #10B981; color: #065F46;"><h4 style="margin:0 0 8px 0;font-weight:700;">🟢 ASSET OPERATIONS STABLE ({current_csat}% CSAT)</h4><p style="margin:0;">Core infrastructure claims verified. Expand user allocation and push version deployments.</p></div>', unsafe_allow_html=True)
        elif current_csat >= 45:
            st.markdown(f'<div class="verdict-box" style="background-color: #FEF3C7; border-left: 6px solid #F59E0B; color: #92400E;"><h4 style="margin:0 0 8px 0;font-weight:700;">🟡 SYSTEM CONDITION VOLATILE ({current_csat}% CSAT)</h4><p style="margin:0;">Friction loops caught inside user workflows. Deploy immediate optimization sprints before scaling footprint.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-box" style="background-color: #FEE2E2; border-left: 6px solid #EF4444; color: #991B1B;"><h4 style="margin:0 0 8px 0;font-weight:700;">🔴 OPERATIONAL EMBARGO LEVEL ({current_csat}% CSAT)</h4><p style="margin:0;">High structural degradation. Halt deployment pipelines and initiate technical-debt architecture reviews immediately.</p></div>', unsafe_allow_html=True)
            
        neg_aspect_tags = df_aspect_isolated[df_aspect_isolated['Aspect_Sentiment'] == 'Negative']['Aspect'].tolist() if not df_aspect_isolated.empty else []
        
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            st.subheader("💡 Suggested Engineering Fixes")
            if 'Product Quality' in neg_aspect_tags:
                st.markdown('<div class="improvement-card"><strong>🔧 Physical R&D Tolerances:</strong> Engineering fault indexes flagged. Audit stress testing procedures.</div>', unsafe_allow_html=True)
            if 'Pricing & Value' in neg_aspect_tags:
                st.markdown('<div class="improvement-card"><strong>🏷️ Value Realignment:</strong> Pricing friction registered. Evaluate elastic multi-tier cost offerings.</div>', unsafe_allow_html=True)
            if not neg_aspect_tags:
                st.markdown('<div class="improvement-card" style="background-color:#E0F2FE; border-left-color:#0EA5E9;"><strong>✅ Systems Optimal:</strong> Target parameters running within expected parameters.</div>', unsafe_allow_html=True)
                
        with o_col2:
            st.subheader("🚨 Priority Account Manager Alerts")
            df_risk = df_isolated[(df_isolated['Sentiment'] == 'Negative') & (df_isolated['Emotion'].isin(['Anger', 'Frustration']))]
            if not df_risk.empty():
                for idx, row in df_risk.head(2).iterrows():
                    st.markdown(f'<div class="red-flag-card"><strong>Critical Ticket:</strong> "{row["Review_Text"]}"</div>', unsafe_allow_html=True)
            else:
                st.info("No customer accounts logged system risk anomalies.")
                
        # Structural Corporate Output Document Download
        executive_summary = f"ENGINEERING ANALYTICS SUMMARY MATRIX: {target_product.upper()}\nDate Run: {datetime.now().strftime('%Y-%m-%d')}\nCSAT Standing: {current_csat}%\nTotal Feed Volumes: {total_rows}"
        st.download_button(label="📥 Download Production Strategic Report", data=executive_summary, file_name=f"Executive_Matrix_{target_product}.txt")
    else:
        st.info("No calculations exist to compute report directives.")

# ==========================================
# WORKSPACE 4: DATA INGESTION MATRIX PIPELINE
# ==========================================
elif app_mode == "📥 Data Ingestion Panel":
    st.header("📥 Ingestion Intake Control Grid")
    
    context_product_label = st.text_input("Target allocation product label context string:", value=target_product if target_product != "No Data Available" else "AlphaPhone X")
    
    panel1, panel2 = st.tabs(["📄 Sheet Bulk File Upload (CSV/Excel)", "✍️ Real-Time Terminal Manual Injection"])
    
    with panel1:
        doc_file = st.file_uploader("Drop target analysis sheet", type=["csv", "xlsx"])
        text_column_header = st.text_input("Target String Parameter Column Header Name Matching Token", value="Review_Text")
        
        if doc_file and context_product_label.strip() and st.button("Process & Integrate File Fields"):
            try:
                df_loaded = pd.read_csv(doc_file) if doc_file.name.endswith('.csv') else pd.read_excel(doc_file)
                if text_column_header in df_loaded.columns:
                    new_append_rows = pd.DataFrame({
                        'Product_Name': context_product_label.strip(),
                        'Review_Text': df_loaded[text_column_header].dropna().astype(str)
                    })
                    st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_append_rows], ignore_index=True)
                    st.success(f"Successfully appended {len(new_append_rows)} items to repository registry!")
                    st.rerun()
                else:
                    st.error(f"Column key string '{text_column_header}' missing inside file layout structure.")
            except Exception as error:
                st.error(f"Failsafe parser execution handler error: {error}")
                
    with panel2:
        terminal_raw_input = st.text_area("Paste unstructured text content line-by-line (Line breaks configure unique assets)")
        if st.button("Inject Terminal Strings to Repository"):
            if terminal_raw_input.strip() and context_product_label.strip():
                clean_lines = [line.strip() for line in terminal_raw_input.split('\n') if line.strip()]
                new_manual_rows = pd.DataFrame({
                    'Product_Name': context_product_label.strip(),
                    'Review_Text': clean_lines
                })
                st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_manual_rows], ignore_index=True)
                st.success(f"Committed {len(clean_lines)} rows safely into active storage context.")
                st.rerun()

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Active System Multi-Product Repository Pool Checklist")
    st.dataframe(st.session_state['reviews_df'], use_container_width=True)

# ==========================================
# 5. HARDENED EXPORT MODULE DATA ENGINE
# ==========================================
if total_rows > 0:
    st.sidebar.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
    st.sidebar.subheader("📥 Data Export Engine")
    csv_bytes = df_isolated.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label=f"Export {target_product} Matrix (.CSV)",
        data=csv_bytes,
        file_name=f"Intel_Export_{target_product.replace(' ', '_')}.csv",
        mime="text/csv"
    )