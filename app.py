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

# 2. FIXED STABLE UI HIGH-CONTRAST CSS CUSTOM OVERRIDES
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
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
        color: #1E293B !important;
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
    
    .action-card p, .improvement-card p, .red-flag-card p,
    .action-card span, .improvement-card span, .red-flag-card span {
        color: #1E293B !important;
    }
    
    .action-card strong, .improvement-card strong, .red-flag-card strong {
        color: #0F172A !important;
        font-weight: 700;
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
    
    found_any = False
    for aspect, keywords in rules.items():
        if any(w in text_lower for w in keywords):
            sentiment = analyze_review_sentiment(text)
            aspects[aspect] = sentiment
            found_any = True
            
    if not found_any:
        aspects['General'] = analyze_review_sentiment(text)
        
    return aspects

def extract_ngrams(texts, n=2):
    words_list = []
    stop_words = {'the', 'a', 'and', 'is', 'i', 'to', 'this', 'it', 'of', 'for', 'in', 'with', 'was', 'but', 'on', 'that', 'my', 'you', 'have', 'with', 'as', 'at', 'be'}
    
    for text in texts:
        if not isinstance(text, str):
            continue
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in clean_text.split() if w not in stop_words and len(w) > 2]
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                words_list.append(ngram)
            
    return Counter(words_list).most_common(10)

# 4. VOLATILE MEMORY INITIALIZATION
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

# Defensive safety guard against completely empty structural lookups
if st.session_state['reviews_df'].empty:
    st.session_state['reviews_df'] = pd.DataFrame(columns=['Product_Name', 'Review_Text'])

df_all = st.session_state['reviews_df'].copy()
df_all['Sentiment'] = df_all['Review_Text'].apply(analyze_review_sentiment)
df_all['Emotion'] = df_all.apply(lambda row: detect_emotion(row['Review_Text'], row['Sentiment']), axis=1)

# 5. SIDEBAR DEFINITION WITH CRASH PATCHES
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 10px;'><img src='https://cdn-icons-png.flaticon.com/512/1486/1486433.png' width='65'></div>", unsafe_allow_html=True)
st.sidebar.title("Intel Engine Filters")

st.sidebar.subheader("🔍 Product Target Selection")
unique_products = sorted(list(df_all['Product_Name'].dropna().unique())) if not df_all.empty else []

if unique_products:
    selected_product = st.sidebar.selectbox("Choose Product to Review Analytics:", unique_products)
else:
    selected_product = "No Data Available"
    st.sidebar.info("Please use 'Data Ingestion' tab to load items.")

# Isolate data structures down strictly to chosen entity scope
df_processed = df_all[df_all['Product_Name'] == selected_product].copy() if unique_products else pd.DataFrame(columns=['Product_Name','Review_Text','Sentiment','Emotion'])

aspect_rows = []
if not df_processed.empty:
    for idx, row in df_processed.iterrows():
        aspect_map = assign_aspects_and_scores(row['Review_Text'])
        for aspect, asp_sent in aspect_map.items():
            aspect_rows.append({
                'Product_Name': row['Product_Name'],
                'Review_Text': row['Review_Text'],
                'Sentiment': row['Sentiment'],
                'Emotion': row['Emotion'],
                'Aspect': aspect,
                'Aspect_Sentiment': asp_sent
            })
df_aspects = pd.DataFrame(aspect_rows) if aspect_rows else pd.DataFrame(columns=['Product_Name','Review_Text','Sentiment','Emotion','Aspect','Aspect_Sentiment'])

app_mode = st.sidebar.radio("Navigate Workspaces:", ["📊 Core Dashboard", "🔍 Deep-Dive Diagnostics", "🔮 Executive Verdict & Action Engine", "📥 Data Ingestion"])

# Sidebar KPI Block Protection Logic
st.sidebar.markdown("<hr style='margin: 15px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
st.sidebar.subheader("🎯 Active Product Status")
total_processed_count = len(df_processed)
if total_processed_count > 0:
    pos_c = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    live_csat = round((pos_c / total_processed_count) * 100)
    if live_csat >= 75:
        st.sidebar.success(f"Market Leader ({live_csat}% CSAT)")
    elif live_csat >= 45:
        st.sidebar.warning(f"Volatile Contester ({live_csat}% CSAT)")
    else:
        st.sidebar.error(f"Critical Pivot Flagged ({live_csat}% CSAT)")
else:
    st.sidebar.info("0 Connected Records Detected")
    live_csat = 0

st.sidebar.subheader("⚙️ Diagnostics Modifiers")
min_freq_filter = st.sidebar.slider("Min Phrase Occurrence Cutoff", min_value=1, max_value=5, value=1)

# Application Main Headers
st.markdown('<div class="main-header">Customer Feedback Intel Engine Pro</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Displaying verified operational data pipelines for target: <strong>{selected_product}</strong></div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB MODULE 1: CORE DASHBOARD
# ----------------------------------------------------
if app_mode == "📊 Core Dashboard":
    st.header(f"📊 Dashboard Analytics: {selected_product}")
    
    total_reviews = len(df_processed)
    pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive']) if total_reviews > 0 else 0
    neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative']) if total_reviews > 0 else 0
    csat_score = round((pos_count / total_reviews) * 100) if total_reviews > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_reviews}</div><div class="metric-label">Target Reviews</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #10B981;">{pos_count}</div><div class="metric-label">Positive Volume</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #EF4444;">{neg_count}</div><div class="metric-label">Negative Volume</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #3B82F6;">{csat_score}%</div><div class="metric-label">Isolated CSAT Score</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if total_reviews > 0:
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Sentiment Distribution Metric")
            sent_counts = df_processed['Sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(sent_counts, values='Count', names='Sentiment', 
                             color='Sentiment',
                             color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'},
                             hole=0.45, template="plotly_white")
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with g2:
            st.subheader("Linguistic Emotion Spectrum Profile")
            emotion_counts = df_processed['Emotion'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            fig_bar = px.bar(emotion_counts, x='Count', y='Emotion', orientation='h',
                             color='Emotion', template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Safe)
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No active records to graph. Switch workspaces to 'Data Ingestion' to map strings.")

# ----------------------------------------------------
# TAB MODULE 2: DEEP DIAGNOSTICS
# ----------------------------------------------------
elif app_mode == "🔍 Deep-Dive Diagnostics":
    st.header("🔍 Feature Friction Breakdown & Pattern Discovery")
    
    if len(df_processed) > 0:
        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Aspect-Based Sentiment Classifications (ABSA)")
            if not df_aspects.empty:
                absa_grouped = df_aspects.groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count')
                fig_absa = px.bar(absa_grouped, x='Aspect', y='Count', color='Aspect_Sentiment',
                                  barmode='stack', template="plotly_white",
                                  color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'})
                st.plotly_chart(fig_absa, use_container_width=True)
            else:
                st.info("No aspect trends detected.")
            
        with d2:
            st.subheader("Contextual Common Phrase N-Grams")
            target_sent = st.selectbox("Filter phrase tracking streams by:", ["Negative", "Positive", "All"])
            text_pool = df_processed['Review_Text'].tolist() if target_sent == "All" else df_processed[df_processed['Sentiment'] == target_sent]['Review_Text'].tolist()
            ngrams_found = extract_ngrams(text_pool, n=2)
            
            filtered_ngrams = [ng for ng in ngrams_found if ng[1] >= min_freq_filter]
            
            if filtered_ngrams:
                ng_df = pd.DataFrame(filtered_ngrams, columns=['Phrase N-Gram', 'Density Count'])
                fig_ng = px.bar(ng_df, x='Density Count', y='Phrase N-Gram', orientation='h',
                                color_discrete_sequence=['#2563EB'], template="plotly_white")
                st.plotly_chart(fig_ng, use_container_width=True)
            else:
                st.info(f"No phrases match the frequency criteria (>= {min_freq_filter}).")

        st.subheader("Query Engine Entry Search Matrix")
        search_query = st.text_input("Filter localized metrics database using text tokens...")
        df_search = df_processed.copy()
        if search_query:
            df_search = df_search[df_search['Review_Text'].str.contains(search_query, case=False)]
        st.dataframe(df_search[['Review_Text', 'Sentiment', 'Emotion']], use_container_width=True)
    else:
        st.info("No query logs compiled for this product segment yet.")

# ----------------------------------------------------
# TAB MODULE 3: ROADMAP ENGINE
# ----------------------------------------------------
elif app_mode == "🔮 Executive Verdict & Action Engine":
    st.header("🔮 AI Business Projections & Strategic Roadmap")
    
    total_reviews = len(df_processed)
    if total_reviews > 0:
        pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
        neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
        csat = (pos_count / total_reviews) * 100
        
        st.subheader("📋 Product Portfolio Health Verdict")
        if csat >= 75:
            verdict_status = "🟢 STRONGLY APPROVED (MARKET OUTPERFORMER)"
            verdict_color = "#D1FAE5"
            verdict_text = f"The portfolio metrics showcase exceptional system standing with an enterprise satisfaction ratio of {csat:.1f}%."
        elif csat >= 45:
            verdict_status = "🟡 CAUTION REQUIRED: CONDITIONALLY APPROVED"
            verdict_color = "#FEF3C7"
            verdict_text = f"The product pipeline displays volatility, holding a conditional satisfaction index of {csat:.1f}%."
        else:
            verdict_status = "🔴 HIGH OPERATIONAL THREAT: CRITICAL INTERVENTION LEVEL"
            verdict_color = "#FEE2E2"
            verdict_text = f"The review tracking framework registers intense customer churn ({csat:.1f}% Aggregate CSAT)."
            
        # map verdict emoji to border color using string-keyed dict (emoji cannot be used as identifier names)
        border_colors = {'🟢': '#10B981', '🟡': '#F59E0B', '🔴': '#EF4444'}
        border_color = border_colors.get(verdict_status[0], '#10B981')
        st.markdown(f"""
        <div class="verdict-box" style="background-color: {verdict_color}; border-left: 6px solid {border_color}; color: #1E293B !important;">
            <h4 style="margin: 0 0 12px 0; color: #0F172A !important; font-weight: 700; font-size: 19px;">{verdict_status}</h4>
            <p style="margin: 0; color: #1E293B !important; font-size: 15px; line-height: 1.6; font-weight: 500;">{verdict_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        neg_aspects = df_aspects[df_aspects['Aspect_Sentiment'] == 'Negative']['Aspect'].tolist() if not df_aspects.empty else []
        
        report_text = f"BOARD RE-ENGINEERING DIRECTIVE FOR {selected_product.upper()}\nCSAT Score: {csat:.2f}%\nVerdict: {verdict_status}\n"
        st.download_button(
            label="📥 Download Executive Verdict Report (.TXT)",
            data=report_text,
            file_name=f"Report_{selected_product.replace(' ', '_')}.txt",
            mime="text/plain"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.subheader("💡 Suggested Changes & Product Improvements")
            if 'Product Quality' in neg_aspects:
                st.markdown('<div class="improvement-card"><strong>🔧 Physical R&D Material Stress Testing:</strong> Hard engineering flaws logged. Re-audit manufacturing tolerances.</div>', unsafe_allow_html=True)
            if 'Pricing & Value' in neg_aspects:
                st.markdown('<div class="improvement-card"><strong>🏷️ Value Strategy Realignment:</strong> Value markers trailing competitors. Review promotional tier configurations.</div>', unsafe_allow_html=True)
            if not neg_aspects:
                st.markdown('<div class="improvement-card" style="border-left-color: #10B981; background-color: #D1FAE5 !important;"><strong>✅ Performance Metrics Stable:</strong> System operating normally.</div>', unsafe_allow_html=True)

        with v_col2:
            st.subheader("🚨 Urgent CRM Customer Success Escalations")
            urgent_df = df_processed[(df_processed['Sentiment'] == 'Negative') & (df_processed['Emotion'].isin(['Anger', 'Frustration']))]
            if not urgent_df.empty:
                for idx, row in urgent_df.head(2).iterrows():
                    st.markdown(f'<div class="red-flag-card"><strong>Priority Audit Loop Required:</strong> "{row["Review_Text"]}"</div>', unsafe_allow_html=True)
            else:
                st.info("No volatile threat escalation patterns matched.")
                
        # Smart Auto-Responder Form
        st.markdown("<hr style='margin: 30px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        st.subheader("🤖 Smart CRM Automated Support Response Drafting")
        review_options = df_processed['Review_Text'].tolist()
        if review_options:
            selected_text = st.selectbox("Select target historical entry to compile response template for:", review_options)
            target_row = df_processed[df_processed['Review_Text'] == selected_text].iloc[0]
            if st.button("Compile Contextual Support Blueprint"):
                st.text_area("Blueprint Response:", value=f"Regarding evaluation for {selected_product}: '{selected_text}'", height=100)
    else:
        st.info("No system logging data present to compute projections.")

# ----------------------------------------------------
# TAB MODULE 4: DATA INGESTION
# ----------------------------------------------------
elif app_mode == "📥 Data Ingestion":
    st.header("📥 Ingestion Data Pipeline")
    
    st.subheader("🏷️ Set Product Context for Ingestion Target")
    ingest_target_product = st.text_input("Enter product name to assign incoming data to:", value=selected_product if selected_product != "No Data Available" else "AlphaPhone X")
    
    tab1, tab2 = st.tabs(["📄 Document Sheet Load (CSV/Excel)", "✍️ Live Real-Time Manual Entry"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload review dataset sheet", type=["csv", "xlsx"])
        text_column = st.text_input("Specify Target Review Text Header String", value="Review_Text")
        
        if uploaded_file is not None and ingest_target_product.strip():
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if text_column in df.columns:
                    new_entries = pd.DataFrame({
                        'Product_Name': ingest_target_product.strip(),
                        'Review_Text': df[text_column].dropna().astype(str)
                    })
                    st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_entries], ignore_index=True)
                    st.success(f"Successfully integrated {len(new_entries)} feedback lines!")
                    st.rerun()
                else:
                    st.error(f"Target Header string '{text_column}' missing inside columns.")
            except Exception as e:
                st.error(f"Parsing execution anomaly caught: {e}")
                
    with tab2:
        manual_review = st.text_area("Paste text reviews (Break lines to form unique records)")
        if st.button("Commit Logs to Master Repository Pool"):
            if manual_review.strip() and ingest_target_product.strip():
                entries = [r.strip() for r in manual_review.split('\n') if r.strip()]
                new_df = pd.DataFrame({
                    'Product_Name': ingest_target_product.strip(),
                    'Review_Text': entries
                })
                st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_df], ignore_index=True)
                st.success(f"Safely added {len(entries)} text segments.")
                st.rerun()

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Global System Multi-Product Repository Pool Check")
    st.dataframe(st.session_state['reviews_df'], use_container_width=True)

# 6. ARCHIVE EXPORTER CSV FOR ISOLATED PRODUCT DATA
if total_processed_count > 0:
    st.sidebar.markdown("<hr style='margin: 20px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
    st.sidebar.subheader("📥 Data Export Matrix")
    csv_data = df_processed.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label=f"Export {selected_product} CSV",
        data=csv_data,
        file_name=f"{selected_product.replace(' ', '_')}_export.csv",
        mime="text/csv"
    )