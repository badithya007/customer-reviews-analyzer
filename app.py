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

# 2. THEME-AGNOSTIC HIGH-CONTRAST CSS CUSTOM OVERRIDES (LIGHT & DARK MODE STABILIZED)
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

# 3. RULE-BASED VECTOR NLP CORES
def analyze_review_sentiment(text):
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
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in clean_text.split() if w not in stop_words and len(w) > 2]
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            words_list.append(ngram)
            
    return Counter(words_list).most_common(10)

# 4. INITIALIZE MULTI-PRODUCT SYSTEM DATABASE
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

# Compute global schemas
df_all = st.session_state['reviews_df'].copy()
df_all['Sentiment'] = df_all['Review_Text'].apply(analyze_review_sentiment)
df_all['Emotion'] = df_all.apply(lambda row: detect_emotion(row['Review_Text'], row['Sentiment']), axis=1)

# 5. SIDEBAR TARGET PRODUCT SELECTOR MATRIX
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 10px;'><img src='https://cdn-icons-png.flaticon.com/512/1486/1486433.png' width='65'></div>", unsafe_allow_html=True)
st.sidebar.title("Intel Engine Filters")

# Product Query Lookup Interface
st.sidebar.subheader("🔍 Product Target Selection")
unique_products = sorted(list(df_all['Product_Name'].unique()))
selected_product = st.sidebar.selectbox("Choose Product to Review Analytics:", unique_products)

# Filter global database specifically down to target product choice
df_processed = df_all[df_all['Product_Name'] == selected_product].copy()

# Dynamic Aspect Evaluation specifically isolated down to target chosen product
aspect_rows = []
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

# Workspace Navigation Tabs
app_mode = st.sidebar.radio("Navigate Workspaces:", ["📊 Core Dashboard", "🔍 Deep-Dive Diagnostics", "🔮 Executive Verdict & Action Engine", "📥 Data Ingestion"])

# Real-Time Status Metric computation for isolated product matrix
st.sidebar.markdown("<hr style='margin: 15px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
st.sidebar.subheader("🎯 Active Product Status")
pos_c = len(df_processed[df_processed['Sentiment'] == 'Positive'])
t_c = len(df_processed)
live_csat = round((pos_c / t_c) * 100) if t_c > 0 else 0

if live_csat >= 75:
    st.sidebar.success(f"Market Leader ({live_csat}% CSAT)")
elif live_csat >= 45:
    st.sidebar.warning(f"Volatile Contester ({live_csat}% CSAT)")
else:
    st.sidebar.error(f"Critical Pivot Flagged ({live_csat}% CSAT)")

st.sidebar.subheader("⚙️ Diagnostics Modifiers")
min_freq_filter = st.sidebar.slider("Min Phrase Occurrence Cutoff", min_value=1, max_value=5, value=1)

# Application Typography Headers
st.markdown('<div class="main-header">Customer Feedback Intel Engine Pro</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Displaying verified operational data pipelines for target: <strong>{selected_product}</strong></div>', unsafe_allow_html=True)

# ----------------------------------------------------
# WORKSPACE MODULE 1: GRAPHICAL EXECUTIVE DASHBOARD
# ----------------------------------------------------
if app_mode == "📊 Core Dashboard":
    st.header(f"📊 Dashboard Analytics: {selected_product}")
    
    total_reviews = len(df_processed)
    pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
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
                             hole=0.45)
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with g2:
            st.subheader("Linguistic Emotion Spectrum Profile")
            emotion_counts = df_processed['Emotion'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            fig_bar = px.bar(emotion_counts, x='Count', y='Emotion', orientation='h',
                             color='Emotion',
                             color_discrete_sequence=px.colors.qualitative.Safe)
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No logs compiled for this product tier yet. Populate records via the Ingestion workspace module.")

# ----------------------------------------------------
# WORKSPACE MODULE 2: DEEP-DIVE DIAGNOSTICS MATRIX
# ----------------------------------------------------
elif app_mode == "🔍 Deep-Dive Diagnostics":
    st.header(f"🔍 Feature Friction Breakdown & Patterns for {selected_product}")
    
    if len(df_processed) > 0:
        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Aspect-Based Sentiment Classifications (ABSA)")
            if not df_aspects.empty:
                absa_grouped = df_aspects.groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count')
                fig_absa = px.bar(absa_grouped, x='Aspect', y='Count', color='Aspect_Sentiment',
                                  barmode='stack',
                                  color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'})
                st.plotly_chart(fig_absa, use_container_width=True)
            else:
                st.info("Insufficient keyword tokens located to map individual system metrics.")
            
        with d2:
            st.subheader("Contextual Common Phrase N-Grams")
            target_sent = st.selectbox("Filter phrase tracking streams by:", ["Negative", "Positive", "All"])
            text_pool = df_processed['Review_Text'].tolist() if target_sent == "All" else df_processed[df_processed['Sentiment'] == target_sent]['Review_Text'].tolist()
            ngrams_found = extract_ngrams(text_pool, n=2)
            
            filtered_ngrams = [ng for ng in ngrams_found if ng[1] >= min_freq_filter]
            
            if filtered_ngrams:
                ng_df = pd.DataFrame(filtered_ngrams, columns=['Phrase N-Gram', 'Density Count'])
                fig_ng = px.bar(ng_df, x='Density Count', y='Phrase N-Gram', orientation='h',
                                color_discrete_sequence=['#2563EB'])
                st.plotly_chart(fig_ng, use_container_width=True)
            else:
                st.info(f"No repeating phrases found crossing the minimum occurrence cutoff of {min_freq_filter}.")

        st.subheader("Query Engine Search Filter")
        search_query = st.text_input("Filter this product's database using custom terms...")
        df_search = df_processed.copy()
        if search_query:
            df_search = df_search[df_search['Review_Text'].str.contains(search_query, case=False)]
        st.dataframe(df_search[['Review_Text', 'Sentiment', 'Emotion']], use_container_width=True)
    else:
        st.info("No evaluation data exists to map granular diagnostic metrics.")

# ----------------------------------------------------
# WORKSPACE MODULE 3: EXECUTIVE ACTION & REPORT GENERATOR ENGINE
# ----------------------------------------------------
elif app_mode == "🔮 Executive Verdict & Action Engine":
    st.header(f"🔮 Projections & Roadmap: {selected_product}")
    
    total_reviews = len(df_processed)
    if total_reviews > 0:
        pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
        neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
        csat = (pos_count / total_reviews) * 100
        
        # PRODUCT HEALTH FINAL VERDICT BOX
        st.subheader("📋 Product Portfolio Health Verdict")
        if csat >= 75:
            verdict_status = "🟢 STRONGLY APPROVED (MARKET OUTPERFORMER)"
            verdict_color = "#D1FAE5"
            verdict_text = f"The portfolio metrics showcase exceptional system standing with an enterprise satisfaction ratio of {csat:.1f}%. Consumer patterns validate all fundamental value propositions. Operational Directive: Accelerate volume scaling pipelines, widen target marketing allocations, and boost product availability indexes."
        elif csat >= 45:
            verdict_status = "🟡 CAUTION REQUIRED: CONDITIONALLY APPROVED (MARKET CONTESTER)"
            verdict_color = "#FEF3C7"
            verdict_text = f"The product pipeline displays volatility, holding a conditional satisfaction index of {csat:.1f}%. Strong structural validation is offset by friction in shipping lines, customer service bottlenecks, or minor component tolerances. Operational Directive: Pause expansion; assign immediate design sprints to resolve current friction vectors before scaling."
        else:
            verdict_status = "🔴 HIGH OPERATIONAL THREAT: CRITICAL INTERVENTION LEVEL"
            verdict_color = "#FEE2E2"
            verdict_text = f"The review tracking framework registers intense customer churn alongside severe operational complaints ({csat:.1f}% Aggregate CSAT). Structural failures indicate deep technical debt or core user frustration metrics. Operational Directive: Halt outbound pipelines, deploy comprehensive structural auditing frameworks, and rebuild target design roadmaps immediately."
            
        st.markdown(f"""
        <div class="verdict-box" style="background-color: {verdict_color}; border-left: 6px solid {dict(🟢='#10B981', 🟡='#F59E0B', 🔴='#EF4444')[verdict_status[0]]}; color: #1E293B !important;">
            <h4 style="margin: 0 0 12px 0; color: #0F172A !important; font-weight: 700; font-size: 19px;">{verdict_status}</h4>
            <p style="margin: 0; color: #1E293B !important; font-size: 15px; line-height: 1.6; font-weight: 500;">{verdict_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # COMPILING FINAL REPORT DOWNLOAD WITH SPECIFIC PRODUCT SCOPE
        neg_aspects = df_aspects[df_aspects['Aspect_Sentiment'] == 'Negative']['Aspect'].tolist() if not df_aspects.empty else []
        
        report_text = f"""==================================================
BOARD EXECUTIVE FEEDBACK REPORT: {selected_product.upper()}
Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

[TARGET PRODUCT SUMMARY METRICS]
Product Lookup Identifier Target: {selected_product}
Total Volume of Customer Records Evaluated: {total_reviews}
Positive Sentiment Volume: {pos_count}
Negative Sentiment Volume: {neg_count}
Calculated Product Satisfaction (CSAT) Rating: {csat:.2f}%

[SYSTEM HEALTH FINAL VERDICT STATUS]
Portfolio Status Rating: {verdict_status}
Summary Analysis: {verdict_text}

[TARGETED ACTIONABLE R&D ROADMAP ITEMS]
"""
        if 'Product Quality' in neg_aspects:
            report_text += "- Physical R&D Material Stress Testing: Data notes structural weak points. Deploy stress test cycles.\n"
        if 'Pricing & Value' in neg_aspects:
            report_text += "- Cost Positioning Strategy: Price friction detected. Launch packaging tier optimizations.\n"
        if 'Customer Service' in neg_aspects or 'Shipping & Delivery' in neg_aspects:
            report_text += "- Operations & Logistics Overhaul: Delivery lag noted. Audit logistics partnerships.\n"
        if not neg_aspects:
            report_text += "- Normal Operational Targets Validated. Continue baseline software sprints.\n"
            
        report_text += f"\n==================================================\nEnd of Compiled Feedback Report for {selected_product}"

        st.download_button(
            label=f"📥 Download Final Verdict Report for {selected_product} (.TXT)",
            data=report_text,
            file_name=f"Executive_Verdict_{selected_product.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # COMPLIANCE IMPROVEMENT CARDS WITH HIGH FONT CONTRAST
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("💡 Suggested Improvements")
            if 'Product Quality' in neg_aspects:
                st.markdown('<div class="improvement-card"><strong>🔧 Material Stress Testing:</strong> Customer data tracks design architectural structural vulnerabilities. Trigger mechanics-lab evaluations to reinforce framework alloy metrics.</div>', unsafe_allow_html=True)
            if 'Pricing & Value' in neg_aspects:
                st.markdown('<div class="improvement-card"><strong>🏷️ Value Structural Realignment:</strong> Price parameters show friction patterns. Introduce tiered pricing strategies or packaging configurations.</div>', unsafe_allow_html=True)
            if 'Customer Service' in neg_aspects or 'Shipping & Delivery' in neg_aspects:
                st.markdown('<div class="improvement-card"><strong>⚙️ Logistics Operations Overhaul:</strong> Transaction traces catch pipeline dispatch delay trends. Audit external fulfillment partners.</div>', unsafe_allow_html=True)
            if not neg_aspects:
                st.markdown('<div class="improvement-card" style="border-left-color: #10B981; background-color: #D1FAE5 !important;"><strong>✅ Target Threshold Met:</strong> Product operations exhibit low systemic friction scores. Run standard support processes.</div>', unsafe_allow_html=True)

        with v_col2:
            st.subheader("🚨 Urgent Account Manager Escalations")
            urgent_df = df_processed[(df_processed['Sentiment'] == 'Negative') & (df_processed['Emotion'].isin(['Anger', 'Frustration']))]
            if not urgent_df.empty:
                for idx, row in urgent_df.head(2).iterrows():
                    st.markdown(f'<div class="red-flag-card"><strong>High Threat Log:</strong> "{row["Review_Text"]}" <br><small style="color: #991B1B !important; font-weight: 700; display: block; margin-top: 6px;">Root Cause Emotion: {row["Emotion"]} | Threat Scope: Immediate Escalation Required</small></div>', unsafe_allow_html=True)
            else:
                st.info("No high-risk emergency escalations isolated within this product category.")

        # AUTOMATED RESPONSE ENGINE
        st.markdown("<hr style='margin: 30px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        st.subheader("🤖 Automated CRM Support Response Blueprint")
        review_options = df_processed['Review_Text'].tolist()
        if review_options:
            selected_text = st.selectbox("Select historical review entry to map response schema:", review_options)
            target_row = df_processed[df_processed['Review_Text'] == selected_text].iloc[0]
            
            if st.button("Compile Support Blueprint"):
                if target_row['Sentiment'] == 'Negative':
                    email_draft = f"Subject: Resolution Protocol: Escalated {selected_product} Experience Audit\n\nDear Customer,\n\nWe tracked your customer feedback log regarding operations with your {selected_product}: \"{selected_text}\". We sincerely apologize for any performance disruptions this introduced.\n\nOur QA engineering leadership has updated your product account file. A priority customer service manager will reach out with resolution remedies.\n\nBest regards,\nCustomer Experience Operations"
                else:
                    email_draft = f"Subject: Deep Appreciation: {selected_product} Operational Review\n\nDear Customer,\n\nThank you for documenting your wonderful review regarding our {selected_product}: \"{selected_text}\". We are thrilled to hear the platform handles your projects effectively!\n\nBest regards,\nCustomer Experience Operations"
                st.text_area("Generated CRM Corporate Template Output:", value=email_draft, height=180)
    else:
        st.info("No calculation matrices exist for this product.")

# ----------------------------------------------------
# WORKSPACE MODULE 4: INGESTION PIPELINE
# ----------------------------------------------------
elif app_mode == "📥 Data Ingestion":
    st.header("📥 Ingestion Data Pipeline")
    
    st.subheader("🏷️ Set Product Context for Ingestion Target")
    ingest_target_product = st.text_input("Enter product name to assign incoming data to:", value=selected_product)
    
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
                    st.success(f"Successfully integrated {len(new_entries)} feedback lines under '{ingest_target_product.strip()}'!")
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
                st.success(f"Safely added {len(entries)} text segments under '{ingest_target_product.strip()}'.")
                st.rerun()

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Global System Multi-Product Repository Pool Checklist")
    st.dataframe(st.session_state['reviews_df'], use_container_width=True)

# 6. ARCHIVE EXPORTER CSV FOR ISOLATED PRODUCT DATA
st.sidebar.markdown("<hr style='margin: 20px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
st.sidebar.subheader("📥 Data Export Matrix")
csv_data = df_processed.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label=f"Export {selected_product} CSV Data",
    data=csv_data,
    file_name=f"{selected_product.replace(' ', '_')}_intel_export.csv",
    mime="text/csv"
)