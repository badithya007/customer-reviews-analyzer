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

# 4. INITIALIZE DATA POOL WORKSPACE
if 'reviews_df' not in st.session_state:
    st.session_state['reviews_df'] = pd.DataFrame({
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

# Compute metrics globally for continuous telemetry validation
df_processed = st.session_state['reviews_df'].copy()
df_processed['Sentiment'] = df_processed['Review_Text'].apply(analyze_review_sentiment)
df_processed['Emotion'] = df_processed.apply(lambda row: detect_emotion(row['Review_Text'], row['Sentiment']), axis=1)

aspect_rows = []
for idx, row in df_processed.iterrows():
    aspect_map = assign_aspects_and_scores(row['Review_Text'])
    for aspect, asp_sent in aspect_map.items():
        aspect_rows.append({
            'Review_Text': row['Review_Text'],
            'Sentiment': row['Sentiment'],
            'Emotion': row['Emotion'],
            'Aspect': aspect,
            'Aspect_Sentiment': asp_sent
        })
df_aspects = pd.DataFrame(aspect_rows)

# 5. SIDEBAR CORE NAVIGATION & STATISTICAL CONTROLS
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 20px;'><img src='https://cdn-icons-png.flaticon.com/512/1486/1486433.png' width='70'></div>", unsafe_allow_html=True)
st.sidebar.title("Feedback Intel Matrix")
app_mode = st.sidebar.radio("Navigate Workspaces:", ["📥 Data Ingestion", "📊 Core Dashboard", "🔍 Deep-Dive Diagnostics", "🔮 Executive Verdict & Action Engine"])

# Live KPI Widget in Sidebar
st.sidebar.markdown("<hr style='margin: 15px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
st.sidebar.subheader("🎯 Real-Time Status Metric")
pos_c = len(df_processed[df_processed['Sentiment'] == 'Positive'])
t_c = len(df_processed)
live_csat = round((pos_c / t_c) * 100) if t_c > 0 else 0

if live_csat >= 75:
    st.sidebar.success(f"Market Leader Standing ({live_csat}% CSAT)")
elif live_csat >= 45:
    st.sidebar.warning(f"Volatile Contester ({live_csat}% CSAT)")
else:
    st.sidebar.error(f"Critical Pivot Flagged ({live_csat}% CSAT)")

# Minimum Frequency Density Slider
st.sidebar.markdown("<hr style='margin: 15px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
st.sidebar.subheader("⚙️ Diagnostics Modifiers")
min_freq_filter = st.sidebar.slider("Min Phrase Occurrence Cutoff", min_value=1, max_value=5, value=1)

# Main Application Typography Headers
st.markdown('<div class="main-header">Customer Feedback Intel Engine Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Converting raw unstructured text streams into verified, boardroom-ready system verdicts.</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# WORKSPACE MODULE 1: INGESTION PIPELINE
# ----------------------------------------------------
if app_mode == "📥 Data Ingestion":
    st.header("📥 Ingestion Control Grid")
    tab1, tab2, tab3 = st.tabs(["📄 Document Data Load (CSV/Excel)", "✍️ Real-Time Stream Entry", "🔗 Live Scraping Array (Simulated)"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload review dataset sheet", type=["csv", "xlsx"])
        text_column = st.text_input("Specify Target Review Column Header String", value="Review_Text")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if text_column in df.columns:
                    st.session_state['reviews_df'] = pd.DataFrame({'Review_Text': df[text_column].dropna().astype(str)})
                    st.success(f"Connected framework pipeline to {len(st.session_state['reviews_df'])} text arrays!")
                    st.rerun()
                else:
                    st.error(f"Target Header parsing string '{text_column}' missing inside column vectors.")
            except Exception as e:
                st.error(f"Parsing execution anomaly caught: {e}")
                
    with tab2:
        manual_review = st.text_area("Paste unstructured reviews directly (Break lines to form individual data records)")
        if st.button("Commit Logs to Pipeline Pool"):
            if manual_review.strip():
                entries = [r.strip() for r in manual_review.split('\n') if r.strip()]
                new_df = pd.DataFrame({'Review_Text': entries})
                st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_df], ignore_index=True)
                st.success(f"Committed {len(entries)} text entries safely.")
                st.rerun()
                
    with tab3:
        st.text_input("Product Target URL Marketplace Node", placeholder="https://www.amazon.com/dp/B0XXXXXXXX")
        platform = st.selectbox("Marketplace Extraction Adapter", ["Amazon API Gateway", "Google Local Reviews Hook", "Trustpilot Core Cloud Extractor"])
        if st.button("Initialize Scraper Connections"):
            st.info(f"Opening background data threads to {platform}... System Configured.")
            st.success("Successfully scraped live streams into working data structures!")

    st.subheader("Active Repository Pool Check")
    st.dataframe(st.session_state['reviews_df'], use_container_width=True)

# ----------------------------------------------------
# WORKSPACE MODULE 2: GRAPHICAL EXECUTIVE DASHBOARD
# ----------------------------------------------------
elif app_mode == "📊 Core Dashboard":
    st.header("📊 Analytical Executive Summaries")
    
    total_reviews = len(df_processed)
    pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
    csat_score = round((pos_count / total_reviews) * 100) if total_reviews > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_reviews}</div><div class="metric-label">Analyzed Logs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #10B981;">{pos_count}</div><div class="metric-label">Positive Volume</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #EF4444;">{neg_count}</div><div class="metric-label">Negative Volume</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #3B82F6;">{csat_score}%</div><div class="metric-label">Aggregate CSAT Score</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
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

# ----------------------------------------------------
# WORKSPACE MODULE 3: DEEP-DIVE DIAGNOSTICS MATRIX
# ----------------------------------------------------
elif app_mode == "🔍 Deep-Dive Diagnostics":
    st.header("🔍 Feature Friction Breakdown & Pattern Discovery")
    
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Aspect-Based Sentiment Classifications (ABSA)")
        absa_grouped = df_aspects.groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count')
        fig_absa = px.bar(absa_grouped, x='Aspect', y='Count', color='Aspect_Sentiment',
                          barmode='stack',
                          color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'})
        st.plotly_chart(fig_absa, use_container_width=True)
        
    with d2:
        st.subheader("Contextual Common Phrase N-Grams")
        target_sent = st.selectbox("Filter phrase tracking streams by:", ["Negative", "Positive", "All"])
        text_pool = df_processed['Review_Text'].tolist() if target_sent == "All" else df_processed[df_processed['Sentiment'] == target_sent]['Review_Text'].tolist()
        ngrams_found = extract_ngrams(text_pool, n=2)
        
        # Apply the modifier input filter configuration from sidebar
        filtered_ngrams = [ng for ng in ngrams_found if ng[1] >= min_freq_filter]
        
        if filtered_ngrams:
            ng_df = pd.DataFrame(filtered_ngrams, columns=['Phrase N-Gram', 'Density Count'])
            fig_ng = px.bar(ng_df, x='Density Count', y='Phrase N-Gram', orientation='h',
                            color_discrete_sequence=['#2563EB'])
            st.plotly_chart(fig_ng, use_container_width=True)
        else:
            st.info(f"No repeating phrases found crossing the minimum occurrence cutoff of {min_freq_filter}.")

    st.subheader("Query Engine Entry Search Matrix")
    search_query = st.text_input("Filter localized metrics database using text tokens (e.g. 'battery', 'slow')...")
    df_search = df_processed.copy()
    if search_query:
        df_search = df_search[df_search['Review_Text'].str.contains(search_query, case=False)]
    st.dataframe(df_search, use_container_width=True)

# ----------------------------------------------------
# WORKSPACE MODULE 4: EXECUTIVE ACTION & REPORT GENERATOR ENGINE
# ----------------------------------------------------
elif app_mode == "🔮 Executive Verdict & Action Engine":
    st.header("🔮 AI Business Projections & Strategic Roadmap")
    
    total_reviews = len(df_processed)
    pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
    csat = (pos_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # THEME-PROOF PRODUCT HEALTH FINAL VERDICT BOX
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
        
    verdict_border_colors = {'🟢': '#10B981', '🟡': '#F59E0B', '🔴': '#EF4444'}
    st.markdown(f"""
    <div class="verdict-box" style="background-color: {verdict_color}; border-left: 6px solid {verdict_border_colors[verdict_status[0]]}; color: #1E293B !important;">
        <h4 style="margin: 0 0 12px 0; color: #0F172A !important; font-weight: 700; font-size: 19px;">{verdict_status}</h4>
        <p style="margin: 0; color: #1E293B !important; font-size: 15px; line-height: 1.6; font-weight: 500;">{verdict_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VERDICT DOWNLOAD COMPILER FOR BOARD MEETINGS
    neg_aspects = df_aspects[df_aspects['Aspect_Sentiment'] == 'Negative']['Aspect'].tolist()
    
    report_text = f"""==================================================
CUSTOMER FEEDBACK ANALYZER SYSTEM BOARD EXECUTIVE REPORT
Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

[METRIC POOL SUMMARY STATISTICS]
Total Logged Customer Reviews Analyzed: {total_reviews}
Positive Sentiment Volume: {pos_count}
Negative Sentiment Volume: {neg_count}
Calculated Customer Satisfaction (CSAT) Score: {csat:.2f}%

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
        
    report_text += "\n==================================================\nEnd of Compiled Feedback Analytics Matrix Report"

    st.download_button(
        label="📥 Download Final Executive Verdict Report (.TXT)",
        data=report_text,
        file_name=f"Executive_Verdict_Report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # COMPLIANCE CARDS WITH CONTRAST CORRECTIONS
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.subheader("💡 Suggested Changes & Product Improvements")
        st.markdown("System transformations derived directly from recurring tracking patterns inside unstructured data arrays:")
        
        if 'Product Quality' in neg_aspects:
            st.markdown('<div class="improvement-card"><strong>🔧 Physical R&D Material Stress Testing:</strong> Data entries note hardware structural vulnerabilities. Trigger immediate testing procedures to upgrade assembly alloy parameters and optimize internal cell thermodynamics.</div>', unsafe_allow_html=True)
        if 'Pricing & Value' in neg_aspects:
            st.markdown('<div class="improvement-card"><strong>🏷️ Value Strategy Structural Re-alignment:</strong> Price friction tracks across entry segments. Implement package unbundling structures, tiered credit plans, or targeted onboarding promos.</div>', unsafe_allow_html=True)
        if 'Customer Service' in neg_aspects or 'Shipping & Delivery' in neg_aspects:
            st.markdown('<div class="improvement-card"><strong>⚙️ Logistics Pipeline & Support Overhaul:</strong> Transaction records catch delivery execution delays. Re-evaluate shipping provider frameworks and introduce automated status tracking loops.</div>', unsafe_allow_html=True)
        if not neg_aspects:
            st.markdown('<div class="improvement-card" style="border-left-color: #10B981; background-color: #D1FAE5 !important;"><strong>✅ Performance Target Threshold Met:</strong> System workflows register below critical friction markers. Proceed with baseline iterative software maintenance schedules.</div>', unsafe_allow_html=True)

    with v_col2:
        st.subheader("🚨 Urgent CRM Customer Success Escalations")
        st.markdown("Individual feedback profiles displaying high brand risk, isolated for priority account manager intervention:")
        
        urgent_df = df_processed[(df_processed['Sentiment'] == 'Negative') & (df_processed['Emotion'].isin(['Anger', 'Frustration']))]
        if not urgent_df.empty:
            for idx, row in urgent_df.head(2).iterrows():
                st.markdown(f'<div class="red-flag-card"><strong>High Threat Log:</strong> "{row["Review_Text"]}" <br><small style="color: #991B1B !important; font-weight: 700; display: block; margin-top: 6px;">Isolated Root Vector: {row["Emotion"]} | Threat Level: Priority Tier-1 Escalation</small></div>', unsafe_allow_html=True)
        else:
            st.info("No volatile customer crisis markers triggered inside this current data block matrix.")

    # ACTION-ITEM ROADMAP MANAGEMENT MATRIX
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏁 Strategic Execution Checklist")
    st.checkbox("Deploy materials lab analysis sprint (R&D Department Task)", value=True if 'Product Quality' in neg_aspects else False)
    st.checkbox("Initialize regional carrier compliance audit (Logistics Operations Task)", value=True if 'Shipping & Delivery' in neg_aspects else False)
    st.checkbox("Review promotional tier packages discount schema (Finance Matrix Task)", value=True if 'Pricing & Value' in neg_aspects else False)

    # AUTOMATED CRM SYSTEM RESPONDER
    st.markdown("<hr style='margin: 30px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
    st.subheader("🤖 Smart CRM Automated Support Response Drafting")
    
    # Safe handling of fallback option array bindings to prevent out-of-index mutations
    review_options = df_processed['Review_Text'].tolist()
    if review_options:
        selected_text = st.selectbox("Select target historical entry to compile response template for:", review_options)
        target_rows = df_processed[df_processed['Review_Text'] == selected_text]
        if not target_rows.empty:
            target_row = target_rows.iloc[0]
            if st.button("Compile Contextual Support Blueprint"):
                if target_row['Sentiment'] == 'Negative':
                    email_draft = f"Subject: Resolution Protocol: Escalated Experience Account Review\n\nDear Customer,\n\nWe tracked your system user evaluation entry outlining service or component delivery friction: \"{selected_text}\". We sincerely apologize for the operational delays this has introduced to your workspace.\n\nOur customer engineering squad has transmitted this diagnostic index directly to our QA Director. We intend to process a priority product exchange or alternative credit remedy profile.\n\nPlease reply with your primary invoice ID code so we can fast-track resolution.\n\nBest regards,\nDirector of Customer Experience Operations"
                else:
                    email_draft = f"Subject: Deep Appreciation: Platform Operational Evaluation Log\n\nDear Customer,\n\nThank you for logging positive operational remarks regarding our platform layout matrices: \"{selected_text}\". We are thrilled to learn the system fits your active project needs!\n\nUser tracking logs like yours keep our sprint engineering targets aligned. Please enjoy verification token INTELFLUID15 for an exclusive 15% discount credit baseline drop on your upcoming account billing cycle.\n\nBest regards,\nDirector of Customer Experience Operations"
                st.text_area("Generated Corporate Response Blueprint Matrix:", value=email_draft, height=200)

# ARCHIVE SEGMENT EXPORTER CSV
st.sidebar.markdown("<hr style='margin: 20px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
st.sidebar.subheader("📥 Data Export Engine")
csv_data = df_processed.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Processed CSV File",
    data=csv_data,
    file_name="feedback_intel_export.csv",
    mime="text/csv"
)