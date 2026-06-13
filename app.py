import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="Advanced Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, corporate-ready UI
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #111827;
    }
    .metric-label {
        font-size: 14px;
        color: #6B7280;
        margin-top: 5px;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .action-card {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    .improvement-card {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    .red-flag-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Define Rule-Based NLP Pipeline Functions to keep app lightweight yet fully functional
def analyze_review_sentiment(text):
    text_lower = text.lower()
    pos_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'satisfied', 'awesome', 'fast', 'smooth', 'impressed', 'wonderful']
    neg_words = ['bad', 'worst', 'terrible', 'horrible', 'hate', 'broken', 'disappointed', 'slow', 'expensive', 'waste', 'useless', 'fail', 'defect', 'poor', 'annoying']
    
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
        if any(w in text_lower for w in ['broken', 'defect', 'fail', 'waste', 'poor']): return 'Frustration'
        if any(w in text_lower for w in ['slow', 'delay', 'wait', 'annoying']): return 'Impatience'
        if any(w in text_lower for w in ['terrible', 'worst', 'hate']): return 'Anger'
        return 'Disappointment'
    elif sentiment == 'Positive':
        if any(w in text_lower for w in ['amazing', 'excellent', 'perfect', 'best', 'wonderful']): return 'Delight'
        return 'Satisfaction'
    return 'Neutral / Indifferent'

def assign_aspects_and_scores(text):
    text_lower = text.lower()
    aspects = {}
    
    rules = {
        'Product Quality': ['quality', 'broken', 'screen', 'battery', 'build', 'material', 'durable', 'defect', 'hardware', 'performs'],
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
    stop_words = {'the', 'a', 'and', 'is', 'i', 'to', 'this', 'it', 'of', 'for', 'in', 'with', 'was', 'but', 'on', 'that', 'my', 'you', 'have', 'with'}
    
    for text in texts:
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in clean_text.split() if w not in stop_words and len(w) > 2]
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            words_list.append(ngram)
            
    return Counter(words_list).most_common(5)

# Sidebar layout navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1486/1486433.png", width=80)
st.sidebar.title("Navigation Panel")
app_mode = st.sidebar.radio("Go to:", ["📥 Data Ingestion", "📊 Core Dashboard", "🔍 Deep-Dive Diagnostics", "🔮 Executive Verdict & Action Engine"])

# Initialize session state for holding review data if empty
if 'reviews_df' not in st.session_state:
    # Comprehensive default evaluation data
    default_data = pd.DataFrame({
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
    st.session_state['reviews_df'] = default_data

# Header Section
st.markdown('<div class="main-header">Customer Feedback Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transforming unstructured user reviews into structured, actionable business intelligence.</div>', unsafe_allow_html=True)

# Run batch analysis pipeline globally on currently selected memory block data pool
df_processed = st.session_state['reviews_df'].copy()
df_processed['Sentiment'] = df_processed['Review_Text'].apply(analyze_review_sentiment)
df_processed['Emotion'] = df_processed.apply(lambda row: detect_emotion(row['Review_Text'], row['Sentiment']), axis=1)

# Aspect Extraction Processing Loop
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

# ----------------------------------------------------
# 1. DATA INGESTION LAYER
# ----------------------------------------------------
if app_mode == "📥 Data Ingestion":
    st.header("📥 Feedback Ingestion Panel")
    
    tab1, tab2, tab3 = st.tabs(["📄 Upload File (CSV/Excel)", "✍️ Direct Text Input", "🔗 Live URL Scraping (Simulated)"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload your dataset containing standard customer reviews", type=["csv", "xlsx"])
        text_column = st.text_input("Enter the name of the column containing reviews", value="Review_Text")
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if text_column in df.columns:
                    st.session_state['reviews_df'] = pd.DataFrame({'Review_Text': df[text_column].dropna().astype(str)})
                    st.rerun()
                else:
                    st.error(f"Column '{text_column}' not found in the uploaded file. Please verify column headers.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    with tab2:
        manual_review = st.text_area("Paste customer reviews here (One review per line)")
        if st.button("Append Manual Input"):
            if manual_review.strip():
                new_reviews = [r.strip() for r in manual_review.split('\n') if r.strip()]
                new_df = pd.DataFrame({'Review_Text': new_reviews})
                st.session_state['reviews_df'] = pd.concat([st.session_state['reviews_df'], new_df], ignore_index=True)
                st.rerun()
            else:
                st.warning("Please type or paste some text first.")
                
    with tab3:
        st.text_input("Product Review Marketplace URL", placeholder="https://www.amazon.com/dp/B0XXXXXXXX")
        platform = st.selectbox("Select Target Marketplace", ["Amazon", "Google Maps Reviews", "Trustpilot", "App Store"])
        if st.button("Trigger Dynamic Web Scraper Pipeline"):
            st.info(f"Simulating API pipeline hook connection to {platform} reviews... Integration ready.")
            st.success("Fetched latest live product reviews into data stream successfully!")

    st.subheader("Current Review Data Pool")
    st.dataframe(st.session_state['reviews_df'], use_container_width=True)

# ----------------------------------------------------
# 2. CORE ANALYTICS DASHBOARD
# ----------------------------------------------------
elif app_mode == "📊 Core Dashboard":
    st.header("📊 Executive Analytics Insights")
    
    total_reviews = len(df_processed)
    pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
    csat_score = round((pos_count / total_reviews) * 100) if total_reviews > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_reviews}</div><div class="metric-label">Total Analyzed Reviews</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #10B981;">{pos_count}</div><div class="metric-label">Positive Reviews</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #EF4444;">{neg_count}</div><div class="metric-label">Negative Reviews</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #3B82F6;">{csat_score}%</div><div class="metric-label">Calculated CSAT Score</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Sentiment Distribution Metric")
        sent_counts = df_processed['Sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(sent_counts, values='Count', names='Sentiment', 
                         color='Sentiment',
                         color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g2:
        st.subheader("Granular Emotion Spectrum Profile")
        emotion_counts = df_processed['Emotion'].value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        fig_bar = px.bar(emotion_counts, x='Count', y='Emotion', orientation='h',
                         color='Emotion',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------------------------------
# 3. DEEP-DIVE DIAGNOSTICS
# ----------------------------------------------------
elif app_mode == "🔍 Deep-Dive Diagnostics":
    st.header("🔍 Diagnostics Breakdown (Root Cause Exploration)")
    
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Aspect-Based Sentiment Distribution (ABSA)")
        absa_grouped = df_aspects.groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count')
        fig_absa = px.bar(absa_grouped, x='Aspect', y='Count', color='Aspect_Sentiment',
                          barmode='stack',
                          color_discrete_map={'Positive':'#10B981','Negative':'#EF4444','Neutral':'#9CA3AF'})
        st.plotly_chart(fig_absa, use_container_width=True)
        
    with d2:
        st.subheader("Top Contextual Common Phrase N-Grams")
        target_sent = st.selectbox("Filter N-grams by Feedback Stream Context", ["Negative", "Positive", "All"])
        
        text_pool = df_processed['Review_Text'].tolist() if target_sent == "All" else df_processed[df_processed['Sentiment'] == target_sent]['Review_Text'].tolist()
        ngrams_found = extract_ngrams(text_pool, n=2)
        
        if ngrams_found:
            ng_df = pd.DataFrame(ngrams_found, columns=['Phrase N-Gram', 'Frequency Density'])
            fig_ng = px.bar(ng_df, x='Frequency Density', y='Phrase N-Gram', orientation='h',
                            color_discrete_sequence=['#60A5FA'])
            st.plotly_chart(fig_ng, use_container_width=True)
        else:
            st.info("Insufficient phrase data for context.")

    st.subheader("Review Data Search Matrix")
    search_query = st.text_input("Enter keyword search filter...")
    df_search = df_processed.copy()
    if search_query:
        df_search = df_search[df_search['Review_Text'].str.contains(search_query, case=False)]
    st.dataframe(df_search, use_container_width=True)

# ----------------------------------------------------
# 4. EXECUTIVE VERDICT & ACTION ENGINE
# ----------------------------------------------------
elif app_mode == "🔮 Executive Verdict & Action Engine":
    st.header("🔮 AI Executive Verdict & Engineering Recommendations")
    
    total_reviews = len(df_processed)
    pos_count = len(df_processed[df_processed['Sentiment'] == 'Positive'])
    neg_count = len(df_processed[df_processed['Sentiment'] == 'Negative'])
    csat = (pos_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # FEATURE 1: AUTOMATED PRODUCT FINAL VERDICT 
    st.subheader("📋 Product Health Final Verdict")
    if csat >= 75:
        verdict_status = "🟢 STRONGLY APPROVED (MARKET LEADER)"
        verdict_color = "#ECFDF5"
        verdict_text = f"The product performs exceptionally well with a high customer satisfaction rating of {csat:.1f}%. Core value propositions are fully validated by the consumer base. Promising outlook; prioritize scaling marketing and production inventory."
    elif csat >= 45:
        verdict_status = "🟡 CAUTION: CONDITIONALLY APPROVED (MARKET CONTESTER)"
        verdict_color = "#FFFBEB"
        verdict_text = f"The product occupies a volatile market position with a satisfaction rate of {csat:.1f}%. While specific design iterations display strong adoption vectors, structural flaws or customer service lag are throttling growth. Quality optimization is required immediately."
    else:
        verdict_status = "🔴 HIGH RISK: CRITICAL INTERVENTION REQUIRED"
        verdict_color = "#FEF2F2"
        verdict_text = f"The product is experiencing intense market churn with an unsatisfactory rating baseline ({csat:.1f}% CSAT). High occurrence of critical negative feedback flags hardware or systemic performance failure. Immediate product halt or pivot suggested."
        
    color_map = {
        "🟢": "#10B981",
        "🟡": "#F59E0B",
        "🔴": "#EF4444",
    }
    border_color = color_map[verdict_status[0]]
    st.markdown(f"""
    <div class="verdict-box" style="background-color: {verdict_color}; border-left: 6px solid {border_color};">
        <h4 style="margin: 0 0 10px 0; color: #111827;">{verdict_status}</h4>
        <p style="margin: 0; color: #374151; font-size: 15px; line-height: 1.6;">{verdict_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # FEATURE 2: SUGGESTED CHANGES AND IMPROVEMENTS
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.subheader("💡 Suggested Changes & Product Improvements")
        st.markdown("Engineering and architectural recommendations derived directly from customer complaints data arrays:")
        
        neg_aspects = df_aspects[df_aspects['Aspect_Sentiment'] == 'Negative']['Aspect'].tolist()
        
        if 'Product Quality' in neg_aspects:
            st.markdown('<div class="improvement-card"><strong>🔧 Physical R&D Revision:</strong> Reviews note physical fragile tolerances. Schedule an immediate engineering stress test to swap out low-durability materials or fix battery assembly bottlenecks.</div>', unsafe_allow_html=True)
        if 'Pricing & Value' in neg_aspects:
            st.markdown('<div class="improvement-card"><strong>🏷️ Pricing Strategy Realignment:</strong> Friction detected regarding perceived cost-to-benefit metrics. Introduce flexible micro-tier packages, seasonal discount matrices, or bonus feature padding.</div>', unsafe_allow_html=True)
        if 'Customer Service' in neg_aspects or 'Shipping & Delivery' in neg_aspects:
            st.markdown('<div class="improvement-card"><strong>⚙️ Operations Optimization:</strong> Streamline operational overhead by adopting automated status updates and adding predictive logistics triggers.</div>', unsafe_allow_html=True)
        if not neg_aspects:
            st.markdown('<div class="improvement-card" style="border-left-color: #10B981; background-color: #ECFDF5;"><strong>✅ No Adjustments Mandatory:</strong> High performance baseline. Focus efforts strictly on minor iterative polish items.</div>', unsafe_allow_html=True)

    with v_col2:
        st.subheader("🚨 Urgent Operational Escalation Flags")
        st.markdown("High-priority severe friction feedback needing instant support operations triage:")
        
        urgent_df = df_processed[(df_processed['Sentiment'] == 'Negative') & (df_processed['Emotion'].isin(['Anger', 'Frustration']))]
        if not urgent_df.empty:
            for idx, row in urgent_df.head(2).iterrows():
                st.markdown(f'<div class="red-flag-card"><strong>Flagged Feedback:</strong> "{row["Review_Text"]}" <br><small style="color: #B91C1C;">Isolated Threat Emotion: {row["Emotion"]} | Priority Level: Critical Priority</small></div>', unsafe_allow_html=True)
        else:
            st.info("No urgent customer crisis triggers active inside current data filters.")

    # Smart response generation block
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🤖 Smart CRM Automated Support Response Drafting")
    selected_text = st.selectbox("Select target text to reply to:", df_processed['Review_Text'].tolist())
    target_row = df_processed[df_processed['Review_Text'] == selected_text].iloc[0]
    
    if st.button("Generate Contextual Email Draft Template"):
        if target_row['Sentiment'] == 'Negative':
            email_draft = f"Subject: Regrettable Experience with Your Order - Assistance Requested\n\nDear Customer,\n\nWe noticed your recent review highlighting an issue: \"{selected_text}\". We sincerely apologize for the frustration.\n\nOur engineering team is routing this tracking log directly to our Quality Assurance department. We want to process an immediate resolution.\n\nWarm regards,\nCustomer Success Team"
        else:
            email_draft = f"Subject: Thank You for Your Valued Feedback!\n\nDear Customer,\n\nThank you so much for sharing your feedback: \"{selected_text}\". We are absolutely thrilled to hear about your great experience!\n\nBest regards,\nCustomer Experience Team"
        st.text_area("Generated Enterprise Response Template:", value=email_draft, height=200)

# Export Functionality Sidebar
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("📥 Export Analyzed Dataset")
csv_data = df_processed.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Processed CSV Structure",
    data=csv_data,
    file_name="processed_customer_feedback.csv",
    mime="text/csv"
)