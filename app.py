import streamlit as st
import pandas as pd
import plotly.express as px
import time
import io

# ==========================================
# 1. PAGE CONFIGURATION & THEME CONSTANTS
# ==========================================
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling overrides via CSS Injection
st.markdown("""
    <style>
        .main { background-color: #FAFAFC; }
        .stMetric { background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        div[data-testid="stExpander"] { background-color: #FFFFFF; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. USER AUTHENTICATION & SESSION STATE
# ==========================================
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def login_form():
    st.markdown("<h2 style='text-align: center;'>🔐 Enterprise Access Gate</h2>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_gate"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                # Correctly structured native Streamlit submit handler
                submitted = st.form_submit_button("Authenticate Session")
                
                if submitted:
                    if username == "admin" and password == "demo123":  # Demo credentials
                        st.session_state['authenticated'] = True
                        st.success("Access Granted.")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please use admin / demo123")

if not st.session_state['authenticated']:
    login_form()
    st.stop()

# ==========================================
# 3. MOCK ALGORITHMS (NLP / SERVQUAL / CSAT)
# ==========================================
def analyze_text_sentiment_and_servqual(text):
    """
    Simulates academic NLP parsing. Maps input string parameters to fine-tuned VADER 
    or operational categories.
    """
    text_lower = text.lower()
    
    # Deterministic fallback logic to construct high-quality demo pipelines
    if any(w in text_lower for w in ["slow", "delayed", "wait", "time", "respond"]):
        dimension = "Responsiveness"
        sentiment = "Negative"
        score = 1
    elif any(w in text_lower for w in ["broken", "error", "crash", "bug", "failed", "fail"]):
        dimension = "Reliability"
        sentiment = "Negative"
        score = 1
    elif any(w in text_lower for w in ["rude", "bad attitude", "unhelpful", "staff"]):
        dimension = "Empathy"
        sentiment = "Negative"
        score = 2
    elif any(w in text_lower for w in ["great", "love", "awesome", "fast", "perfect", "helpful"]):
        dimension = "Assurance"
        sentiment = "Positive"
        score = 5
    else:
        dimension = "Tangibles"
        sentiment = "Neutral"
        score = 3
        
    return sentiment, dimension, score

def generate_mock_insights(dimension):
    """Generates targeted strategic consultancy advice based on SERVQUAL vectors."""
    insights = {
        "Responsiveness": [
            "🚨 **Bottleneck Warning:** High frequency of latency complaints detected in support pipelines.",
            "👉 **Action Item 1:** Implement tiered routing for Tier-1 support tickets to reduce initial triage wait times.",
            "👉 **Action Item 2:** Spin up standard automated macro-responses for frequent transactional bottlenecks."
        ],
        "Reliability": [
            "🚨 **Systemic Fault Warning:** Core application architecture stability is negatively driving your CSAT metrics.",
            "👉 **Action Item 1:** Coordinate a DevOps infrastructure fire-drill to map API drop-offs.",
            "👉 **Action Item 2:** Introduce localized circuit breakers in your primary user checkout sequence."
        ],
        "Empathy": [
            "🚨 **Frontline Friction:** Sentiment dips indicate tone and soft skills friction during customer transitions.",
            "👉 **Action Item 1:** Mandate customer handling alignment workshops for the incoming Support cohort.",
            "👉 **Action Item 2:** Redesign escalation paths to guarantee a white-glove experience for high-LTV users."
        ],
        "Tangibles": [
            "🚨 **Interface/Asset Friction:** User interface layout clarity is disrupting operational UX.",
            "👉 **Action Item 1:** Audit the styling clarity of primary transaction buttons for accessibility compliance."
        ],
        "Assurance": [
            "💡 **Strength Capitalization:** Positive security and institutional trust sentiment is trending high.",
            "👉 **Action Item 1:** Highlight infrastructure uptime certificates in public-facing product marketing copy."
        ]
    }
    return insights.get(dimension, ["👉 Monitor incoming volume vectors for subtle quality indicators."])

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brainstorming.png", width=80)
    st.title("Feedback Analyzer")
    st.markdown("🌐 *System Standard: Academic Frameworks Edition (SERVQUAL & CSAT)*")
    st.divider()
    
    st.subheader("Session Meta")
    st.info("User: Senior Administrator\nRole: Product Management")
    
    if st.button("Log Out"):
        st.session_state['authenticated'] = False
        st.rerun()

# ==========================================
# 5. CORE APPLICATION WORKFLOW (UI tabs)
# ==========================================
st.markdown("# 🧠 Customer Feedback Analytics Command Center")
st.markdown("Transform unstructured consumer expressions into verifiable business health intelligence metrics.")
st.divider()

tab1, tab2 = st.tabs(["📝 Ad-Hoc Input Triage", "📁 Batch File Ingestion Engine"])

# --- TAB 1: SINGLE RECORD AD-HOC INPUT ---
with tab1:
    st.subheader("Real-Time Single Review Triage")
    with st.form("single_review_form"):
        user_review = st.text_area(
            "Paste Raw Customer Review Message:",
            placeholder="Type or paste the raw text payload received from your end customer channel...",
            help="Minimum 10 characters required for analytical validation processing."
        )
        submit_review = st.form_submit_button("Execute Text Analytics Pipeline")
        
    if submit_review:
        if len(user_review.strip()) < 10:
            st.warning("Validation Failed: The feedback block entered is too concise. Minimum length required is 10 characters.")
        else:
            with st.spinner("🧠 Initializing NLP Models..."):
                time.sleep(0.4)
                sentiment, dimension, score = analyze_text_sentiment_and_servqual(user_review)
            
            # Formulate individual output metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Inferred Sentiment Category", sentiment)
            with c2:
                st.metric("Mapped SERVQUAL Attribute", dimension)
            with c3:
                st.metric("Calculated Likert CSAT Value", f"{score} / 5")
                
            st.divider()
            st.subheader("💡 Prescriptive Strategic Action Plan")
            
            insights_list = generate_mock_insights(dimension)
            for item in insights_list:
                st.write(item)

# --- TAB 2: BATCH DATASET PROCESSING ---
with tab2:
    st.subheader("High-Throughput File Dropzone")
    uploaded_file = st.file_uploader(
        "Upload raw operational feedback files", 
        type=["csv", "xlsx"],
        help="System strictly enforces processing capability limits up to 200MB natively per session upload."
    )
    
    if uploaded_file is not None:
        try:
            # Safely determine parsing library mappings
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
                
            st.success(f"File ingestion successful. Found {len(df_raw)} total operational logs.")
            
            # Target Column Mapping Layer
            target_col = st.selectbox(
                "Map your dataset review text target column:",
                options=df_raw.columns,
                help="Select the column containing the raw feedback strings."
            )
            
            if st.button("Execute High-Throughput Matrix Processing"):
                with st.spinner("Processing NLP Vector Maps & Calculating CSAT Matrices..."):
                    sentiments, dimensions, scores = [], [], []
                    
                    for text in df_raw[target_col].astype(str):
                        sent, dim, scr = analyze_text_sentiment_and_servqual(text)
                        sentiments.append(sent)
                        dimensions.append(dim)
                        scores.append(scr)
                        
                    df_raw['Inferred_Sentiment'] = sentiments
                    df_raw['SERVQUAL_Dimension'] = dimensions
                    df_raw['Computed_CSAT_Score'] = scores
                    
                    # Store to partition data explicitly inside the runtime environment memory
                    st.session_state['processed_df'] = df_raw
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Critical Ingestion Interrupt: {str(e)}")

# ==========================================
# 6. DASHBOARD & VISUALIZATION LAYER
# ==========================================
if 'processed_df' in st.session_state:
    df_analysed = st.session_state['processed_df']
    
    st.divider()
    st.markdown("## 📊 Macro Analytics Insights Dashboard")
    
    # Calculate Academic Metric Formulations
    total_reviews = len(df_analysed)
    pos_reviews = len(df_analysed[df_analysed['Inferred_Sentiment'] == 'Positive'])
    neu_reviews = len(df_analysed[df_analysed['Inferred_Sentiment'] == 'Neutral'])
    neg_reviews = len(df_analysed[df_analysed['Inferred_Sentiment'] == 'Negative'])
    
    calculated_csat = ((pos_reviews + (0.5 * neu_reviews)) / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Render Master Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calculated Net CSAT Index", f"{calculated_csat:.1f}%")
    m2.metric("Total Batch Records Ingested", total_reviews)
    m3.metric("Total Negative Risks Flagged", neg_reviews, delta="- Action Required", delta_color="inverse")
    m4.metric("Positive Customer Interactions", pos_reviews)
    
    st.write("")
    
    # Render Visual Dashboard Plots via Grid Structuring
    g1, g2 = st.columns(2)
    
    with g1:
        st.markdown("### Sentiment Volume Contribution Profile")
        fig_pie = px.pie(
            df_analysed, 
            names='Inferred_Sentiment',
            color='Inferred_Sentiment',
            color_discrete_map={'Positive': '#2ECC71', 'Neutral': '#F1C40F', 'Negative': '#E74C3C'},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g2:
        st.markdown("### Systemic Failures mapped by SERVQUAL Framework")
        dim_counts = df_analysed['SERVQUAL_Dimension'].value_counts().reset_index()
        dim_counts.columns = ['Dimension', 'Incident Frequency Count']
        
        fig_bar = px.bar(
            dim_counts, 
            x='Dimension', 
            y='Incident Frequency Count',
            color='Dimension',
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # 7. PRESCRIPTIVE STRATEGIC TRIAGE SECTION
    # ==========================================
    st.markdown("### ⚡ AI Prescriptive Action Items (Filtered on Core Operational Vulnerabilities)")
    
    unique_neg_dims = df_analysed[df_analysed['Inferred_Sentiment'] == 'Negative']['SERVQUAL_Dimension'].unique()
    
    if len(unique_neg_dims) == 0:
        st.success("🎉 Operational Excellence Metric Cleared: Zero negative feedback loops identified in this file matrix.")
    else:
        for dim in unique_neg_dims:
            with st.expander(f"🔴 System Deficit Detected within Category Vector: {dim}"):
                insights = generate_mock_insights(dim)
                for ins in insights:
                    st.write(ins)
                    
    # ==========================================
    # 8. EXPORT CHANNELS
    # ==========================================
    st.divider()
    st.markdown("### 💾 Report Archival & Compilation Extraction")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv_buffer = io.StringIO()
        df_analysed.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Structured Analytics Report (.CSV)",
            data=csv_data,
            file_name="Operational_Feedback_Analysis_Export.csv",
            mime="text/csv"
        )
        
    with col_dl2:
        report_text = f"""CUSTOM FEEDBACK ANALYZER EXECUTIVE SUMMARY REPORT
============================================================
Calculated Net CSAT Index: {calculated_csat:.2f}%
Total Records Ingested: {total_reviews}
Total Negative Risk Alerts Flagged: {neg_reviews}
============================================================
Generated via Streamlit Academic Framework System Engine v1.0"""
        
        st.download_button(
            label="📄 Download Executive Memo Briefing (.TXT)",
            data=report_text,
            file_name="Executive_Feedback_Summary_Briefing.txt",
            mime="text/plain"
        )
