import streamlit as tf
import pandas as pd
import nltk
import re
import importlib.util
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

if importlib.util.find_spec("sklearn.feature_extraction.text") is not None:
    sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
    TfidfVectorizer = sklearn_text.TfidfVectorizer
    _HAS_SKLEARN = True
else:
    TfidfVectorizer = None
    _HAS_SKLEARN = False
import plotly.express as px

# Ensure NLTK VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# --- Page Configuration ---
tf.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Header ---
tf.title("📊 AI-Driven Customer Feedback Analyzer")
tf.markdown("""
This application analyzes product reviews to extract sentiment distribution, 
identify primary customer complaints, calculate CSAT scores, and provide structured strategic action items.
""")
tf.markdown("---")

# --- Sidebar Inputs ---
tf.sidebar.header("📋 Input Configuration")

product_category = tf.sidebar.selectbox(
    "Select Product Category:",
    ["Electronics", "Apparel & Fashion", "Software/SaaS", "Food & Beverage", "Healthcare", "Other"]
)

input_method = tf.sidebar.radio("Choose Input Method:", ("Upload CSV/Excel File", "Paste Raw Reviews"))

reviews_list = []

if input_method == "Upload CSV/Excel File":
    uploaded_file = tf.sidebar.file_uploader("Upload a file containing reviews", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            # Allow user to pick the column containing text reviews
            column_options = df_input.columns.tolist()
            text_col = tf.sidebar.selectbox("Select the column containing the reviews:", column_options)
            reviews_list = df_input[text_col].dropna().astype(str).tolist()
        except Exception as e:
            tf.sidebar.error(f"Error reading file: {e}")

else:
    raw_text = tf.sidebar.text_area(
        "Paste reviews here (Place each individual review on a new line):",
        height=200,
        placeholder="The battery life is terrible.\nI absolutely loved the customer service!\nGreat build quality but shipping was slow."
    )
    if raw_text.strip():
        reviews_list = [line.strip() for line in raw_text.split('\n') if line.strip()]

# --- Main Processing Logic ---
if reviews_list:
    total_reviews = len(reviews_list)
    
    # 1. Process Sentiments
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    negative_reviews = []
    all_scores = []
    
    for review in reviews_list:
        score = sia.polarity_scores(review)['compound']
        all_scores.append(score)
        
        if score >= 0.05:
            positive_count += 1
        elif score <= -0.05:
            negative_count += 1
            negative_reviews.append(review)
        else:
            neutral_count += 1

    # 2. Key Metrics Calculation
    # Customer Satisfaction Score (CSAT) approximation based on positive sentiment percentage
    csat_score = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0

    # --- Dashboard Layout ---
    col1, col2, col3 = tf.columns(3)
    with col1:
        tf.metric(label="Total Reviews Analyzed", value=total_reviews)
    with col2:
        tf.metric(label="Target Product Category", value=product_category)
    with col3:
        tf.metric(label="Calculated CSAT Score", value=f"{csat_score:.1f}%")

    tf.markdown("---")

    # --- Visualization Section ---
    chart_col, details_col = tf.columns([3, 2])
    
    with chart_col:
        tf.subheader("📈 Sentiment Distribution")
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [positive_count, neutral_count, negative_count]
        })
        fig = px.pie(
            sentiment_data, 
            values='Count', 
            names='Sentiment', 
            color='Sentiment',
            color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#ffbb78', 'Negative': '#d62728'},
            hole=0.4
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        tf.plotly_chart(fig, use_container_width=True)

    with details_col:
        tf.subheader("📋 Analysis Summary Breakdown")
        tf.write(f"🟢 **Positive Feedback:** {positive_count} ({ (positive_count/total_reviews)*100:.1f}%)")
        tf.write(f"🟡 **Neutral Feedback:** {neutral_count} ({ (neutral_count/total_reviews)*100:.1f}%)")
        tf.write(f"🔴 **Negative Feedback:** {negative_count} ({ (negative_count/total_reviews)*100:.1f}%)")
        
        # Display automated warning flag for high negativity
        if (negative_count / total_reviews) > 0.35:
            tf.error("⚠️ **Urgent Operational Flag:** Negative sentiment exceeds 35%. Immediate product/service review recommended.")

    tf.markdown("---")

    # --- Topic Modeling / Common Complaints Extraction ---
    tf.subheader("🔍 Core Complaints & Keyword Vectors")
    
    if len(negative_reviews) >= 2:
        try:
            # Using TF-IDF to discover unique keywords driving negative reviews
            if _HAS_SKLEARN and TfidfVectorizer is not None:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
                vectorizer.fit(negative_reviews)
                keywords = vectorizer.get_feature_names_out()
            else:
                # Fallback: simple frequency-based keyword extraction
                stopwords = set([
                    'the','and','is','in','it','of','to','a','for','with','this','that','on','was','are','but','have','has'
                ])
                words = []
                for r in negative_reviews:
                    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", r.lower())
                    words.extend([t for t in tokens if t not in stopwords])
                keywords = [w for w, _ in Counter(words).most_common(5)]

            if len(keywords) > 0:
                tf.markdown("Based on standard text clustering algorithms, the top complaints revolve around these key thematic vectors:")
                for kw in keywords:
                    tf.markdown(f"- **Issue Area identified:** *{kw.capitalize()}* related concerns.")
            else:
                tf.markdown("💡 *Not enough diverse negative feedback text data to isolate recurring structural keywords accurately.*")
        except Exception:
            tf.markdown("💡 *Not enough diverse negative feedback text data to isolate recurring structural keywords accurately.*")
    else:
        tf.markdown("✅ *Negative feedback density is too low to extract distinct trending complaint clusters.*")

    tf.markdown("---")

    # --- Final Verdict & AI Suggestions Engine ---
    tf.subheader("🧠 Final Verdict & Strategic Action Matrix")
    
    # Logic matrix to simulate deterministic enterprise-grade decision metrics
    if csat_score >= 75:
        verdict_status = "Excellent Market Positioning"
        verdict_color = "green"
        verdict_text = f"The product is executing excellently within the **{product_category}** segment. The consumer base highlights major value propositions, keeping negative trends below threshold targets."
        suggestions = [
            f"**Capitalize on Strength:** Double down on marketing campaigns accentuating the features mentioned positively in the text logs.",
            "**Scale Support Infrastructure:** Ensure customer success teams maintain current speed of response to protect this high CSAT benchmark.",
            "**Iterative Retention:** Implement loyalty point milestones for current active users to lock in market share."
        ]
    elif 40 <= csat_score < 75:
        verdict_status = "Moderate Performance Risk"
        verdict_color = "orange"
        verdict_text = f"The feedback signals standard core capabilities but indicates emerging structural frictions within the **{product_category}** workflow. Unresolved minor bugs/complaints threaten user retention metrics long-term."
        suggestions = [
            "**Isolate Friction Points:** Audit the engineering/design logs specifically touching upon the isolated complaint keywords shown above.",
            "**Proactive Outreach:** Initiate an automated response framework to actively remedy issues expressed by neutral/negative reviewers.",
            "**Quality Control Interventions:** Cross-verify vendor or software component stability to weed out sporadic product failures."
        ]
    else:
        verdict_status = "Critical Operational Intervention Required"
        verdict_color = "red"
        verdict_text = f"The asset is severely underperforming in the **{product_category}** market tier. Severe churn is highly probable due to systemic breakdowns across core functional expectations."
        suggestions = [
            "**Emergency Patch/Recall Implementation:** Halt minor feature updates and redirect dev/manufacturing sprint cycles exclusively to address primary user complaints.",
            "**Reputation Management Protocol:** Deploy senior support personnel to connect with highly dissatisfied high-value corporate/retail accounts.",
            "**Root-Cause Process Overhaul:** Re-evaluate foundational manufacturing quality gates or software architecture baselines to trace structural flaws."
        ]

    # Render Final Verdict Block
    tf.markdown(f"#### **Status Baseline:** :{verdict_color}[{verdict_status}]")
    tf.info(verdict_text)
    
    # Render Action Items
    tf.markdown("#### **Prioritized Strategic Action Plan:**")
    for action in suggestions:
        tf.markdown(action)

else:
    tf.info("👋 Welcome! Please configure your product category and paste reviews or upload a CSV file in the sidebar to start processing.")