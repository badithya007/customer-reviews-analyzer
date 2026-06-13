import streamlit as st
import pandas as pd
import plotly.express as px
import importlib

try:
    vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
except ImportError:
    vader_module = importlib.import_module("vaderSentiment")

SentimentIntensityAnalyzer = getattr(vader_module, "SentimentIntensityAnalyzer")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rule-based dictionary for Product Categories & Key Themes
THEME_DICTIONARY = {
    "Shipping & Delivery": ["delivery", "shipping", "late", "delay", "arrived", "package", "post", "track"],
    "Product Quality": ["broken", "damaged", "cheap", "quality", "material", "tore", "broke", "plastic", "faulty"],
    "Customer Service": ["support", "service", "refund", "return", "agent", "manager", "chat", "email", "called"],
    "Pricing & Value": ["expensive", "price", "worth", "cost", "waste", "money", "cheap", "overpriced"],
    "Usability & App": ["app", "website", "interface", "bug", "error", "crash", "slow", "login", "button"]
}

# Sample Dataset for immediate testing
SAMPLE_REVIEWS = [
    {"Review": "The delivery was delayed by 4 days. Package arrived damaged."},
    {"Review": "Excellent product quality! Very sturdy and worth every penny."},
    {"Review": "Customer service was absolutely terrible. Refused to give me a refund."},
    {"Review": "The new app update keeps crashing whenever I try to checkout."},
    {"Review": "Way too expensive for such cheap plastic material. Disappointed."},
    {"Review": "Fast shipping and fantastic material. Highly recommend!"},
    {"Review": "The agent on live chat was extremely helpful and resolved my issue fast."},
    {"Review": "I received the wrong item, and shipping took forever."},
    {"Review": "Great value for money, simple interface, and fast checkout."},
    {"Review": "The product broke after just two days of normal use."},
]

# ==========================================
# HELPER FUNCTIONS (CORE ENGINE)
# ==========================================
def analyze_sentiment(text: str) -> dict:
    """Analyzes text using VADER and classifies into Positive, Negative, or Neutral."""
    if not isinstance(text, str) or text.strip() == "":
        return {"score": 0.0, "sentiment": "Neutral"}
    
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return {"score": compound, "sentiment": sentiment}

def extract_reasons_and_themes(text: str) -> tuple:
    """Maps reviews to categories and extracts problem keywords cleanly."""
    text_lower = str(text).lower()
    matched_themes = []
    matched_keywords = []
    
    for theme, keywords in THEME_DICTIONARY.items():
        for kw in keywords:
            if kw in text_lower:
                matched_keywords.append(kw)
                if theme not in matched_themes:
                    matched_themes.append(theme)
                    
    # Fallback safe values if no keywords match
    if not matched_themes:
        matched_themes.append("General / Unclassified")
    if not matched_keywords:
        matched_keywords.append("general feedback")
        
    return ", ".join(matched_themes), matched_keywords

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw dataframe by running sentiment and categorization pipelines."""
    if "Review" not in df.columns:
        raise ValueError("The dataset must contain a column named 'Review'.")
    
    # Drop rows where the Review column is completely empty/null
    df = df.dropna(subset=["Review"]).copy()
    df["Review"] = df["Review"].astype(str)
    
    if df.empty:
        return df

    # Process sentiments
    sentiments = df["Review"].apply(analyze_sentiment)
    df["Sentiment Score"] = [s["score"] for s in sentiments]
    df["Sentiment"] = [s["sentiment"] for s in sentiments]
    
    # Process Categories and Reasons
    themes_and_reasons = df["Review"].apply(extract_reasons_and_themes)
    df["Inferred Category"] = [tr[0] for tr in themes_and_reasons]
    df["Keywords/Reasons"] = [tr[1] for tr in themes_and_reasons]
    
    return df

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("📊 Advanced Customer Feedback Analyzer")
st.markdown("Convert messy consumer text reviews into structured, actionable business intelligence instantly.")

# Sidebar Configuration
st.sidebar.header("📥 Data Source Selection")
data_source = st.sidebar.radio(
    "Choose Input Method:", 
    ["Upload Dataset (CSV/XLSX)", "Paste Raw Reviews", "Load Sample Data"]
)

raw_df = None

# Logic for data source extraction with robust structural validation
if data_source == "Upload Dataset (CSV/XLSX)":
    uploaded_file = st.sidebar.file_uploader("Upload review file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            if "Review" not in raw_df.columns:
                st.sidebar.error("❌ Critical Error: Missing a column named 'Review'")
                raw_df = None
            else:
                st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

elif data_source == "Paste Raw Reviews":
    text_input = st.sidebar.text_area(
        "Paste text reviews (One review per line):", 
        value="The delivery was incredibly slow.\nAmazing build quality, loving it!\nCustomer support didn't answer my emails."
    )
    if text_input.strip():
        lines = [line.strip() for line in text_input.split("\n") if line.strip()]
        if lines:
            raw_df = pd.DataFrame(lines, columns=["Review"])
    else:
        st.sidebar.warning("⚠️ Please enter at least one text review line.")

else:
    raw_df = pd.DataFrame(SAMPLE_REVIEWS)
    st.sidebar.success("Sample data loaded!")

# Main dashboard logic executing if data is loaded safely
if raw_df is not None and not raw_df.empty:
    try:
        # Run analytical pipeline
        processed_df = process_dataframe(raw_df)
        
        if processed_df.empty:
            st.warning("The input data doesn't contain any valid text to analyze.")
            st.stop()
            
        # Calculate high-level business metrics
        total_reviews = len(processed_df)
        pos_count = len(processed_df[processed_df["Sentiment"] == "Positive"])
        neg_count = len(processed_df[processed_df["Sentiment"] == "Negative"])
        neu_count = len(processed_df[processed_df["Sentiment"] == "Neutral"])
        
        # CSAT Score Formula: (Positive Reviews / Total Reviews) * 100
        csat_score = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
        
        # ------------------------------------------
        # SECTION 1: EXECUTIVE KPI METRICS
        # ------------------------------------------
        st.markdown("### 📈 Executive KPI Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Reviews Evaluated", f"{total_reviews:,}")
        m2.metric("Customer Satisfaction (CSAT)", f"{csat_score:.1f}%")
        m3.metric("Positive Reviews", f"{pos_count} 👍", delta=f"{(pos_count/total_reviews)*100:.1f}%" if total_reviews > 0 else "0%")
        m4.metric("Negative Reviews", f"{neg_count} 👎", delta=f"{(neg_count/total_reviews)*100:.1f}%" if total_reviews > 0 else "0%", delta_color="inverse")
        
        st.write("---")
        
        # ------------------------------------------
        # SECTION 2: GRAPHICAL CHARTS
        # ------------------------------------------
        st.markdown("### 📊 Sentiment Distribution & Categorization Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive Sentiment Chart
            fig_sent = px.pie(
                names=["Positive", "Negative", "Neutral"],
                values=[pos_count, neg_count, neu_count],
                color=["Positive", "Negative", "Neutral"],
                color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
                hole=0.4,
                title="Proportional Sentiment Breakdown"
            )
            fig_sent.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_sent, use_container_width=True)
            
        with col2:
            # Breakdown of volume by categories safely handle variations split
            category_counts = processed_df["Inferred Category"].str.split(", ").explode().value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            
            fig_cat = px.bar(
                category_counts,
                x="Count",
                y="Category",
                orientation='h',
                title="Review Volume by Inferred Product Category / Department",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cat.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cat, use_container_width=True)

        st.write("---")
        
        # ------------------------------------------
        # SECTION 3: REASONS FOR DISLIKES & ALERTS
        # ------------------------------------------
        st.markdown("### 🔍 Root Cause Analysis: Reasons for Dislikes & Complaints")
        
        # Isolate issues from negative feedback
        negative_df = processed_df[processed_df["Sentiment"] == "Negative"]
        
        # Safe structural verification checking that negative feedback has keywords lists
        has_negative_keywords = False
        if not negative_df.empty:
            all_neg_keywords = negative_df["Keywords/Reasons"].explode().dropna().tolist()
            # Filter out generic terms if more descriptive terms exist
            all_neg_keywords = [k for k in all_neg_keywords if k != "general feedback"]
            if all_neg_keywords:
                has_negative_keywords = True
                
        if has_negative_keywords:
            keyword_counts = pd.Series(all_neg_keywords).value_counts().reset_index()
            keyword_counts.columns = ["Complaint Factor / Keyword", "Frequency Count"]
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write("**Top Complaint Drivers Indicated in Negative Feedback:**")
                fig_kw = px.bar(
                    keyword_counts.head(8),
                    x="Frequency Count",
                    y="Complaint Factor / Keyword",
                    orientation='h',
                    color_discrete_sequence=["#d9534f"],
                    title="Key Negative Pain-Points"
                )
                fig_kw.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_kw, use_container_width=True)
                
            with c2:
                st.write("**🤖 Automated Operational Prescriptions:**")
                top_complaints = keyword_counts["Complaint Factor / Keyword"].head(3).tolist()
                
                for complaint in top_complaints:
                    if complaint in ["delivery", "shipping", "late", "delay", "package", "arrived"]:
                        st.error("⚠️ **Supply Chain Alert:** High volume of complaints regarding shipping delays or packages. *Recommendation:* Re-evaluate SLAs with logistics partners.")
                    elif complaint in ["broken", "damaged", "cheap", "quality", "material", "broke"]:
                        st.error("⚠️ **Quality Assurance Warning:** Customers citing structural damage or poor quality material. *Recommendation:* Audit specific factory batches or packaging protections.")
                    elif complaint in ["support", "service", "refund", "return"]:
                        st.error("⚠️ **Customer Experience Deficit:** Issues centered around customer help or refund execution latency. *Recommendation:* Implement automated support macros.")
                    elif complaint in ["app", "website", "bug", "crash", "slow"]:
                        st.error("⚠️ **Technical / UI Friction:** Customers reporting performance optimization drops. *Recommendation:* Review the crash logs on the checkout processes.")
        else:
            st.success("🎉 **Excellent Quality Standing:** No systemic negative keywords or structural complaints were isolated from the current parsed data.")

        st.write("---")

        # ------------------------------------------
        # SECTION 4: DATA TABLE AUDITING
        # ------------------------------------------
        st.markdown("### 📋 Granular Feedback Audit Logs")
        
        filter_sentiment = st.multiselect("Filter Records by Sentiment Profile:", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
        
        filtered_df = processed_df[processed_df["Sentiment"].isin(filter_sentiment)]
        
        display_df = filtered_df[["Review", "Sentiment", "Sentiment Score", "Inferred Category"]].copy()
        st.dataframe(display_df, use_container_width=True)
        
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Annotated Structured Feedback (CSV)",
            data=csv_data,
            file_name="analyzed_customer_feedback.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Execution Error during evaluation pipeline: {e}")
else:
    st.info("👋 Welcome! Please utilize the left sidebar options to upload a file, paste content, or load default parameters.")