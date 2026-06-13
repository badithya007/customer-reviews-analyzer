import importlib
import streamlit as st
import pandas as pd
import plotly.express as px
import re

try:
    vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
    SentimentIntensityAnalyzer = vader_module.SentimentIntensityAnalyzer
except (ImportError, ModuleNotFoundError):
    class SentimentIntensityAnalyzer:
        def __init__(self):
            self.positive_words = {
                "incredible", "premium", "great", "fast", "love", "worth", "good", "excellent", "amazing", "happy", "satisfied", "super", "clean"
            }
            self.negative_words = {
                "late", "delay", "broken", "damaged", "cheap", "waste", "refused", "terrible", "crashing", "frustrating", "overpriced", "broke", "faulty", "problem", "error"
            }

        def polarity_scores(self, text: str):
            tokens = re.findall(r"\w+", text.lower())
            pos = sum(1 for token in tokens if token in self.positive_words)
            neg = sum(1 for token in tokens if token in self.negative_words)
            total = max(len(tokens), 1)
            compound = (pos - neg) / total
            return {
                "compound": compound,
                "pos": pos / total,
                "neg": neg / total,
                "neu": max(0.0, 1.0 - (pos + neg) / total)
            }

# Initialize the self-contained sentiment engine
analyzer = SentimentIntensityAnalyzer()

# ==========================================
# APPLICATION SETUP & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="Enterprise Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

# Strict dictionary mapping keywords to explicit categories and explicit reasons for complaints
ANALYSIS_MAP = {
    "Shipping & Delivery": {
        "keywords": ["delivery", "shipping", "late", "delay", "arrived", "package", "post", "carrier", "box"],
        "complaint_reason": "Logistics & Shipping Delays",
        "action": "Optimize supply chain routing and audit carrier transit SLAs."
    },
    "Product Quality": {
        "keywords": ["broken", "damaged", "cheap", "quality", "material", "tore", "broke", "plastic", "faulty", "defect"],
        "complaint_reason": "Material & Quality Defects",
        "action": "Initiate a factory QA audit on recent component batches."
    },
    "Customer Support": {
        "keywords": ["support", "service", "refund", "return", "agent", "manager", "chat", "email", "called", "help"],
        "complaint_reason": "Support Responsiveness & Politeness",
        "action": "Scale support team staffing or streamline refund authorization rules."
    },
    "Pricing & Value": {
        "keywords": ["expensive", "price", "worth", "cost", "waste", "money", "overpriced", "charge"],
        "complaint_reason": "Price-to-Value Mismatch",
        "action": "Review competitive pricing models or offer targeted loyalty discounts."
    },
    "Digital Platform / App": {
        "keywords": ["app", "website", "interface", "bug", "error", "crash", "slow", "login", "button", "ui"],
        "complaint_reason": "UI Performance & Technical Glitches",
        "action": "Assign an engineering sprint to fix UI latency and crash logs."
    }
}

# In-memory sample data matching your required format exactly
MOCK_DATABASE = [
    {"Review": "The delivery was late by 4 days and the package arrived broken."},
    {"Review": "Incredible quality! Material feels premium and definitely worth the price."},
    {"Review": "Customer service agent refused to issue a refund. Complete waste of time."},
    {"Review": "The app keeps crashing on the checkout page. Highly frustrating website interface."},
    {"Review": "Terrible cheap plastic quality, it broke on the first day."},
    {"Review": "Super fast shipping and great customer support team!"},
    {"Review": "Too expensive for what you actually get. Overpriced product."},
    {"Review": "The item was heavily damaged during shipping delivery."},
    {"Review": "Clean interface, fast app processing, and cheap price."},
    {"Review": "I love the product quality but support emails took a week to answer."}
]

# ==========================================
# PROCESSING PIPELINE
# ==========================================
def process_feedback_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes reviews, applies sentiment tracking, tags categories and lists explicit reasons."""
    # Prevent mutations on original frame
    working_df = df.dropna(subset=["Review"]).copy()
    working_df["Review"] = working_df["Review"].astype(str)
    
    sentiment_labels = []
    sentiment_scores = []
    inferred_categories = []
    dislike_reasons = []
    suggested_actions = []
    
    for review_text in working_df["Review"]:
        lower_text = review_text.lower()
        
        # 1. Core Sentiment Evaluation
        scores = analyzer.polarity_scores(review_text)
        compound = scores['compound']
        sentiment_scores.append(compound)
        
        if compound >= 0.05:
            sentiment_labels.append("Positive")
        elif compound <= -0.05:
            sentiment_labels.append("Negative")
        else:
            sentiment_labels.append("Neutral")
            
        # 2. Strict Keyword Structural Mapping
        matched_cats = []
        matched_reasons = []
        matched_actions = []
        
        for category, rules in ANALYSIS_MAP.items():
            if any(keyword in lower_text for keyword in rules["keywords"]):
                matched_cats.append(category)
                matched_reasons.append(rules["complaint_reason"])
                matched_actions.append(rules["action"])
                
        # Default fallbacks if no precise keyword patterns match
        if not matched_cats:
            inferred_categories.append("General Feedback")
            dislike_reasons.append("Unclassified/General Mention")
            suggested_actions.append("Monitor sentiment trends for specific indicators.")
        else:
            # Drop duplicates while maintaining order
            inferred_categories.append(", ".join(list(dict.fromkeys(matched_cats))))
            dislike_reasons.append(", ".join(list(dict.fromkeys(matched_reasons))))
            suggested_actions.append(" | ".join(list(dict.fromkeys(matched_actions))))
            
    working_df["Sentiment"] = sentiment_labels
    working_df["Score"] = sentiment_scores
    working_df["Product Category"] = inferred_categories
    working_df["Reason for Dislike"] = dislike_reasons
    working_df["Actionable Recommendation"] = suggested_actions
    
    return working_df

# ==========================================
# MODERN GRAPHICAL USER INTERFACE
# ==========================================
st.title("📊 Customer Feedback Analytics Dashboard")
st.markdown("Instantly break down raw customer reviews into clear categories, sentiments, and explicit reasons for dislikes.")

# Sidebar Controls
st.sidebar.header("📥 Feedback Intake Engine")
input_strategy = st.sidebar.radio(
    "Select Input Source:",
    ["Use Clean Sample Dataset", "Upload Spreadsheet (CSV/XLSX)", "Paste Raw Texts"]
)

raw_data_frame = None

if input_strategy == "Use Clean Sample Dataset":
    raw_data_frame = pd.DataFrame(MOCK_DATABASE)
    st.sidebar.success("Loaded default sample reviews.")

elif input_strategy == "Upload Spreadsheet (CSV/XLSX)":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])