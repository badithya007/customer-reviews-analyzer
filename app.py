import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import importlib
try:
    # try common package names for VADER
    try:
        vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
    except ImportError:
        vader_module = importlib.import_module("vader_sentiment.vader_sentiment")
    SentimentIntensityAnalyzer = vader_module.SentimentIntensityAnalyzer
except Exception:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception:
        SentimentIntensityAnalyzer = None
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
try:
    sklearn_feature_module = importlib.import_module("sklearn.feature_extraction.text")
    sklearn_cluster_module = importlib.import_module("sklearn.cluster")
    TfidfVectorizer = sklearn_feature_module.TfidfVectorizer
    KMeans = sklearn_cluster_module.KMeans
except ImportError:
    TfidfVectorizer = None
    KMeans = None
import re
import io
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0F0F1A; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 1px solid #7C3AED44;
        border-radius: 16px;
        padding: 18px 22px;
    }
    [data-testid="stMetricLabel"] { color: #A78BFA !important; font-size: 13px !important; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #E2E8F0 !important; font-size: 28px !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }
    .section-header {
        background: linear-gradient(90deg, #7C3AED22, transparent);
        border-left: 4px solid #7C3AED;
        padding: 10px 20px;
        border-radius: 0 12px 12px 0;
        margin: 24px 0 16px 0;
        font-size: 18px;
        font-weight: 700;
        color: #E2E8F0;
        letter-spacing: 0.02em;
    }
    .verdict-box {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 2px solid #7C3AED;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 12px 0;
    }
    .verdict-title { font-size: 22px; font-weight: 800; color: #A78BFA; margin-bottom: 10px; letter-spacing: 0.03em; }
    .verdict-text { font-size: 15px; line-height: 1.7; color: #CBD5E1; }
    .suggestion-box {
        background: linear-gradient(135deg, #0D2137, #0A1628);
        border: 1px solid #3B82F644;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .chip-pos { display:inline-block; background:#064E3B; color:#6EE7B7; border:1px solid #10B98144; border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; margin:3px; }
    .chip-neg { display:inline-block; background:#4C0519; color:#FCA5A5; border:1px solid #EF444444; border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; margin:3px; }
    .chip-neu { display:inline-block; background:#1E3A5F; color:#93C5FD; border:1px solid #3B82F644; border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; margin:3px; }
    .review-card {
        background: #1A1A2E; border-radius: 12px; padding: 14px 18px; margin: 6px 0;
        border-left: 4px solid #7C3AED; font-size: 14px; color: #CBD5E1; line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: #1A1A2E; border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px; color: #94A3B8; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #7C3AED !important; color: white !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0D0D1A 0%, #1A1A2E 100%); border-right: 1px solid #7C3AED33; }
    hr { border-color: #7C3AED22 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #5B21B6); color: white; border: none;
        border-radius: 10px; font-weight: 700; letter-spacing: 0.05em; padding: 8px 24px; transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #8B5CF6, #6D28D9); transform: translateY(-1px); box-shadow: 0 4px 15px #7C3AED55; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F0F1A; }
    ::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 3px; }
    [data-testid="stFileUploadDropzone"] { background: #1A1A2E; border: 2px dashed #7C3AED55; border-radius: 12px; }
    .stTextArea textarea { background: #1A1A2E; border: 1px solid #7C3AED44; border-radius: 12px; color: #E2E8F0; font-size: 14px; }
    .stSelectbox > div > div { background: #1A1A2E; border: 1px solid #7C3AED44; color: #E2E8F0; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
    .streamlit-expanderHeader { background: #1A1A2E; border-radius: 10px; font-weight: 600; color: #A78BFA; }
    .stNumberInput > div > div > input { background: #1A1A2E; color: #E2E8F0; border: 1px solid #7C3AED44; }
</style>
""", unsafe_allow_html=True)


PRODUCT_CATEGORIES = [
    "Electronics & Tech", "Food & Beverage", "Fashion & Apparel",
    "Healthcare & Wellness", "Software & Apps", "Retail & E-Commerce",
    "Hospitality & Travel", "Financial Services", "Education & E-Learning",
    "Automotive", "Beauty & Personal Care", "Home & Living",
    "Entertainment & Media", "Telecommunications", "Other / Custom",
]

SAMPLE_REVIEWS = """The product quality is outstanding! Delivery was super fast and packaging was perfect.
Customer service resolved my issue within minutes. Highly recommend this to everyone!
Terrible experience. The item arrived broken and customer support was unhelpful.
Average product, nothing special. Expected better quality for the price.
Love this brand! Have been using it for years and they never disappoint.
The app crashes frequently and the battery drain is horrible. Waste of money.
Excellent value for money. Works exactly as described. Will buy again.
Very disappointed. The product stopped working after just 2 weeks.
Good product overall but the instructions were unclear. Took a while to set up.
Amazing! Exceeded all my expectations. The build quality is top-notch.
Shipping took forever and the item was damaged. Terrible packaging.
Great customer service! They replaced my faulty unit with no questions asked.
The product is decent but overpriced compared to competitors.
Absolutely love it! Best purchase I've made this year. Game changer!
Not worth the money. Feels cheap and poorly made.
"""


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def get_sentiment_label(compound):
    if compound >= 0.05:
        return "Positive", "#10B981"
    elif compound <= -0.05:
        return "Negative", "#EF4444"
    else:
        return "Neutral", "#F59E0B"


def extract_keywords(texts, top_n=15, exclude_words=None):
    stop = set(stopwords.words("english"))
    extra = {
        "product", "item", "one", "use", "get", "got", "would", "could",
        "really", "also", "even", "much", "still", "just", "like", "good",
        "bad", "great", "said", "told", "made", "used", "bought", "ordered",
        "received", "make", "time", "very", "way", "well", "back", "came",
    }
    if exclude_words:
        extra.update(exclude_words)
    stop.update(extra)
    all_words = []
    for text in texts:
        tokens = word_tokenize(clean_text(text))
        words = [w for w in tokens if w not in stop and len(w) > 3]
        all_words.extend(words)
    return Counter(all_words).most_common(top_n)


def compute_satisfaction_score(df):
    avg_compound = df["compound"].mean()
    pos_ratio = (df["sentiment"] == "Positive").mean()
    neg_ratio = (df["sentiment"] == "Negative").mean()
    score = 5 + (avg_compound * 3) + (pos_ratio * 2) - (neg_ratio * 1.5)
    return round(max(0, min(10, score)), 1)


def get_verdict(score, pos_pct, neg_pct, df):
    if score >= 8.0:
        return "🏆", "EXCEPTIONAL", "#10B981", f"Customers are overwhelmingly satisfied. Score {score}/10 with {pos_pct:.0f}% positive reviews — elite performance. Focus on maintaining consistency and scaling what works."
    elif score >= 6.5:
        return "✅", "GOOD", "#6EE7B7", f"Solid performance with score {score}/10 and {pos_pct:.0f}% positive. Addressing the {neg_pct:.0f}% negative feedback could push this into exceptional territory."
    elif score >= 5.0:
        return "⚠️", "AVERAGE", "#F59E0B", f"Mixed reception at {score}/10. The {pos_pct:.0f}% positive / {neg_pct:.0f}% negative split indicates specific pain points that need urgent attention to prevent customer churn."
    elif score >= 3.0:
        return "🔴", "POOR", "#EF4444", f"Below average at {score}/10. With {neg_pct:.0f}% negative reviews, serious product or service issues need immediate intervention. Customer trust is at risk."
    else:
        return "🚨", "CRITICAL", "#DC2626", f"Critical situation — score {score}/10 with {neg_pct:.0f}% negative feedback. Immediate comprehensive action required. Brand reputation is under serious threat."


def generate_suggestions(df, keywords_neg, keywords_pos, category):
    suggestions = []
    neg_words = [w for w, _ in keywords_neg]
    pos_words = [w for w, _ in keywords_pos]
    neg_pct = (df["sentiment"] == "Negative").mean() * 100
    if neg_pct > 40:
        suggestions.append(("🔧 Quality Control", "Critical", "Significant quality issues detected. Implement stricter QC checkpoints before dispatch. Consider third-party quality audits."))
    if any(w in neg_words for w in ["shipping", "delivery", "arrived", "late", "slow", "package", "packaging"]):
        suggestions.append(("📦 Shipping & Packaging", "High", "Delivery speed and packaging are pain points. Partner with faster logistics providers and invest in protective packaging materials."))
    if any(w in neg_words for w in ["support", "service", "response", "staff", "team", "help", "unhelpful", "rude"]):
        suggestions.append(("💬 Customer Support", "High", "Customer service issues flagged. Implement 24/7 support channels, train staff on empathy, and set SLA targets under 2 hours."))
    if any(w in neg_words for w in ["price", "expensive", "overpriced", "cost", "cheap", "worth", "money"]):
        suggestions.append(("💰 Pricing Strategy", "Medium", "Customers perceive value/price mismatch. Review competitive pricing. Consider tiered plans or bundles to justify price points."))
    if any(w in neg_words for w in ["battery", "performance", "slow", "crash", "broken", "working", "defect"]):
        suggestions.append(("⚡ Product Performance", "Critical", "Performance failures detected. Accelerate testing cycles, fix reported bugs, and issue a reliability update or firmware patch."))
    if any(w in neg_words for w in ["instruction", "manual", "setup", "confusing", "unclear", "understand", "difficult"]):
        suggestions.append(("📖 UX & Documentation", "Medium", "Setup/usability issues reported. Create video tutorials, improve in-app onboarding, and revamp user manuals with visuals."))
    if any(w in pos_words for w in ["recommend", "love", "amazing", "excellent", "best", "great", "fantastic"]):
        suggestions.append(("⭐ Leverage Advocates", "Opportunity", "High satisfaction advocates exist. Launch a referral program and collect video testimonials. Use positive reviews in marketing."))
    if any(w in pos_words for w in ["fast", "quick", "speedy", "prompt"]):
        suggestions.append(("🚀 Amplify Speed", "Opportunity", "Speed is a key differentiator customers love. Highlight fast delivery/response in marketing and maintain this competitive edge."))
    category_suggestions = {
        "Electronics & Tech": ("🔌 Tech Support Portal", "Medium", "Create a self-service tech support portal with FAQs, troubleshooting guides, and community forums."),
        "Food & Beverage": ("🍽️ Freshness Guarantee", "High", "Introduce a freshness guarantee program and display expiry details clearly on packaging."),
        "Fashion & Apparel": ("📐 Size Guide", "Medium", "Invest in an interactive size guide and virtual try-on feature to reduce returns."),
        "Healthcare & Wellness": ("🏥 Expert Consultation", "High", "Partner with certified professionals to offer consultation services and add credibility to your product claims."),
        "Software & Apps": ("🐛 Bug Bounty Program", "Medium", "Launch a bug bounty or beta testing program to identify issues before they reach end users."),
        "Hospitality & Travel": ("🌟 Loyalty Program", "Opportunity", "Implement a tiered loyalty rewards program to retain frequent customers and incentivize repeat bookings."),
    }
    if category in category_suggestions:
        suggestions.append(category_suggestions[category])
    if not suggestions:
        suggestions.append(("✅ Maintain Standards", "Low", "Performance is good. Focus on continuous monitoring, collecting regular feedback, and iterating on small improvements."))
    return suggestions


def make_gauge(score):
    color = "#10B981" if score >= 7 else "#F59E0B" if score >= 5 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 5, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
        number={"suffix": "/10", "font": {"size": 36, "color": "#E2E8F0"}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1A1A2E", "borderwidth": 0,
            "steps": [
                {"range": [0, 3], "color": "#1F0A0A"}, {"range": [3, 6], "color": "#1A1A0A"},
                {"range": [6, 8], "color": "#0A1A12"}, {"range": [8, 10], "color": "#0A1A0A"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score},
        },
        title={"text": "Customer Satisfaction Score", "font": {"size": 14, "color": "#A78BFA"}},
    ))
    fig.update_layout(paper_bgcolor="#0F0F1A", font={"color": "#E2E8F0"}, height=280, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def make_donut(pos, neg, neu):
    fig = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"], values=[pos, neg, neu], hole=0.65,
        marker=dict(colors=["#10B981", "#EF4444", "#F59E0B"], line=dict(color="#0F0F1A", width=3)),
        textinfo="percent+label", textfont=dict(size=13, color="#E2E8F0"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A", font=dict(color="#E2E8F0"),
        legend=dict(font=dict(color="#E2E8F0"), bgcolor="rgba(0,0,0,0)"),
        showlegend=True, height=320, margin=dict(l=10, r=10, t=30, b=10),
        annotations=[dict(text=f"<b>{pos+neg+neu}</b><br>Reviews", x=0.5, y=0.5, font_size=16, font_color="#A78BFA", showarrow=False)],
    )
    return fig


def make_bar(keywords, title, color="#7C3AED"):
    if not keywords:
        return go.Figure()
    words, counts = zip(*keywords[:12])
    fig = go.Figure(go.Bar(
        x=list(counts), y=list(words), orientation="h",
        marker=dict(color=list(counts), colorscale=[[0, color + "55"], [1, color]], line=dict(width=0)),
        text=list(counts), textposition="outside", textfont=dict(color="#CBD5E1", size=12),
        hovertemplate="<b>%{y}</b>: %{x} mentions<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#A78BFA", size=15)),
        paper_bgcolor="#0F0F1A", plot_bgcolor="#1A1A2E",
        xaxis=dict(showgrid=False, color="#475569", title="Frequency"),
        yaxis=dict(showgrid=False, color="#E2E8F0", autorange="reversed"),
        height=380, margin=dict(l=20, r=60, t=50, b=20), font=dict(color="#E2E8F0"),
    )
    return fig


def make_wordcloud(texts, title, colormap="Purples"):
    if not texts:
        return None
    combined = " ".join([clean_text(t) for t in texts])
    stop = set(stopwords.words("english"))
    try:
        wc = WordCloud(
            width=700, height=350, background_color="#0F0F1A", colormap=colormap,
            stopwords=stop, min_font_size=10, max_words=80, prefer_horizontal=0.85, collocations=False,
        ).generate(combined)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0F0F1A")
        ax.set_facecolor("#0F0F1A")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, color="#A78BFA", fontsize=14, fontweight="bold", pad=10)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def make_sentiment_timeline(df):
    if "date" not in df.columns:
        return None
    try:
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2 = df2.dropna(subset=["date"])
        if df2.empty:
            return None
        df2 = df2.sort_values("date")
        df2["rolling_avg"] = df2["compound"].rolling(5, min_periods=1).mean()
        fig = go.Figure()
        for sent, color in [("Positive", "#10B981"), ("Negative", "#EF4444"), ("Neutral", "#F59E0B")]:
            sub = df2[df2["sentiment"] == sent]
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["compound"], mode="markers",
                marker=dict(color=color, size=7, opacity=0.7), name=sent))
        fig.add_trace(go.Scatter(x=df2["date"], y=df2["rolling_avg"], mode="lines",
            line=dict(color="#A78BFA", width=2.5, dash="dash"), name="Trend"))
        fig.update_layout(
            title=dict(text="Sentiment Trend Over Time", font=dict(color="#A78BFA", size=15)),
            paper_bgcolor="#0F0F1A", plot_bgcolor="#1A1A2E",
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="#1E293B", color="#E2E8F0", range=[-1.1, 1.1]),
            legend=dict(font=dict(color="#E2E8F0"), bgcolor="rgba(0,0,0,0)"),
            height=320, margin=dict(l=20, r=20, t=50, b=20), font=dict(color="#E2E8F0"),
        )
        return fig
    except Exception:
        return None


def cluster_reviews(reviews, n_clusters=4):
    if TfidfVectorizer is None or KMeans is None:
        return None, None
    if len(reviews) < n_clusters * 2:
        n_clusters = max(2, len(reviews) // 2)
    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
        X = vectorizer.fit_transform(reviews)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        feature_names = vectorizer.get_feature_names_out()
        cluster_topics = {}
        for i in range(n_clusters):
            center = km.cluster_centers_[i]
            top_idx = center.argsort()[-5:][::-1]
            cluster_topics[i] = [feature_names[j] for j in top_idx]
        return labels, cluster_topics
    except Exception:
        return None, None


def parse_reviews(text):
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) == 1:
        sents = re.split(r"[.!?]+", lines[0])
        lines = [s.strip() for s in sents if len(s.strip()) > 15]
    return lines


def analyze_reviews(reviews, product_name, category):
    analyzer = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    records = []
    for i, rev in enumerate(reviews):
        scores = analyzer.polarity_scores(rev)
        label, color = get_sentiment_label(scores["compound"])
        records.append({
            "id": i + 1, "review": rev, "sentiment": label, "color": color,
            "compound": scores["compound"], "positive_score": scores["pos"],
            "negative_score": scores["neg"], "neutral_score": scores["neu"],
        })
    return pd.DataFrame(records)


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:36px;'>📊</div>
        <div style='font-size:18px; font-weight:800; color:#A78BFA;'>Feedback Analyzer</div>
        <div style='font-size:11px; color:#64748B; margin-top:4px;'>AI-Powered Review Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ Configuration**")
    product_name = st.text_input("Product / Brand Name", placeholder="e.g. iPhone 15, Zomato, Nike Air")
    category = st.selectbox("Product Category", PRODUCT_CATEGORIES)
    if category == "Other / Custom":
        category = st.text_input("Enter your category", placeholder="e.g. Pet Care")
    st.markdown("---")
    st.markdown("**🔢 Analysis Settings**")
    top_n_keywords = st.slider("Keywords to extract", 5, 25, 15)
    n_topics = st.slider("Topic clusters", 2, 8, 4)
    show_raw = st.checkbox("Show individual reviews", value=False)
    show_clusters = st.checkbox("Show topic clusters", value=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#475569; text-align:center;'>
        Built for business analysts & product teams<br>
        Powered by VADER · TF-IDF · KMeans
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 30px 0 10px 0;'>
    <div style='font-size:32px; font-weight:900; background:linear-gradient(90deg,#A78BFA,#60A5FA);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        Customer Feedback Analyzer
    </div>
    <div style='font-size:14px; color:#64748B; margin-top:6px;'>
        Paste reviews or upload a file · Uncover sentiment, complaints, praises & actionable insights
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

tab_input, tab_results = st.tabs(["📥  Input Reviews", "📈  Analysis Results"])

# ─── TAB 1: INPUT ──────────────────────────────────────────────────────────────
with tab_input:
    col_paste, col_upload = st.columns([1, 1], gap="large")

    with col_paste:
        st.markdown('<div class="section-header">✍️ Paste Reviews</div>', unsafe_allow_html=True)
        use_sample = st.checkbox("Load sample reviews (demo)", value=False)
        default_val = SAMPLE_REVIEWS if use_sample else ""
        review_text = st.text_area(
            "One review per line (or separated by punctuation)",
            value=default_val, height=300,
            placeholder="The product quality is great!\nDelivery was too slow...",
        )

    with col_upload:
        st.markdown('<div class="section-header">📁 Upload File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV, Excel or TXT (one review per row)", type=["csv", "xlsx", "xls", "txt"])
        col_name = None
        if uploaded:
            try:
                fname = uploaded.name.lower()
                if fname.endswith(".txt"):
                    raw = uploaded.read().decode("utf-8", errors="ignore")
                    review_text = raw
                    st.success(f"✅ Loaded {len(parse_reviews(raw))} reviews from TXT")
                else:
                    df_up = pd.read_csv(uploaded) if fname.endswith(".csv") else pd.read_excel(uploaded)
                    st.write("**Preview:**")
                    st.dataframe(df_up.head(3), use_container_width=True)
                    text_cols = [c for c in df_up.columns if df_up[c].dtype == object]
                    col_name = st.selectbox("Select the review column", text_cols) if text_cols else None
                    if col_name:
                        review_text = "\n".join(df_up[col_name].dropna().astype(str).tolist())
                        date_cols = [c for c in df_up.columns if "date" in c.lower() or "time" in c.lower()]
                        if date_cols:
                            st.info(f"📅 Date column detected: **{date_cols[0]}** — timeline chart enabled.")
                        st.success(f"✅ Loaded {len(df_up)} reviews from file")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    st.markdown("---")
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("🔍 Analyze Feedback", use_container_width=True)
    with col_info:
        if not product_name:
            st.info("💡 Set a product name in the sidebar for personalized insights.")

# ─── TAB 2: RESULTS ────────────────────────────────────────────────────────────
with tab_results:
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if analyze_btn and review_text and review_text.strip():
        with st.spinner("🔄 Analyzing reviews..."):
            reviews = parse_reviews(review_text)
            if len(reviews) < 2:
                st.error("Please provide at least 2 reviews.")
                st.stop()
            df = analyze_reviews(reviews, product_name or "Product", category)
            if uploaded and col_name:
                try:
                    fname = uploaded.name.lower()
                    df_source = pd.read_csv(uploaded) if fname.endswith(".csv") else pd.read_excel(uploaded)
                    date_cols = [c for c in df_source.columns if "date" in c.lower() or "time" in c.lower()]
                    if date_cols:
                        dates = df_source[date_cols[0]].dropna().astype(str).tolist()
                        if len(dates) == len(df):
                            df["date"] = dates
                except Exception:
                    pass
            st.session_state.df = df
            st.session_state.reviews = reviews
            st.session_state.category = category
            st.session_state.product_name = product_name or "Product"
            st.session_state.analysis_done = True
        st.success("✅ Analysis complete! Switch to the **Analysis Results** tab.")
    elif analyze_btn:
        st.warning("Please paste or upload some reviews first.")

    if st.session_state.get("analysis_done") and "df" in st.session_state:
        df = st.session_state.df
        reviews = st.session_state.reviews
        category = st.session_state.category
        pname = st.session_state.product_name

        pos_df = df[df["sentiment"] == "Positive"]
        neg_df = df[df["sentiment"] == "Negative"]
        neu_df = df[df["sentiment"] == "Neutral"]
        n_pos, n_neg, n_neu, n_total = len(pos_df), len(neg_df), len(neu_df), len(df)
        pos_pct = n_pos / n_total * 100
        neg_pct = n_neg / n_total * 100
        neu_pct = n_neu / n_total * 100

        score = compute_satisfaction_score(df)
        emoji, level, verdict_color, verdict_text = get_verdict(score, pos_pct, neg_pct, df)
        keywords_neg = extract_keywords(neg_df["review"].tolist(), top_n=top_n_keywords) if not neg_df.empty else []
        keywords_pos = extract_keywords(pos_df["review"].tolist(), top_n=top_n_keywords) if not pos_df.empty else []
        suggestions = generate_suggestions(df, keywords_neg, keywords_pos, category)

        # ── Overview metrics ──────────────────────────────────────────────────
        st.markdown(f'<div class="section-header">📌 Overview — {pname} · {category}</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: st.metric("Total Reviews", n_total)
        with m2: st.metric("Positive", f"{n_pos} ({pos_pct:.0f}%)", delta=f"+{pos_pct:.0f}%")
        with m3: st.metric("Negative", f"{n_neg} ({neg_pct:.0f}%)", delta=f"-{neg_pct:.0f}%", delta_color="inverse")
        with m4: st.metric("Neutral", f"{n_neu} ({neu_pct:.0f}%)")
        with m5: st.metric("Avg Sentiment", f"{df['compound'].mean():.3f}")
        with m6: st.metric("Satisfaction", f"{score}/10")

        st.markdown("---")
        col_gauge, col_donut = st.columns(2)
        with col_gauge:
            st.plotly_chart(make_gauge(score), use_container_width=True)
        with col_donut:
            st.markdown('<div class="section-header">📊 Sentiment Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(make_donut(n_pos, n_neg, n_neu), use_container_width=True)

        # ── Final Verdict ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🏁 Final Verdict</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="verdict-box" style="border-color:{verdict_color}44;">
            <div class="verdict-title" style="color:{verdict_color};">{emoji} {level}</div>
            <div class="verdict-text">{verdict_text}</div>
            <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
                <span class="chip-pos">✅ {n_pos} Positive</span>
                <span class="chip-neg">❌ {n_neg} Negative</span>
                <span class="chip-neu">➖ {n_neu} Neutral</span>
                <span style="display:inline-block;background:#2D1B69;color:#C4B5FD;border:1px solid #7C3AED44;
                    border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;margin:3px;">Score: {score}/10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Complaints & Praises ──────────────────────────────────────────────
        st.markdown("---")
        col_neg_bar, col_pos_bar = st.columns(2)
        with col_neg_bar:
            st.markdown('<div class="section-header">⚠️ Top Complaints</div>', unsafe_allow_html=True)
            if keywords_neg:
                st.plotly_chart(make_bar(keywords_neg, "Most Frequent Complaint Keywords", "#EF4444"), use_container_width=True)
                st.markdown(" ".join([f'<span class="chip-neg">{w} ({c})</span>' for w, c in keywords_neg[:8]]), unsafe_allow_html=True)
            else:
                st.success("🎉 No significant negative reviews detected!")
        with col_pos_bar:
            st.markdown('<div class="section-header">⭐ Top Praises</div>', unsafe_allow_html=True)
            if keywords_pos:
                st.plotly_chart(make_bar(keywords_pos, "Most Frequent Praise Keywords", "#10B981"), use_container_width=True)
                st.markdown(" ".join([f'<span class="chip-pos">{w} ({c})</span>' for w, c in keywords_pos[:8]]), unsafe_allow_html=True)
            else:
                st.info("No notable positive highlights yet.")

        # ── Word Clouds ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">☁️ Word Clouds</div>', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            fig_wc = make_wordcloud(pos_df["review"].tolist(), "Positive Review Cloud", "YlGn")
            if fig_wc:
                st.pyplot(fig_wc, use_container_width=True); plt.close()
        with wc2:
            fig_wc2 = make_wordcloud(neg_df["review"].tolist(), "Negative Review Cloud", "Reds")
            if fig_wc2:
                st.pyplot(fig_wc2, use_container_width=True); plt.close()

        # ── Timeline ──────────────────────────────────────────────────────────
        timeline_fig = make_sentiment_timeline(df)
        if timeline_fig:
            st.markdown("---")
            st.markdown('<div class="section-header">📅 Sentiment Timeline</div>', unsafe_allow_html=True)
            st.plotly_chart(timeline_fig, use_container_width=True)

        # ── Topic Clusters ────────────────────────────────────────────────────
        if show_clusters and len(reviews) >= 4:
            st.markdown("---")
            st.markdown('<div class="section-header">🗂️ Review Topic Clusters</div>', unsafe_allow_html=True)
            labels, cluster_topics = cluster_reviews([clean_text(r) for r in reviews], n_topics)
            if labels is not None:
                cluster_colors = ["#7C3AED","#3B82F6","#10B981","#F59E0B","#EF4444","#EC4899","#06B6D4","#8B5CF6"]
                df["cluster"] = labels
                for cid, words in cluster_topics.items():
                    count = (df["cluster"] == cid).sum()
                    color = cluster_colors[cid % len(cluster_colors)]
                    with st.expander(f"📁 Topic {cid+1}: {' · '.join(words[:3]).title()} — {count} reviews", expanded=False):
                        chip_html = " ".join([f'<span style="display:inline-block;background:{color}22;color:{color};border:1px solid {color}44;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;margin:2px;">{w}</span>' for w in words])
                        st.markdown(f"**Key terms:** {chip_html}", unsafe_allow_html=True)
                        st.markdown(f"Sentiment: **{df[df['cluster']==cid]['sentiment'].value_counts().idxmax()}** dominant")
                        for _, row in df[df["cluster"] == cid].head(4).iterrows():
                            sent_e = "✅" if row["sentiment"] == "Positive" else "❌" if row["sentiment"] == "Negative" else "➖"
                            st.markdown(f'<div class="review-card">{sent_e} {row["review"]}</div>', unsafe_allow_html=True)

        # ── Suggestions ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">💡 Actionable Suggestions</div>', unsafe_allow_html=True)
        priority_colors = {"Critical":"#EF4444","High":"#F97316","Medium":"#F59E0B","Low":"#10B981","Opportunity":"#3B82F6"}
        for title, priority, desc in suggestions:
            pcolor = priority_colors.get(priority, "#7C3AED")
            st.markdown(f"""
            <div class="suggestion-box">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                    <span style="font-size:16px;font-weight:700;color:#E2E8F0;">{title}</span>
                    <span style="background:{pcolor}22;color:{pcolor};border:1px solid {pcolor}44;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700;">{priority}</span>
                </div>
                <div style="font-size:13px;color:#94A3B8;line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Individual Reviews ────────────────────────────────────────────────
        if show_raw:
            st.markdown("---")
            st.markdown('<div class="section-header">📋 Individual Reviews</div>', unsafe_allow_html=True)
            filter_sent = st.multiselect("Filter by sentiment", ["Positive","Negative","Neutral"], default=["Positive","Negative","Neutral"])
            for _, row in df[df["sentiment"].isin(filter_sent)].iterrows():
                sent_e = "✅" if row["sentiment"]=="Positive" else "❌" if row["sentiment"]=="Negative" else "➖"
                st.markdown(f"""
                <div class="review-card" style="border-left-color:{row['color']};">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span>{sent_e} <b>{row['sentiment']}</b></span>
                        <span style="color:#64748B;font-size:12px;">#{row['id']} · Score: {row['compound']:.3f}</span>
                    </div>
                    {row['review']}
                </div>
                """, unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📤 Export Results</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            csv_data = df.drop(columns=["color","cluster"], errors="ignore").to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv_data, f"feedback_{pname.replace(' ','_')}.csv", "text/csv", use_container_width=True)
        with ec2:
            summary = {
                "product": pname, "category": category, "total_reviews": n_total,
                "positive": n_pos, "negative": n_neg, "neutral": n_neu,
                "satisfaction_score": score, "verdict": level,
                "top_complaints": [w for w,_ in keywords_neg[:5]],
                "top_praises": [w for w,_ in keywords_pos[:5]],
                "suggestions": [{"title": t, "priority": p} for t,p,_ in suggestions],
                "generated_at": datetime.now().isoformat(),
            }
            st.download_button("⬇️ Download JSON", json.dumps(summary, indent=2).encode("utf-8"),
                f"report_{pname.replace(' ','_')}.json", "application/json", use_container_width=True)
        with ec3:
            txt = f"""CUSTOMER FEEDBACK ANALYSIS REPORT
=====================================
Product : {pname}
Category: {category}
Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}

SUMMARY
-------
Total   : {n_total}
Positive: {n_pos} ({pos_pct:.1f}%)
Negative: {n_neg} ({neg_pct:.1f}%)
Neutral : {n_neu} ({neu_pct:.1f}%)
Score   : {score}/10

VERDICT: {emoji} {level}
{verdict_text}

TOP COMPLAINTS
--------------
{chr(10).join([f'  • {w} ({c} mentions)' for w,c in keywords_neg[:8]])}

TOP PRAISES
-----------
{chr(10).join([f'  • {w} ({c} mentions)' for w,c in keywords_pos[:8]])}

SUGGESTIONS
-----------
{chr(10).join([f'  [{p}] {t}' for t,p,_ in suggestions])}
"""
            st.download_button("⬇️ Download TXT Report", txt.encode("utf-8"),
                f"report_{pname.replace(' ','_')}.txt", "text/plain", use_container_width=True)

    elif not st.session_state.get("analysis_done"):
        st.markdown("""
        <div style='text-align:center; padding:80px 40px; color:#475569;'>
            <div style='font-size:56px; margin-bottom:16px;'>📊</div>
            <div style='font-size:20px; font-weight:700; color:#64748B; margin-bottom:8px;'>No analysis yet</div>
            <div style='font-size:14px;'>Go to <b>Input Reviews</b>, paste or upload reviews, then click <b>Analyze Feedback</b>.</div>
        </div>
        """, unsafe_allow_html=True)