import streamlit as st  # type: ignore
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None
    # If pandas isn't available, show an error and disable file-upload option gracefully.
    st.warning("Optional dependency 'pandas' is not available. Uploading CSV/Excel files will be disabled.")
try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None
    st.warning("Optional dependency 'plotly' is not available. Interactive charts will be disabled.")
try:
    # Import dynamically to avoid static analysis/import-time errors when
    # the optional dependency isn't installed in the environment.
    import importlib
    tb = importlib.import_module("textblob")
    TextBlob = getattr(tb, "TextBlob")
    sentiment_enabled = True
except Exception:
    TextBlob = None
    sentiment_enabled = False
    st.warning("Optional dependency 'textblob' is not available. Sentiment analysis will be disabled.")
from collections import Counter
import re
try:
    # Import dynamically to avoid static analysis/import-time errors when
    # the optional dependency isn't installed in the environment.
    import importlib
    wc = importlib.import_module("wordcloud")
    WordCloud = getattr(wc, "WordCloud")
except Exception:
    WordCloud = None
    st.warning("Optional dependency 'wordcloud' is not available. Word cloud generation will be disabled.")
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None
    st.warning("Optional dependency 'matplotlib' is not available. Plots may be disabled.")

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================
st.title("📊 Customer Feedback Analyzer")
st.markdown(
    """
    Analyze customer reviews using NLP and Sentiment Analysis.

    **Features**
    - Sentiment Analysis
    - Customer Satisfaction Score (CSAT)
    - Complaint Detection
    - Business Recommendations
    - Interactive Visualizations
    """
)

# ==========================================================
# SAMPLE DATA
# ==========================================================
sample_reviews = [
    "The product quality is excellent and delivery was fast.",
    "Customer support was very helpful and professional.",
    "The package arrived late and the box was damaged.",
    "Delivery took too long and I am disappointed.",
    "Amazing service and great experience.",
    "Poor quality product and terrible support.",
    "Fast shipping and excellent packaging.",
    "The item stopped working after one week.",
    "Customer service solved my issue quickly.",
    "Late delivery and bad packaging."
]

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Data Input")

data_option = st.sidebar.radio(
    "Choose Input Method",
    [
        "Paste Reviews",
        "Upload File",
        "Load Sample Data"
    ]
)

reviews = []

# ==========================================================
# PASTE REVIEWS
# ==========================================================
if data_option == "Paste Reviews":

    review_text = st.text_area(
        "Paste Reviews (One Review Per Line)",
        height=250
    )

    if review_text.strip():
        reviews = [
            line.strip()
            for line in review_text.split("\n")
            if line.strip()
        ]

# ==========================================================
# UPLOAD FILE
# ==========================================================
elif data_option == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        try:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

            if "Review" not in df.columns:
                st.error(
                    "Uploaded file must contain a column named 'Review'"
                )

            else:
                reviews = (
                    df["Review"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ==========================================================
# SAMPLE DATA
# ==========================================================
else:

    if st.sidebar.button("Load Sample Dataset"):
        reviews = sample_reviews

# ==========================================================
# ANALYSIS FUNCTIONS
# ==========================================================
def get_sentiment(text):
    # If TextBlob (sentiment engine) is not available, return a default neutral result.
    if not sentiment_enabled or TextBlob is None:
        return 0.0, "Sentiment Disabled"

    score = TextBlob(text).sentiment.polarity
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label


def extract_keywords(texts):

    stop_words = {
        "the", "a", "an", "and", "or", "is",
        "was", "were", "to", "of", "for",
        "with", "on", "in", "it", "this",
        "that", "very", "my", "i"
    }

    words = []

    for text in texts:

        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        for token in tokens:
            if len(token) > 3 and token not in stop_words:
                words.append(token)

    return Counter(words)


def generate_recommendations(common_words):

    recommendations = []

    keywords = [word for word, _ in common_words]

    if "delivery" in keywords or "late" in keywords:
        recommendations.append(
            "🚚 Optimize logistics: High volume of complaints regarding shipping delays."
        )

    if "support" in keywords:
        recommendations.append(
            "📞 Improve customer support response times and service quality."
        )

    if "quality" in keywords:
        recommendations.append(
            "🏭 Review manufacturing quality control processes."
        )

    if "packaging" in keywords:
        recommendations.append(
            "📦 Improve packaging standards to reduce damage complaints."
        )

    if "damaged" in keywords:
        recommendations.append(
            "🔍 Inspect shipping and handling procedures."
        )

    if not recommendations:
        recommendations.append(
            "✅ Overall customer feedback appears healthy. Continue monitoring trends."
        )

    return recommendations


# ==========================================================
# RUN ANALYSIS
# ==========================================================
if reviews:

    st.success(f"{len(reviews)} reviews loaded successfully.")

    df_reviews = pd.DataFrame(
        reviews,
        columns=["Review"]
    )

    sentiments = []
    scores = []

    for review in reviews:

        score, sentiment = get_sentiment(review)

        sentiments.append(sentiment)
        scores.append(score)

    df_reviews["Sentiment"] = sentiments
    df_reviews["Score"] = scores

    # ======================================================
    # METRICS
    # ======================================================
    total_reviews = len(df_reviews)

    positive_count = len(
        df_reviews[df_reviews["Sentiment"] == "Positive"]
    )

    negative_count = len(
        df_reviews[df_reviews["Sentiment"] == "Negative"]
    )

    neutral_count = len(
        df_reviews[df_reviews["Sentiment"] == "Neutral"]
    )

    csat_score = round(
        ((positive_count + (0.5 * neutral_count))
         / total_reviews) * 100,
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Positive Reviews",
        positive_count
    )

    col3.metric(
        "Negative Reviews",
        negative_count
    )

    col4.metric(
        "CSAT Score",
        f"{csat_score}%"
    )

    st.divider()

    # ======================================================
    # SENTIMENT CHARTS
    # ======================================================
    left, right = st.columns(2)

    sentiment_counts = (
        df_reviews["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    with left:

        st.subheader("Sentiment Distribution")

        fig_pie = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            hole=0.4
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with right:

        st.subheader("Sentiment Breakdown")

        fig_bar = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            text_auto=True
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # COMPLAINT ANALYSIS
    # ======================================================
    st.subheader("🔍 Most Common Complaints")

    negative_reviews = df_reviews[
        df_reviews["Sentiment"] == "Negative"
    ]["Review"].tolist()

    keyword_counts = extract_keywords(
        negative_reviews
    )

    top_keywords = keyword_counts.most_common(10)

    if top_keywords:

        keyword_df = pd.DataFrame(
            top_keywords,
            columns=["Keyword", "Frequency"]
        )

        col_a, col_b = st.columns(2)

        with col_a:

            fig_keywords = px.bar(
                keyword_df,
                x="Keyword",
                y="Frequency",
                text_auto=True
            )

            st.plotly_chart(
                fig_keywords,
                use_container_width=True
            )

        with col_b:

            wordcloud_text = " ".join(
                negative_reviews
            )

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(wordcloud_text)

            fig, ax = plt.subplots()

            ax.imshow(wordcloud)
            ax.axis("off")

            st.pyplot(fig)

    else:
        st.info(
            "No significant complaints detected."
        )

    st.divider()

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================
    st.subheader("💡 Suggested Actions")

    recommendations = generate_recommendations(
        top_keywords
    )

    for rec in recommendations:
        st.write(rec)

    st.divider()

    # ======================================================
    # REVIEW TABLE
    # ======================================================
    st.subheader("📄 Review Analysis")

    st.dataframe(
        df_reviews,
        use_container_width=True
    )

    csv = df_reviews.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇ Download Results",
        data=csv,
        file_name="customer_feedback_analysis.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Load sample data, paste reviews, or upload a file to begin analysis."
    )