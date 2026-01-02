# =========================================
# STREAMLIT TRENDING TOPICS WORDCLOUD APP
# Facebook | Reddit | Twitter (Free APIs)
# =========================================

import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Trending Topics WordCloud",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Trending Topics WordCloud Generator")
st.caption("Free APIs • NLP (TF-IDF) • WordCloud Visualization")

# ---------------- FUNCTIONS ----------------
def fetch_reddit_posts(query, limit):
    url = f"https://www.reddit.com/search.json?q={query}&limit={limit}"
    headers = {"User-Agent": "StreamlitTrendingApp/1.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return [post["data"]["title"] for post in data["data"]["children"]]

def generate_wordcloud(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    scores = tfidf_df.sum().sort_values(ascending=False)
    return dict(scores.head(100))

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📘 Facebook", "👽 Reddit", "🐦 Twitter"])

def common_ui(platform_name):
    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(f"Enter topic for {platform_name}", placeholder="e.g. AI, Elections, Football")

    with col2:
        word_limit = st.slider(
            "Number of posts",
            min_value=500,
            max_value=5000,
            step=500,
            value=1000
        )

    if st.button(f"Generate WordCloud ({platform_name})"):
        if not topic.strip():
            st.warning("Please enter a topic.")
            return

        with st.spinner("Fetching data & generating wordcloud..."):
            try:
                posts = fetch_reddit_posts(topic, word_limit)

                if len(posts) < 10:
                    st.error("Not enough data found.")
                    return

                word_freq = generate_wordcloud(posts)

                wc = WordCloud(
                    width=1000,
                    height=500,
                    background_color="white"
                ).generate_from_frequencies(word_freq)

                fig, ax = plt.subplots(figsize=(12,6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")

                st.success(f"WordCloud generated for '{topic}' on {platform_name}")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- TAB CONTENT ----------------
with tab1:
    st.subheader("📘 Facebook Trending Topics")
    st.info("Using public discussion signals (simulated via Reddit search)")
    common_ui("Facebook")

with tab2:
    st.subheader("👽 Reddit Trending Topics")
    st.info("Live Reddit public API")
    common_ui("Reddit")

with tab3:
    st.subheader("🐦 Twitter Trending Topics")
    st.info("Simulated Twitter trends using public Reddit search")
    common_ui("Twitter")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • NLP TF-IDF • WordCloud • Free APIs")
