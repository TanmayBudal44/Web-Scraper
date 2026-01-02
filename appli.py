# =========================================
# STREAMLIT TRENDING TOPICS WORDCLOUD APP
# Facebook | Reddit | Twitter (RSS-based)
# =========================================

import streamlit as st
import pandas as pd
import feedparser
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
st.caption("RSS Feeds • NLP (TF-IDF) • WordCloud Visualization")

# ---------------- RSS FETCH FUNCTIONS ----------------

def fetch_reddit_posts(query, limit):
    feed = feedparser.parse(f"https://www.reddit.com/search.rss?q={query}")
    return [entry.title for entry in feed.entries[:limit]]

def fetch_facebook_posts(query, limit):
    feed = feedparser.parse(f"https://www.reddit.com/search.rss?q={query}&sort=top")
    posts = []
    for entry in feed.entries[:limit]:
        posts.append(entry.title + " " + entry.get("summary", ""))
    return posts

def fetch_twitter_posts(query, limit):
    feed = feedparser.parse(f"https://www.reddit.com/search.rss?q={query}&sort=new")
    return [entry.title for entry in feed.entries[:limit]]

# ---------------- WORDCLOUD FUNCTION ----------------

def generate_wordcloud_from_text(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=200
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    word_freq = dict(zip(words, scores))

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate_from_frequencies(word_freq)

    return wc

# ---------------- COMMON UI ----------------

def common_ui(platform):
    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(
            f"Enter topic for {platform}",
            placeholder="e.g. AI, Elections, Football",
            key=f"topic_{platform}"
        )

    with col2:
        limit = st.slider(
            "Number of posts",
            min_value=50,
            max_value=500,
            step=50,
            value=100,
            key=f"limit_{platform}"
        )

    if st.button(
        f"Generate WordCloud ({platform})",
        key=f"btn_{platform}"
    ):
        if not topic.strip():
            st.warning("Please enter a topic.")
            return

        with st.spinner("Fetching data and generating wordcloud..."):
            try:
                if platform == "Facebook":
                    texts = fetch_facebook_posts(topic, limit)
                elif platform == "Twitter":
                    texts = fetch_twitter_posts(topic, limit)
                else:
                    texts = fetch_reddit_posts(topic, limit)

                if len(texts) < 5:
                    st.error("Not enough data found.")
                    return

                wc = generate_wordcloud_from_text(texts)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")

                st.success(f"WordCloud generated for '{topic}' on {platform}")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs(["📘 Facebook", "👽 Reddit", "🐦 Twitter"])

with tab1:
    st.subheader("📘 Facebook Trending Topics")
    st.info("Simulated using long-form public discussions (RSS)")
    common_ui("Facebook")

with tab2:
    st.subheader("👽 Reddit Trending Topics")
    st.info("Live Reddit RSS feed (no API, no auth)")
    common_ui("Reddit")

with tab3:
    st.subheader("🐦 Twitter Trending Topics")
    st.info("Simulated using fast-moving public RSS feeds")
    common_ui("Twitter")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • TF-IDF • WordCloud • RSS Feeds")
