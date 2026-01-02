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

# ---------------- DATA FETCH FUNCTIONS ----------------

def fetch_reddit_posts(query, limit):
    url = "https://api.reddit.com/search"
    headers = {"User-Agent": "Mozilla/5.0 (StreamlitApp/1.0)"}
    params = {"q": query, "limit": min(limit, 100), "sort": "hot"}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()

    return [p["data"]["title"] for p in data["data"]["children"]]


def fetch_facebook_posts(query, limit):
    url = "https://api.reddit.com/search"
    headers = {"User-Agent": "Mozilla/5.0 (StreamlitApp/1.0)"}
    params = {"q": query, "limit": min(limit, 100), "sort": "top", "t": "week"}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()

    posts = []
    for p in data["data"]["children"]:
        title = p["data"]["title"]
        body = p["data"].get("selftext", "")
        posts.append(f"{title} {body}")

    return posts


def fetch_twitter_posts(query, limit):
    url = "https://api.reddit.com/search"
    headers = {"User-Agent": "Mozilla/5.0 (StreamlitApp/1.0)"}
    params = {"q": query, "limit": min(limit, 100), "sort": "new"}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()

    return [p["data"]["title"] for p in data["data"]["children"]]

# ---------------- WORDCLOUD FUNCTION ----------------

def generate_wordcloud_from_text(texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
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

# ---------------- COMMON UI FUNCTION ----------------

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
            min_value=500,
            max_value=5000,
            step=500,
            value=1000,
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
    st.info("Simulated using public long-form discussion data")
    common_ui("Facebook")

with tab2:
    st.subheader("👽 Reddit Trending Topics")
    st.info("Live Reddit public API")
    common_ui("Reddit")

with tab3:
    st.subheader("🐦 Twitter Trending Topics")
    st.info("Simulated using fast-moving public discussions")
    common_ui("Twitter")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • NLP TF-IDF • WordCloud • Free APIs")
