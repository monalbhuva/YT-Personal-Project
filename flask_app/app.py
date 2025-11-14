# src/app/app.py

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file,render_template
from flask_cors import CORS
import io
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from datetime import datetime
from googleapiclient.discovery import build
from flask import send_from_directory
from dotenv import load_dotenv
load_dotenv()


# -----------------------------
# Flask, Model, Vectorizer
# -----------------------------

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./logreg_bow_model.pkl"
VECTORIZER_PATH = "./bow_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

  # <- PUT YOUR KEY HERE

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'no', 'however', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])

        lemma = WordNetLemmatizer()
        comment = ' '.join([lemma.lemmatize(w) for w in comment.split()])

        return comment
    except:
        return comment

# -----------------------------
# Fetch Comments Using YouTube API
# -----------------------------
def fetch_youtube_comments(video_id, max_comments=300):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []

    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": top_comment["textDisplay"],
                "timestamp": top_comment["publishedAt"]
            })

            if len(comments) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# -----------------------------
# Predict Sentiments
# -----------------------------
def predict_sentiments(comments):
    processed = [preprocess_comment(x["text"]) for x in comments]
    X = vectorizer.transform(processed)
    preds = model.predict(X)

    for i, p in enumerate(preds):
        comments[i]["sentiment"] = int(p)
    return comments

# -----------------------------
# Summary
# -----------------------------
def generate_summary(df):
    total = len(df)
    counts = df["sentiment"].value_counts().to_dict()
    return {
        "total_comments": total,
        "positive": counts.get(1, 0),
        "neutral": counts.get(0, 0),
        "negative": counts.get(-1, 0)
    }

# -----------------------------
# Generate Charts (Pie, WC, Trend)
# -----------------------------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

def generate_pie_chart(summary):
    labels = ["Positive", "Neutral", "Negative"]
    sizes = [summary["positive"], summary["neutral"], summary["negative"]]
    colors = ['#36A2EB', '#C9CBCF', '#FF6384']

    plt.figure(figsize=(6, 6))
    patches, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        textprops={'color': 'white'}
    )
    plt.legend(
        patches,
        labels,
        title="Sentiment",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3
    )
    plt.axis("equal")

    path = os.path.join(STATIC_DIR, "pie_chart.png")
    plt.savefig(path, transparent=True)
    plt.close()
    return "static/pie_chart.png"


def generate_wordcloud(df):
    text = " ".join(df["text"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="black", colormap="Blues").generate(text)

    path = os.path.join(STATIC_DIR, "wordcloud.png")
    wc.to_file(path)
    return "static/wordcloud.png"


def generate_trend_graph(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    monthly = df.resample("M")["sentiment"].value_counts().unstack().fillna(0)

    plt.figure(figsize=(12, 6))
    colors = {-1: "red", 0: "gray", 1: "green"}

    for s in [-1, 0, 1]:
        if s in monthly.columns:
            plt.plot(monthly.index, monthly[s], marker="o", color=colors[s],
                     label={-1: "Negative", 0: "Neutral", 1: "Positive"}[s])

    plt.title("Sentiment Trend Over Time")
    plt.xlabel("Month")
    plt.ylabel("Comment Count")
    plt.legend()
    plt.grid(True)

    path = os.path.join(STATIC_DIR, "trend_graph.png")
    plt.savefig(path)
    plt.close()
    return "static/trend_graph.png"


# -----------------------------
# MAIN ENDPOINT — YouTube Sentiment Analysis
# -----------------------------
@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube():
    try:
        data = request.json
        url = data.get("video_url")
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Extract video ID
        # if "v=" in url:
        #     video_id = url.split("v=")[1].split("&")[0]
        # else:
        #     return jsonify({"error": "Invalid YouTube URL"}), 400

        video_id = extract_video_id(url)

        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400
        

        # Fetch comments using YouTube API
        comments = fetch_youtube_comments(video_id)
        if not comments:
            return jsonify({"error": "No comments fetched"}), 500

        # Predict sentiments
        comments = predict_sentiments(comments)
        df = pd.DataFrame(comments)

        # Summary + charts
        summary = generate_summary(df)
        pie = generate_pie_chart(summary)
        wc = generate_wordcloud(df)
        trend = generate_trend_graph(df)

        return jsonify({
            "summary": summary,
            "sample": comments[:10],
            "charts": {
                "pie_chart": pie,
                "wordcloud": wc,
                "trend_graph": trend
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",                # normal YouTube link
        r"youtu\.be/([^?&]+)",       # short link
        r"embed/([^?&]+)",           # embed link
        r"shorts/([^?&]+)"           # shorts link
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('static', filename)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
