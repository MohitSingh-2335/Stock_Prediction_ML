# src/agents/sentiment_agent.py
#
# Phase 4c: FinBERT sentiment agent. Fetches recent BTC headlines
# (CryptoCompare News API, free/no-auth), scores with FinBERT, aggregates
# to daily sentiment_score. Fails soft (skips feature, no synthetic
# default) — same contract as fear_greed_agent.py / onchain_agent.py.

import requests
import pandas as pd

NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
_pipeline = None


def _get_pipeline():
    """Lazy-load so importing this module doesn't force transformers/torch
    load for scripts that don't need sentiment."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    return _pipeline


def fetch_headlines(categories="BTC", lang="EN"):
    """CryptoCompare's free tier has no historical date-range param —
    returns whatever is currently in the feed (typically last few days)."""
    try:
        resp = requests.get(
            NEWS_URL, params={"categories": categories, "lang": lang}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("Data", [])
        if not data:
            return None
        df = pd.DataFrame(data)[["published_on", "title"]]
        df["date"] = pd.to_datetime(df["published_on"], unit="s").dt.date
        return df[["date", "title"]]
    except Exception as e:
        print(f"[sentiment_agent] fetch failed: {e}")
        return None


def score_headlines(headlines_df):
    """Per-day mean(P(positive) - P(negative)), range [-1, 1]."""
    try:
        clf = _get_pipeline()
        results = clf(headlines_df["title"].tolist(), truncation=True)
        signed = []
        for r in results:
            label, conf = r["label"].lower(), r["score"]
            signed.append(conf if label == "positive" else (-conf if label == "negative" else 0.0))
        out = headlines_df.copy()
        out["score"] = signed
        return out.groupby("date")["score"].mean().reset_index(name="sentiment_score")
    except Exception as e:
        print(f"[sentiment_agent] scoring failed: {e}")
        return None


def merge_sentiment(df):
    """Merges daily sentiment_score into hourly OHLCV df by date.
    Fails soft: returns df unchanged on any fetch/scoring failure."""
    headlines = fetch_headlines()
    if headlines is None:
        print("[sentiment_agent] no headlines — skipping sentiment_score feature.")
        return df

    daily = score_headlines(headlines)
    if daily is None:
        return df

    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df = df.merge(daily, on="date", how="left")
    df.drop(columns=["date"], inplace=True)
    return df