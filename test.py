import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
BUZZWORDS = ["humbled", "thrilled to announce", "excited to share",
             "game-changer", "synergy", "leverage", "move the needle",
             "thought leader", "disrupt", "growth mindset", "deep dive",
             "grateful", "blessed", "unpopular opinion"]

CLOSERS = ["agree?", "thoughts?", "who's with me", "comment below",
           "what do you think", "repost if"]
def count_hits(text, phrases):
    text = text.lower()
    return sum(text.count(phrase) for phrase in phrases)
def rule_signals(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # "broetry": what fraction of lines are tiny one-liners?
    short = sum(1 for line in lines if len(line.split()) <= 5)
    broetry = short / len(lines) if lines else 0

    buzzwords = count_hits(text, BUZZWORDS)
    closers = count_hits(text, CLOSERS)

    # lines that open with an emoji (or any non-keyboard character)
    emoji_bullets = sum(1 for line in lines if not line[0].isascii())

    hashtags = text.count("#")

    # each signal adds points, capped so no single one dominates
    score = 0
    score += min(20, broetry * 28)              # broetry is worth up to 20
    score += min(14, buzzwords * 4)             # 4 points per buzzword
    score += min(10, closers * 6)               # 6 points per "Agree?"
    score += min(8, emoji_bullets * 2)          # 2 points per emoji bullet
    score += min(8, max(0, hashtags - 2) * 2)   # 2 free hashtags, then 2 points each

    return {"broetry": round(broetry, 2), "buzzwords": buzzwords,
            "closers": closers, "emoji_bullets": emoji_bullets,
            "hashtags": hashtags, "rule_subscore": round(min(60, score), 1)}
import requests

HF_MODEL = "facebook/bart-large-mnli"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

def hf_performative_score(text, token):
    labels = ["humble authentic personal story",
              "performative self-promotional corporate content"]

    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    response = requests.post(HF_URL,
                             headers={"Authorization": f"Bearer {token}"},
                             json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    # the API returns a list of {"label": ..., "score": ...} dicts
    scores = {item["label"]: item["score"] for item in data}
    return scores.get("performative self-promotional corporate content", 0.0)
def analyze(text, token):
    sig = rule_signals(text)
    hf = hf_performative_score(text, token)
    score = round(min(100, sig["rule_subscore"] + hf * 40))
    return score, sig

def verdict(score):
    if score >= 80: return "Certified Artisanal Slop 🥫"
    if score >= 60: return "Peak LinkedIn Cringe 💼"
    if score >= 40: return "Mildly Insufferable 😬"
    if score >= 20: return "Suspiciously Normal 🤔"
    return "An Actual Human Wrote This 😮"
def main():
    if not HF_TOKEN:
        raise SystemExit("No token found. Add HF_TOKEN=hf_... to your .env file.")

    # paste the LinkedIn post you want to score between the triple quotes:
    text = """I got rejected 100 times.

Then everything changed.

Here's what I learned 👇

I'm humbled and grateful to announce I'm now a thought leader.

We need to leverage synergy to move the needle.

Agree?

#motivation #grindset #blessed"""

    score, sig = analyze(text, HF_TOKEN)
    print(f"\n  Slop Score: {score}/100  —  {verdict(score)}\n")

if __name__ == "__main__":
    main()