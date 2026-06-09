"""
LinkedIn Slop Detector — paste a post, get a Slop Score.
Powered by the Hugging Face API (zero-shot classification) + simple rule signals.
"""

import os
import re
import requests
from dotenv import load_dotenv
from card import make_card # the bonus card generator (see card.py)

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = "facebook/bart-large-mnli"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# ---- rule-based "slop signals" ----
BUZZWORDS = ["humbled","thrilled to announce","excited to share","game-changer",
             "synergy","leverage","move the needle","thought leader","disrupt",
             "growth mindset","deep dive","grateful","blessed","unpopular opinion"]
CLOSERS = ["agree?","thoughts?","who's with me","comment below",
           "what do you think","repost if"]

def _hits(text, phrases):
    t = text.lower()
    return sum(t.count(p) for p in phrases)

def rule_signals(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    broetry = (sum(1 for l in lines if len(l.split()) <= 5)/len(lines)) if lines else 0
    buzz = _hits(text, BUZZWORDS)
    closers = _hits(text, CLOSERS)
    emoji_bullets = len(re.findall(r"^[\s]*[\U0001F300-\U0001FAFF\u2600-\u27BF]", text, re.M))
    hashtags = len(re.findall(r"#\w+", text))
    score = (min(20, broetry*28) + min(14, buzz*4) + min(10, closers*6)
             + min(8, emoji_bullets*2) + min(8, max(0, hashtags-2)*2))
    return {"broetry": round(broetry,2), "buzzwords": buzz, "closers": closers,
            "emoji_bullets": emoji_bullets, "hashtags": hashtags,
            "rule_subscore": round(min(60, score),1)}

# ---- the Hugging Face zero-shot call ----
def hf_performative_score(text, token):
    labels = ["humble authentic personal story",
              "performative self-promotional corporate content"]
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    r = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                      json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # the router returns a list of {"label":..., "score":...} dicts
    scores = {item["label"]: item["score"] for item in data}
    return scores.get("performative self-promotional corporate content", 0.0)

def verdict(score):
    if score >= 80: return "Certified Artisanal Slop 🥫"
    if score >= 60: return "Peak LinkedIn Cringe 💼"
    if score >= 40: return "Mildly Insufferable 😬"
    if score >= 20: return "Suspiciously Normal 🤔"
    return "An Actual Human Wrote This 😮"

def analyze(text, token):
    sig = rule_signals(text)
    hf = hf_performative_score(text, token)
    score = round(min(100, sig["rule_subscore"] + hf*40))
    return score, sig

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

    # bonus: save a shareable card (see card.py)
    make_card(score, sig)
    print("  Saved your card to slop_card.png 🥫\n")

if __name__ == "__main__":
    main()