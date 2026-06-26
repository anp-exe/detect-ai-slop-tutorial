import os, requests
from dotenv import load_dotenv
from card import make_card

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

def count_corporate_buzzwords(text):
    """How many LinkedIn buzzwords the post uses (humbled, synergy, ...)."""
    BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
                 "thought leader", "grateful", "blessed", "move the needle"]
    return sum(text.lower().count(b) for b in BUZZWORDS)

def engagement_bait(text):
    """Comment-bait closers, plus a point if the whole post ends on a question."""
    CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]
    hits = sum(text.lower().count(c) for c in CLOSERS)
    return hits + 1 if text.strip().endswith("?") else hits

def excess_dashes(text):
    """Dashes beyond a 3-dash grace (em-dashes, en-dashes, spaced hyphens)."""
    DASHES = ["—", "–", " - "]
    total = sum(text.count(d) for d in DASHES)
    return max(0, total - 3)

def anaphora_hits(text):
    """Lines that repeat another line's opening two words (e.g. 'Culture is built...')."""
    from collections import Counter
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    starts = Counter(" ".join(l.lower().split()[:2]) for l in lines if len(l.split()) >= 2)
    return sum(c for c in starts.values() if c >= 2)

def broetry_ratio(text):
    """Fraction of lines that are tiny one-liners (only trusted once there are 6+ lines)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 6)
    return short / len(lines) if len(lines) >= 6 else 0

def emoji_bullets(text):
    """How many lines start with an emoji (decorative bullets)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return sum(1 for l in lines if not l[0].isascii())

def rule_score(text):
    score = min(20, broetry_ratio(text) * 28)
    score += min(14, count_corporate_buzzwords(text) * 4)
    score += min(12, engagement_bait(text) * 6)
    score += min(12, emoji_bullets(text) * 2)
    score += min(8, excess_dashes(text) * 3)
    score += min(12, anaphora_hits(text) * 3)
    return min(80, score)

def offense_count(signals):
    """How many slop signals tripped their threshold (the boxes on the card)."""
    flags = [
        signals["broetry"] >= 0.4,
        signals["buzzwords"] >= 1,
        signals["closers"] >= 1,
        signals["emoji_bullets"] >= 2,
        signals["dashes"] > 0,
        signals["anaphora"] >= 2,
    ]
    return sum(flags)

def hf_performative_score(text, token):
    labels = ["humble authentic personal story",
              "performative self-promotional corporate content"]
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    r = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                      json=payload, timeout=30)
    r.raise_for_status()
    scores = {item["label"]: item["score"] for item in r.json()}
    return scores.get(labels[1], 0.0)

def main():
    # paste your own post between the triple quotes!
    text = """"
I got rejected 100 times.
Then everything changed.
We need to leverage synergy to move the needle.
Culture is built when teams win.
Culture is built when people care.
Agree?
#motivation #grindset #blessed

"""

    rules = rule_score(text)
    hf = hf_performative_score(text, HF_TOKEN)

    signals = {
        "broetry": broetry_ratio(text),
        "buzzwords": count_corporate_buzzwords(text),
        "closers": engagement_bait(text),
        "emoji_bullets": emoji_bullets(text),
        "dashes": excess_dashes(text),
        "anaphora": anaphora_hits(text),
    }

    if offense_count(signals) == 0:
        # no tells flagged: lean on the AI alone, kept low
        score = round(hf * 25)
    else:
        # scale the whole blend up to use the full range
        score = round(min(100, (rules + hf * 40) * 1.4))

    if score >= 70:   label = "Certified Artisanal Slop 🥫"
    elif score >= 50: label = "Peak LinkedIn Cringe 💼"
    elif score >= 30: label = "Mildly Insufferable 😬"
    elif score >= 15: label = "Suspiciously Normal 🤔"
    else:             label = "An Actual Human Wrote This 😮"

    print(f"\n  Slop Score: {score}/100  —  {label}\n")
    make_card(score, signals)

if __name__ == "__main__":
    main()