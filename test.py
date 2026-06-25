import os, requests
from dotenv import load_dotenv
from card import make_card, offense_count

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
             "thought leader", "grateful", "blessed", "move the needle"]
CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]
ANTITHESIS = ["it's not just", "it's not about", "isn't just", "isn't about",
              "isn't always", "isn't only", "not just about", "it's not that",
              "no longer"]

def count_hits(text, phrases):
    return sum(text.lower().count(p) for p in phrases)

def engagement_bait(text):
    hits = count_hits(text, CLOSERS)
    if text.strip().endswith("?"):
        hits += 1
    return hits

def count_dashes(text):
    return text.count("—") + text.count("–") + text.count(" - ")

def anaphora_hits(text):
    """Lines that repeat another line's opening two words (e.g. 'Culture is built...')."""
    from collections import Counter
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    starts = Counter(" ".join(l.lower().split()[:2]) for l in lines if len(l.split()) >= 2)
    return sum(c for c in starts.values() if c >= 2)

def rule_signals(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 6)
    # only trust the one-liner ratio once there are enough lines to be meaningful
    broetry = short / len(lines) if len(lines) >= 6 else 0
    emoji_bullets = sum(1 for l in lines if not l[0].isascii())

    dashes = count_dashes(text)

    score = min(20, broetry * 28)
    score += min(14, count_hits(text, BUZZWORDS) * 4)
    score += min(12, engagement_bait(text) * 6)
    score += min(12, emoji_bullets * 2)
    score += min(12, max(0, text.count("#") - 2) * 2)
    score += min(8, max(0, dashes - 3) * 3)
    score += min(10, count_hits(text, ANTITHESIS) * 6)
    score += min(12, anaphora_hits(text) * 3)
    return min(80, score)

def hf_performative_score(text, token):
    labels = ["humble authentic personal story",
              "performative self-promotional corporate content"]
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    r = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                      json=payload, timeout=30)
    r.raise_for_status()
    scores = {item["label"]: item["score"] for item in r.json()}
    return scores.get(labels[1], 0.0)

def verdict(score):
    if score >= 70: return "Certified Artisanal Slop 🥫"
    if score >= 50: return "Peak LinkedIn Cringe 💼"
    if score >= 30: return "Mildly Insufferable 😬"
    if score >= 15: return "Suspiciously Normal 🤔"
    return "An Actual Human Wrote This 😮"

def main():
    text = """"
Christina Koch (44) is an electrical engineer. She holds the record for longest continuous time in space by a woman, of 328 days. Victor Glover (46) is a US Navy test pilot. He joined Nasa in 2013 and made his first spaceflight in 2020. He was the first Black person to stay on the space station for an extended period of six months. 

Congratulations are in order to Koch and Glover for paving the way for women and people of color.

"""

    rules = rule_signals(text)
    hf = hf_performative_score(text, HF_TOKEN)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    signals = {
        "broetry": (sum(1 for l in lines if len(l.split()) <= 6) / len(lines)) if len(lines) >= 6 else 0,
        "buzzwords": count_hits(text, BUZZWORDS),
        "closers": engagement_bait(text),
        "hashtags": text.count("#"),
        "emoji_bullets": sum(1 for l in lines if not l[0].isascii()),
        "dashes": count_dashes(text),
        "antithesis": count_hits(text, ANTITHESIS),
        "anaphora": anaphora_hits(text),
    }

    if offense_count(signals) == 0:
        # no tells flagged: lean on the AI alone, kept low
        score = round(hf * 25)
    else:
        # scale the whole blend up to use the full range
        score = round(min(100, (rules + hf * 40) * 1.4))

    print(f"\n  Slop Score: {score}/100  —  {verdict(score)}\n")
    make_card(score, signals)

if __name__ == "__main__":
    main()