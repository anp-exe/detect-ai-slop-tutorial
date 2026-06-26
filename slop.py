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

def score_signals(text):
    """Run each deterministic signal once, then derive the weighted score, the offense count, and the raw values (for the card)."""
    broetry   = broetry_ratio(text)
    buzzwords = count_corporate_buzzwords(text)
    closers   = engagement_bait(text)
    emoji     = emoji_bullets(text)
    dashes    = excess_dashes(text)
    anaphora  = anaphora_hits(text)

    score = min(20, broetry * 28)
    score += min(14, buzzwords * 4)
    score += min(12, closers * 6)
    score += min(12, emoji * 2)
    score += min(8, dashes * 3)
    score += min(12, anaphora * 3)

    offenses = sum([
        broetry >= 0.4,
        buzzwords >= 1,
        closers >= 1,
        emoji >= 2,
        dashes > 0,
        anaphora >= 2,
    ])

    signals = {
        "broetry": broetry,
        "buzzwords": buzzwords,
        "closers": closers,
        "emoji_bullets": emoji,
        "dashes": dashes,
        "anaphora": anaphora,
    }
    return min(80, score), offenses, signals

def zero_shot(text, labels, token):
    """Ask the zero-shot classifier for the probability that text matches labels[1]."""
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    r = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                      json=payload, timeout=30)
    r.raise_for_status()
    scores = {item["label"]: item["score"] for item in r.json()}
    return scores.get(labels[1], 0.0)

def performative_score(text, token):
    """How performative and self-promotional the post reads (vs a humble personal story)."""
    labels = ["humble authentic personal story",
              "performative self-promotional corporate content"]
    return zero_shot(text, labels, token)

def generic_score(text, token):
    """How much the post reads like generic AI filler (vs a specific personal experience)."""
    labels = ["a specific personal experience",
              "generic AI-generated filler"]
    return zero_shot(text, labels, token)

def score_ai_signals(text, token):
    """Run each AI signal once and average them into one 0-1 slop 'vibe'."""
    performative = performative_score(text, token)
    generic      = generic_score(text, token)
    return (performative + generic) / 2

def main():
    # paste your own post between the triple quotes!
    text = """I got rejected 100 times.
Then everything changed.
We need to leverage synergy to move the needle.
Culture is built when teams win.
Culture is built when people care.
Agree?
#motivation #grindset #blessed"""

    rules, offenses, signals = score_signals(text)
    vibe = score_ai_signals(text, HF_TOKEN)

    if offenses == 0:
        # no tells flagged: lean on the AI alone, kept low
        score = round(vibe * 25)
    else:
        # scale the whole blend up to use the full range
        score = round(min(100, (rules + vibe * 40) * 1.4))

    if score >= 70:   label = "Certified Artisanal Slop 🥫"
    elif score >= 50: label = "Peak LinkedIn Cringe 💼"
    elif score >= 30: label = "Mildly Insufferable 😬"
    elif score >= 15: label = "Suspiciously Normal 🤔"
    else:             label = "An Actual Human Wrote This 😮"

    print(f"\n  Slop Score: {score}/100  —  {label}\n")
    make_card(score, signals)

if __name__ == "__main__":
    main()