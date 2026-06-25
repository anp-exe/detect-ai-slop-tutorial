import os, re, requests
from dotenv import load_dotenv
from card import make_card

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
             "thought leader", "grateful", "blessed", "move the needle"]
CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]
QUESTION_BAIT = ["do you think", "what's your", "what are your", "what do you",
                 "how many", "how do you", "let me know", "drop a comment"]
FILLER = ["simply", "genuinely", "truly", "really", "actually",
          "literally", "honestly", "ultimately", "absolutely"]
ANTITHESIS = [r"it'?s not\b.{0,70}?\bit'?s\b",
              r"isn'?t (?:just |always |only |about |simply )?.{0,70}?\bit'?s\b",
              r"not just\b.{0,70}?\bbut\b"]

def count_hits(text, phrases):
    return sum(text.lower().count(p) for p in phrases)

def engagement_bait(text):
    hits = count_hits(text, CLOSERS) + count_hits(text, QUESTION_BAIT)
    if text.strip().endswith("?"):
        hits += 1
    return hits

def antithesis_hits(text):
    return sum(len(re.findall(p, text.lower())) for p in ANTITHESIS)

def count_dashes(text):
    return text.count("—") + text.count("–") + text.count(" - ")

def rule_signals(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 5)
    broetry = short / len(lines) if lines else 0
    emoji_bullets = sum(1 for l in lines if not l[0].isascii())

    dashes = count_dashes(text)

    score = min(20, broetry * 28)
    score += min(14, count_hits(text, BUZZWORDS) * 4)
    score += min(12, engagement_bait(text) * 6)
    score += min(8, emoji_bullets * 2)
    score += min(8, max(0, text.count("#") - 2) * 2)
    score += min(8, max(0, dashes - 3) * 3)
    score += min(10, antithesis_hits(text) * 6)
    score += min(8, count_hits(text, FILLER) * 3)
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
    if score >= 80: return "Certified Artisanal Slop 🥫"
    if score >= 60: return "Peak LinkedIn Cringe 💼"
    if score >= 40: return "Mildly Insufferable 😬"
    return "An Actual Human Wrote This 😮"

def main():
    if not HF_TOKEN:
        raise SystemExit("No token found. Add HF_TOKEN=hf_... to your .env file.")

    text = """"Hi Kirstie, unfortunately 4 stages is just overkill for a job, and puts me off going for roles like this."

This was feedback I received from a candidate for a Senior Individual Contributor role.

The process was:

📞 Initial screening call
💻 Technical test
💬 Discussion about the test/wider technical assessment 
🤝 Final interview

Four touchpoints. Yet, from the candidate's perspective, it was simply too much. 

Every interview stage usually has a good reason behind it - assessing technical skills, understanding how someone thinks, or getting stakeholder buy-in.

But while each stage makes sense on its own, together they can become a barrier. The best candidates are often the busiest, and if the process feels too time-consuming, they'll simply opt out.

The goal isn't always fewer interviews, it's making sure every stage genuinely adds value.

How many interview stages do you think is reasonable for a senior IC role?"""

    rules = rule_signals(text)
    hf = hf_performative_score(text, HF_TOKEN)
    score = round(min(100, rules + hf * 40))
    print(f"\n  Slop Score: {score}/100  —  {verdict(score)}\n")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    signals = {
        "broetry": (sum(1 for l in lines if len(l.split()) <= 5) / len(lines)) if lines else 0,
        "buzzwords": count_hits(text, BUZZWORDS),
        "closers": engagement_bait(text),
        "hashtags": text.count("#"),
        "emoji_bullets": sum(1 for l in lines if not l[0].isascii()),
        "dashes": count_dashes(text),
        "antithesis": antithesis_hits(text),
        "filler": count_hits(text, FILLER),
    }
    make_card(score, signals)

if __name__ == "__main__":
    main()