import os, requests
from dotenv import load_dotenv
from card import make_card

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

def count_hits(text, phrases):
    return sum(text.lower().count(p) for p in phrases)

CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]

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

def broetry_ratio(text):
    """Fraction of lines that are tiny one-liners (only trusted once there are 6+ lines)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 6)
    return short / len(lines) if len(lines) >= 6 else 0

def emoji_bullets(text):
    """How many lines start with an emoji (decorative bullets)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return sum(1 for l in lines if not l[0].isascii())

BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
             "thought leader", "grateful", "blessed", "move the needle"]

def rule_signals(text):
    dashes = count_dashes(text)
    score = min(20, broetry_ratio(text) * 28)
    score += min(14, count_hits(text, BUZZWORDS) * 4)
    score += min(12, engagement_bait(text) * 6)
    score += min(12, emoji_bullets(text) * 2)
    score += min(12, max(0, text.count("#") - 2) * 2)
    score += min(8, max(0, dashes - 3) * 3)
    score += min(12, anaphora_hits(text) * 3)
    return min(80, score)

def offense_count(signals):
    """How many slop signals tripped their threshold (the boxes on the card)."""
    flags = [
        signals["broetry"] >= 0.4,
        signals["buzzwords"] >= 1,
        signals["closers"] >= 1,
        signals["hashtags"] >= 4,
        signals["emoji_bullets"] >= 2,
        signals["dashes"] > 3,
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

def verdict(score):
    if score >= 70: return "Certified Artisanal Slop 🥫"
    if score >= 50: return "Peak LinkedIn Cringe 💼"
    if score >= 30: return "Mildly Insufferable 😬"
    if score >= 15: return "Suspiciously Normal 🤔"
    return "An Actual Human Wrote This 😮"

def main():
    # 👇 paste your own post between the triple quotes!
    text = """"
She Plays To Win STEM Challenge Day for Secondary School Girls 
📅 26 June 2026
🎓 Secondary School Girls STEM Challenge Day in London 
👩‍🎓 Empowering girls through Chess♟️ | Maths 🔢 | Coding 💻
Giving Girls the Best Opening Move in Life

We are thrilled to host this amazing day with several secondary school girls and invite inspiring STEM professionals, industry leaders, innovators and role models coming together to empower the next generation of girls:
♟️ Strategic Thinking through Chess
💻 Creative Problem Solving through Coding
🔢 Mathematical Confidence & Challenge
🚀 Real-World STEM Inspiration

Hugely indebted to FundApps Get with the Program Aila Money Fai M.
STEM Learning UK 

Thankful to each of our speakers for making time to join us and share their career journeys and run sessions to inspire a future engineer, scientist, technologist, entrepreneur, data scientist or innovator! Jeni Trice Nitika Vyas, CFA 

Credit to the Founder Lorin D'Costa, our Trustees Kanwal Bhatia Dipal Patel Jennifer Gelain-Sohn Naheed Vyas Rashmi Prabhakar, our Volunteers Alisha Vyas Emmanuelle Gelain-Sohn for sparing time to further this cause. And some of our supporters and allies Nadia Edwards-Dashti Charlotte de Metz 

By sharing experiences, challenges and successes, we can help girls see what is possible when talent meets opportunity:
✨ Be the role model you wish you'd had.
✨ Inspire confidence and ambition.
✨ Help unlock future STEM careers.

Empowering Girls. Building Confidence. Solving Problems, Creating Futures.
www.sheplaystowin.co.uk

hashtag#GivingGirlsTheBestOpeningMoveInLife hashtag#ShePlaysToWin hashtag#STEMChallengeDay hashtag#GirlsinSTEM hashtag#WomenInTech hashtag#FutureLeaders hashtag#GirlsInTech hashtag#GirlsWhoCode hashtag#ChessInEducation hashtag#STEMEducation hashtag#RoleModelsMatter hashtag#DiversityInTech hashtag#InspiringTheFuture hashtag#SecondarySchools hashtag#STEMCareers

Chess♟️ | Maths 🔢 | Coding 💻

"""

    rules = rule_signals(text)
    hf = hf_performative_score(text, HF_TOKEN)

    signals = {
        "broetry": broetry_ratio(text),
        "buzzwords": count_hits(text, BUZZWORDS),
        "closers": engagement_bait(text),
        "hashtags": text.count("#"),
        "emoji_bullets": emoji_bullets(text),
        "dashes": count_dashes(text),
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