<img src="header.png" width="60%">

# Build an AI Slop Detector with the Hugging Face API

> **Project Tutorials** / `PYTHON` `AI` `INTERMEDIATE`
>
> **by Anna** ([@anp-exe](https://www.codedex.io/@anp-exe)) ·
>
> 45 min read
>
> |                   |                                        |
> |-------------------|----------------------------------------|
> | **PREREQUISITES** | Python fundamentals                    |
> | **VERSIONS**      | Python 3.10, requests 2.x, Pillow 10.x |

## Introduction

![slop.gif](slop.gif)

Are you sick of reading AI slop on LinkedIn? The "I got rejected 100 times. Then everything changed 👇" broetry, the buzzword soup, the "Agree?" bait?

In this tutorial, we'll build a tool that gives any post a **Slop Score /100** with a verdict, then saves it as a shareable card.

<img src="img.png" width="450">

> **A quick note:** truly detecting whether an AI *wrote* something is famously unreliable, even the paid tools get it wrong. So instead we'll measure how much a post reeks of the **AI-slop *style***: the broetry, buzzwords, and engagement bait.

Along the way you'll learn the **Hugging Face API** for **zero-shot text classification**, and how to blend AI judgment with your own transparent rules.

## What is Hugging Face? 🤗

Hugging Face is a platform for machine learning models. It offers tons of pre-trained models for tasks like text classification and sentiment analysis, all callable through a simple API with just a few lines of Python.

## What We're Building

1. **Rule signals**: functions that sniff out broetry, buzzwords, and bait.
2. **Hugging Face**: a zero-shot model that scores how "performative" a post feels.
3. **A Slop Score**: both halves combined into one number with a verdict.
4. **A reward**: a card generator that turns your score into a shareable image. 🥫

## Setting Up

We'll need [Python 3](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/). Create a file called **slop.py**, then install three packages:

```bash
pip install requests Pillow python-dotenv
```

`requests` talks to Hugging Face, `Pillow` draws the card, and `python-dotenv` keeps your token safe.

## Getting a Hugging Face Token

The **Inference API** runs AI models with a simple web request. No GPU, no downloads. Just grab a free token:

1. Make a free account at [huggingface.co](https://huggingface.co).
2. Go to **Settings → Access Tokens → New token** (a "Read" token is fine).
3. Copy it (it starts with `hf_`).

![img_5.png](img_5.png)

> ⚠️ Treat your token like a password. Never paste it into your code or commit it to GitHub.

Create a file called `.env` and add your token on one line, no quotes:

```
HF_TOKEN=hf_your_token_here
```

Then load it at the top of **slop.py**:

```python
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
```

> 💡 Add `.env` to your `.gitignore` so your token never reaches GitHub. Secrets live in `.env`, never in the code.

## Step 1: Sniff Out the Slop (Rule Signals)

A lot of slop is detectable with simple patterns before we even touch AI. Start with the phrases we're hunting for:

```python
BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
    "thought leader", "grateful", "blessed", "move the needle"]
CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]
```

A one-liner helper counts how many of those phrases appear:

```python
def count_hits(text, phrases):
    return sum(text.lower().count(phrase) for phrase in phrases)
```

Now the scoring function. First, **broetry**, the fraction of lines that are tiny one-liners (the signature LinkedIn format):

```python
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 5)
    broetry = short / len(lines) if lines else 0
```

Then **emoji bullets**, using a neat trick: `isascii()` is `False` for emoji, so a line *starting* with one is almost certainly a ✨ decorative ✨ bullet:

```python
    emoji_bullets = sum(1 for l in lines if not l[0].isascii())
```

Finally we add up every signal, each capped with `min()` so no single offense can max out the score on its own:

```python
    score = min(20, broetry * 28)
    score += min(14, count_hits(text, BUZZWORDS) * 4)
    score += min(10, count_hits(text, CLOSERS) * 6)
    score += min(8, emoji_bullets * 2)
    score += min(8, max(0, text.count("#") - 2) * 2)
```

These signals are *transparent*: you can see exactly why a post scored high, which makes the result feel fair (and funny). The full function is in the assembled file at the end.

## Step 2: Bring in the AI (Hugging Face Zero-Shot)

Rules only go so far. To catch the *overall vibe* we'll use a **zero-shot classifier**: a model that sorts text into labels *we invent on the spot*, no training needed.

We invent two labels and `POST` the post text to the model:

```python
labels = ["humble authentic personal story",
          "performative self-promotional corporate content"]
payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
```

The model returns a probability for each label, and we pull out the "performative" one (a number from 0 to 1):

```python
data = response.json()
scores = {item["label"]: item["score"] for item in data}
return scores.get("performative self-promotional corporate content", 0.0)
```

How does it work? The model was trained to judge whether one sentence *implies* another. We exploit that by effectively asking "does this post imply the label 'performative content'?" That's the magic.

> 💡 **First-run tip:** free models "sleep" when idle, so your first request might take ~20 seconds while the model wakes up. Just run it again.

## Step 3: Combine into a Slop Score

Now we blend the two halves: the rule subscore (0 to 60) plus the AI's probability scaled to 0 to 40.

```python
score = round(min(100, rule_subscore + hf * 40))
```

Then map the number to a verdict:

```python
def verdict(score):
    if score >= 80: return "Certified Artisanal Slop 🥫"
    if score >= 60: return "Peak LinkedIn Cringe 💼"
    if score >= 40: return "Mildly Insufferable 😬"
    if score >= 20: return "Suspiciously Normal 🤔"
    return "An Actual Human Wrote This 😮"
```

Splitting the score this way is deliberate: if the AI is unsure, the transparent rules still ground the result, and vice versa. A genuinely useful pattern for any "AI + heuristics" project.

## Step 4: Put It All Together

Here's the complete **slop.py**. Paste the post you want to score between the triple quotes:

```python
import os, requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

HF_MODEL = "facebook/bart-large-mnli"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
    "thought leader", "grateful", "blessed", "move the needle"]
CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]

def count_hits(text, phrases):
    return sum(text.lower().count(phrase) for phrase in phrases)

def rule_signals(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 5)
    broetry = short / len(lines) if lines else 0
    emoji_bullets = sum(1 for l in lines if not l[0].isascii())

    score = min(20, broetry * 28)
    score += min(14, count_hits(text, BUZZWORDS) * 4)
    score += min(10, count_hits(text, CLOSERS) * 6)
    score += min(8, emoji_bullets * 2)
    score += min(8, max(0, text.count("#") - 2) * 2)
    return round(min(60, score), 1)

def hf_performative_score(text, token):
    labels = ["humble authentic personal story",
              "performative self-promotional corporate content"]
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    response = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                             json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    scores = {item["label"]: item["score"] for item in data}
    return scores.get("performative self-promotional corporate content", 0.0)

def verdict(score):
    if score >= 80: return "Certified Artisanal Slop 🥫"
    if score >= 60: return "Peak LinkedIn Cringe 💼"
    if score >= 40: return "Mildly Insufferable 😬"
    if score >= 20: return "Suspiciously Normal 🤔"
    return "An Actual Human Wrote This 😮"

def main():
    if not HF_TOKEN:
        raise SystemExit("No token found. Add HF_TOKEN=hf_... to your .env file.")

    text = """I got rejected 100 times.
Then everything changed.
Here's what I learned 👇
I'm humbled and grateful to announce I'm now a thought leader.
We need to leverage synergy to move the needle.
Agree?
#motivation #grindset #blessed"""

    rules = rule_signals(text)
    hf = hf_performative_score(text, HF_TOKEN)
    score = round(min(100, rules + hf * 40))
    print(f"\n  Slop Score: {score}/100  —  {verdict(score)}\n")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python slop.py
```

```
  Slop Score: 88/100  —  Certified Artisanal Slop 🥫
```

Try it on posts from your feed. The worse the post, the higher the score.

## Step 5: Your Reward, a Shareable Card

A terminal score is fun, but you want something to *post*. Grab two files from the project repo and drop them in your folder:

- **`card.py`**: the card generator
- **`NotoColorEmoji.ttf`**: the emoji font, so your card looks the same on every computer

Add the import and call `make_card` at the end of `main()`:

```python
from card import make_card
make_card(score, verdict(score))
```

Run **slop.py** again and a **slop_card.png** appears in your folder, ready to post. 🎉

> <img src="slop_card.png" width="400">

> 💡 **Want to peek inside `card.py`?** Go for it. It uses Pillow to draw a gradient, a circular score meter (`arc`), and rounded corners (a mask). A great file to study once the main project works.

## Conclusion

You did it! You learned how to:

- Use the **Hugging Face Inference API** with **zero-shot classification** (invent your own labels!)
- Combine **AI judgment with transparent rules**, a useful real-world pattern
- Keep your API token safe with a `.env` file
- Turn a result into a shareable card

## What Next?

- **More signals:** detect the "🧵 thread" opener or ALL CAPS WORDS.
- **Browser extension:** score posts right in your LinkedIn feed.
- **Web app:** wrap it in Streamlit so anyone can paste and score.

## More Resources

- [Hugging Face Inference API docs](https://huggingface.co/docs/api-inference)
- [Zero-shot classification explained](https://huggingface.co/tasks/zero-shot-classification)
- [Pillow documentation](https://pillow.readthedocs.io/)