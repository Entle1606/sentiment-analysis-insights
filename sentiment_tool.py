#!/usr/bin/env python3
"""
Aspect-Based Sentiment & Emotion Analysis Tool
==============================================
Scores customer feedback with a lexicon approach (word valence + negation +
intensifiers), breaks sentiment down by aspect (delivery, support, pricing...),
detects emotions, and prints insights with suggested solutions.

Standard library only. matplotlib is optional (used for a chart if installed).

Usage
-----
    python sentiment_tool.py --sample
    python sentiment_tool.py reviews.csv --column review
    python sentiment_tool.py reviews.csv --column review --out results

If the CSV also has a "label" column (positive / neutral / negative), the tool
reports accuracy against it, which is useful for validating the scorer.
"""
import argparse
import csv
import math
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# 1. Lexicons
# --------------------------------------------------------------------------
POS = {"love": 3, "loved": 3, "great": 2.5, "excellent": 3, "amazing": 3, "fantastic": 3,
       "awesome": 3, "good": 2, "nice": 1.5, "happy": 2.5, "helpful": 2, "friendly": 2,
       "fast": 1.5, "quick": 1.5, "easy": 1.5, "perfect": 3, "best": 3, "wonderful": 3,
       "smooth": 1.5, "reliable": 2, "recommend": 2, "recommended": 2, "pleased": 2,
       "satisfied": 2, "impressed": 2.5, "brilliant": 3, "comfortable": 1.5, "clean": 1.5,
       "value": 1.5, "delicious": 2.5, "beautiful": 2.5, "enjoy": 2, "enjoyed": 2,
       "fine": 0.8, "solid": 1.5, "superb": 3, "thanks": 1.5, "thank": 1.5,
       "professional": 2, "responsive": 2, "intuitive": 2, "worth": 2, "quality": 1.5,
       "outstanding": 3, "thankfully": 2, "thankful": 2, "grateful": 2,
       "positive": 1.2}
NEG = {"bad": -2, "terrible": -3, "awful": -3, "horrible": -3, "worst": -3, "hate": -3,
       "hated": -3, "poor": -2.5, "slow": -1.8, "broken": -2.5, "disappointed": -2.5,
       "disappointing": -2.5, "useless": -3, "rude": -2.5, "expensive": -1.5,
       "overpriced": -2, "buggy": -2, "crash": -2.5, "crashes": -2.5, "crashed": -2.5,
       "late": -1.5, "delay": -1.5, "delayed": -2, "dirty": -2, "noisy": -1.5,
       "waste": -2.5, "refund": -1.5, "faulty": -2.5, "defective": -2.5, "problem": -1.5,
       "problems": -1.5, "issue": -1.2, "issues": -1.2, "annoying": -2, "frustrating": -2.5,
       "frustrated": -2.5, "confusing": -1.8, "difficult": -1.5, "unreliable": -2.5,
       "cheap": -1, "fail": -2.5, "failed": -2.5, "never": -0.8, "complaint": -2,
       "scam": -3.5, "ignored": -2.2, "worse": -2.5, "unhappy": -2.5, "missing": -1.5,
       "wrong": -1.8, "lost": -1.8, "stuck": -1.5, "drains": -2.5, "slower": -1.8,
       "keeps": -0.3, "pain": -2.5, "painful": -2.5, "suffer": -2.5,
       "damaged": -2.5, "damage": -2, "pricey": -1, "skeptical": -1.2, "sceptical": -1.2,
       "suspicious": -1.5}
LEXICON = {**POS, **NEG}
NEGATORS = {"not", "no", "never", "isnt", "wasnt", "dont", "didnt", "cant", "wont",
            "doesnt", "hardly", "without", "havent", "hasnt", "hadnt", "wouldnt",
            "couldnt", "shouldnt", "arent", "werent"}
BOOSTERS = {"very": 1.3, "really": 1.3, "extremely": 1.5, "so": 1.2, "absolutely": 1.5,
            "totally": 1.3, "incredibly": 1.5, "slightly": 0.7, "somewhat": 0.8,
            "barely": 0.6, "severe": 1.4, "severely": 1.4, "overwhelmingly": 1.4,
            "bit": 0.8}

# Aspects: keyword prefixes (short keywords must match exactly) + suggested fixes
ASPECTS = {
    "Customer support": {
        "kw": ["support", "service", "agent", "helpdesk", "team", "staff", "reply",
               "replied", "replies", "answer", "waited"],
        "fix": ["Set a response-time target (e.g. under 24h) and track it",
                "Add auto-acknowledgement emails and a ticket status page",
                "Create a FAQ/chatbot for the most repeated questions"]},
    "Delivery": {
        "kw": ["deliver", "shipping", "shipped", "package", "packaging", "arrived",
               "courier", "parcel"],
        "fix": ["Show realistic delivery windows and live tracking",
                "Review courier performance and switch underperformers",
                "Proactively notify customers about delays"]},
    "Pricing & value": {
        "kw": ["price", "cost", "expensive", "overpriced", "value", "cheap", "money",
               "worth", "afford"],
        "fix": ["Benchmark prices against competitors",
                "Offer tiered plans, bundles or loyalty discounts",
                "Communicate value clearly (features, warranty, quality)"]},
    "Product quality": {
        "kw": ["product", "quality", "battery", "build", "material", "durab", "purchase"],
        "fix": ["Run root-cause analysis on defect and battery complaints",
                "Tighten supplier quality checks before dispatch",
                "Offer easy warranty replacement for faulty items"]},
    "App performance": {
        "kw": ["update", "loading", "load", "crash", "software", "bug", "bugs", "buggy",
               "slow", "freez", "glitch", "lag", "laggy"],
        "fix": ["Prioritise crash and load-time fixes in the next sprint",
                "Add crash reporting and performance monitoring",
                "Beta-test updates with a small user group before release"]},
    "Usability & design": {
        "kw": ["design", "interface", "easy", "intuitive", "navigation", "instruction",
               "confus", "layout", "setup"],
        "fix": ["Run quick usability tests on the confusing screens",
                "Add onboarding tips and clearer instructions",
                "Simplify navigation and reduce steps to key tasks"]},
    "Checkout & payment": {
        "kw": ["checkout", "payment", "pay", "billing", "card"],
        "fix": ["Audit payment failures and add more payment methods",
                "Allow guest checkout and reduce the number of steps",
                "Show clear error messages with a retry option"]},
    "Returns & refunds": {
        "kw": ["refund", "return"],
        "fix": ["Publish a simple, clear refund policy",
                "Automate refund approvals for low-value claims",
                "Give customers status updates during the refund process"]},
    "Subscription & billing": {
        "kw": ["subscri", "premium", "trial", "cancel", "renew", "billed", "charged",
               "charge", "membership", "paywall", "free", "plan", "annual", "monthly"],
        "fix": ["Label free vs premium content clearly and stop gating every tap behind a trial prompt",
                "Show price, trial length and renewal terms before sign-up",
                "Make cancellation and plan changes easy, and send renewal reminders"]},
    "Content & features": {
        "kw": ["content", "meditat", "library", "feature", "filter", "search", "music",
               "voice", "narrat", "stories", "story", "playlist", "selection", "variety",
               "option"],
        "fix": ["Add search and filters (length, voice, music, keyword)",
                "Offer previews so users can sample content before committing",
                "Use usage data and surveys to prioritise new content and features"]},
    "Marketing": {
        "kw": ["spam", "unsubscri", "newsletter", "promo", "promotion", "advert",
               "ad", "ads", "notific", "pitch", "sales", "upsell", "marketing"],
        "fix": ["Cap email and notification frequency and let users choose topics",
                "Honour unsubscribe requests immediately",
                "Keep promotions out of core use and onboarding"]},
}

EMOTIONS = {
    "Joy": ["love", "happy", "great", "excellent", "amazing", "fantastic", "awesome",
            "perfect", "best", "wonderful", "enjoy", "pleased", "brilliant", "superb",
            "delight", "thank", "grateful", "worth"],
    "Trust": ["reliab", "recommend", "professional", "trust", "solid", "helpful",
              "friendly", "responsive", "honest", "quality", "comfortable", "intuitive"],
    "Anger": ["hate", "terrible", "awful", "horrible", "rude", "scam", "worst",
              "ridiculous", "ignored", "furious"],
    "Frustration": ["frustrat", "annoy", "confus", "difficult", "stuck", "crash", "slow",
                    "buggy", "fail", "failure", "waited", "delay", "late"],
    "Disappointment": ["disappoint", "poor", "waste", "unhappy", "overpriced", "expensive",
                       "defective", "faulty", "worse", "drains"],
    "Worry": ["worri", "worry", "concern", "afraid", "unsafe", "lost", "missing", "risk", "skeptic", "sceptic", "suspici"],
}

SAMPLE = """The app is fantastic, really easy to use and super fast!
Terrible customer service. I waited three days and nobody replied.
Delivery was late but the product quality is excellent.
It's okay. Does the job, nothing special.
I love the new design, very intuitive and clean.
The app keeps crashing and it is frustrating. Waste of money.
Support team was friendly and helpful, solved my problem quickly.
Not good. The battery drains very fast and the price is too expensive.
Great value for the price, would recommend to friends.
Checkout is confusing and the payment failed twice.
Absolutely perfect, best purchase I have made this year!
Package arrived on Tuesday.
Slow loading times and buggy updates, really disappointed.
Staff were professional and responsive, thank you!
Not bad at all, actually quite comfortable and reliable.
The refund process was difficult and nobody answered my emails.
Smooth setup, excellent instructions, happy with everything.
Overpriced for what you get, and the quality is poor.
Delivery was quick and the packaging was clean.
I hate the new update, everything is slower and confusing.""".splitlines()


# --------------------------------------------------------------------------
# 2. Scoring functions
# --------------------------------------------------------------------------
# Words that introduce a thought or fear, not an experience ("I thought it was a scam")
COGNITIVE = {"thought", "think", "thinking", "suspected", "assumed", "expected", "feared",
             "worried", "skeptical", "sceptical", "doubted", "figured", "imagined", "afraid"}
HYPOTHETICAL = {"if", "unless"}          # "if it isn't right, I'll suffer" is not an event
NEG_EMOTIONS = {"Anger", "Frustration", "Disappointment", "Worry"}
SUFFIXES = ("s", "es", "ed", "ing", "ly", "er", "d")
# Explicit verdicts: reviewers often state their conclusion directly
NEG_VERDICTS = ["not worth", "wasnt worth", "isnt worth", "waste of money", "do not buy",
                "dont buy", "would not recommend", "wouldnt recommend", "never again",
                "want a refund"]
POS_VERDICTS = ["well worth", "worth it", "worth every", "highly recommend",
                "would recommend", "love it", "game changer", "best purchase"]


def normalise(text):
    return text.lower().replace("\u2019", "'").replace("n't", "nt")


def tokenize(text):
    return re.sub(r"[^a-z\s]", " ", normalise(text)).split()


def explicit_signal(text):
    """Star ratings ('5/5', '4 stars') and verdict phrases -> value in [-1, 1], or None."""
    low = normalise(text)
    m = re.search(r"\b(\d(?:\.\d)?)\s*/\s*(5|10)\b", low)
    if m:
        r = float(m.group(1)) / float(m.group(2)) * 5
    else:
        m = re.search(r"\b(\d(?:\.\d)?)\s*(?:out of 5\s*)?stars?\b|\b(\d(?:\.\d)?) out of 5\b", low)
        r = float(m.group(1) or m.group(2)) if m else None
    if r is not None and 0 <= r <= 5:
        return max(-1.0, min(1.0, (r - 3) / 2))
    if any(p in low for p in NEG_VERDICTS):
        return -1.0
    if any(p in low for p in POS_VERDICTS):
        return 1.0
    return None


def walk(text):
    """Yield (word, previous_words, sentence_weight, context_weight) for each token.
    - Later sentences weigh more (reviews usually end with the verdict).
    - Words after 'if/unless' (whole sentence) are discounted: hypothetical.
    - Words after 'thought/expected/skeptical...' (same clause) are discounted."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()] or [text]
    n = len(sentences)
    for si, sent in enumerate(sentences):
        sw = 1 + (si / (n - 1) if n > 1 else 0)
        hypo = False
        for seg in re.split(r"[;,]", sent):
            tokens, cog = tokenize(seg), False
            for i, w in enumerate(tokens):
                ctx = (0.2 if hypo else 1.0) * (0.3 if cog else 1.0)
                yield w, tokens[max(0, i - 3):i], sw, ctx
                cog = cog or w in COGNITIVE
                hypo = hypo or w in HYPOTHETICAL


def score(text):
    """Return (compound score in [-1, 1], list of (word, valence) hits)."""
    total, hits = 0.0, []
    for word, window, sw, ctx in walk(text):
        if word not in LEXICON:
            continue
        value = LEXICON[word]
        for prev in window:
            value *= BOOSTERS.get(prev, 1)       # intensifiers scale the word
        if any(p in NEGATORS for p in window):
            value *= -0.75                       # negation flips and softens it
        total += value * sw * ctx
        hits.append((word, value * ctx))
    if "!" in text:
        total *= 1.1
    compound = total / math.sqrt(total * total + 8) if hits else 0.0
    signal = explicit_signal(text)
    if signal is not None:                       # blend explicit rating/verdict
        compound = 0.65 * signal + 0.35 * compound
    return max(-1.0, min(1.0, compound)), hits


def label(c):
    return "Positive" if c >= 0.25 else "Negative" if c <= -0.25 else "Neutral"


def emotion_match(word, stem):
    return match(word, stem)


def detect_emotion(text, sentiment=None):
    counts = Counter()
    for word, window, _, ctx in walk(text):
        negated = any(p in NEGATORS for p in window)
        for emo, stems in EMOTIONS.items():
            if any(emotion_match(word, s) for s in stems):
                if negated:
                    if emo in ("Joy", "Trust"):
                        counts["Disappointment"] += 1   # "not happy"
                elif not (ctx < 0.5 and emo in NEG_EMOTIONS):
                    counts[emo] += 1
    if sentiment == "Positive":                  # keep emotion consistent with sentiment
        counts = Counter({e: c for e, c in counts.items() if e not in NEG_EMOTIONS})
    elif sentiment == "Negative":
        counts = Counter({e: c for e, c in counts.items() if e in NEG_EMOTIONS}) or counts
    return counts.most_common(1)[0][0] if counts else "Neutral"


CLAUSE_SPLIT = re.compile(
    r"[.;!?,]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhile\b|\byet\b", re.I)


def match(word, kw):
    """Keywords of 5+ letters match as prefixes; shorter ones match exactly (plus common
    endings), so 'free' does not match 'freeze' and 'late' does not match 'later'."""
    if len(kw) >= 5:
        return word.startswith(kw)
    return word == kw or any(word == kw + s for s in SUFFIXES)


def score_aspects(text):
    """Score each clause separately and credit it to the aspects it mentions."""
    result = defaultdict(list)
    for clause in CLAUSE_SPLIT.split(text):
        tokens = tokenize(clause)
        if not tokens:
            continue
        c, _ = score(clause)
        for aspect, info in ASPECTS.items():
            if any(match(w, k) for w in tokens for k in info["kw"]):
                result[aspect].append(c)
    return dict(result)


# --------------------------------------------------------------------------
# 3. Analysis, insights and reporting
# --------------------------------------------------------------------------
def analyse(texts):
    rows = []
    for t in texts:
        c, hits = score(t)
        rows.append({"text": t, "score": round(c, 3), "label": label(c),
                     "emotion": detect_emotion(t, label(c)), "aspects": score_aspects(t),
                     "hits": hits})
    return rows


def aspect_stats(rows):
    stats = {}
    for r in rows:
        for aspect, scores in r["aspects"].items():
            s = stats.setdefault(aspect, {"n": 0, "pos": 0, "neu": 0, "neg": 0, "sum": 0.0})
            for c in scores:
                s["n"] += 1
                s["sum"] += c
                s["pos" if c >= 0.25 else "neg" if c <= -0.25 else "neu"] += 1
    for s in stats.values():
        s["avg"] = s["sum"] / s["n"]
    return stats


def priority(s):
    share = s["neg"] / s["n"]
    if share >= 0.5 and s["neg"] >= 2:
        return "High"
    return "Medium" if share >= 0.25 else "Low"


def print_report(rows):
    n = len(rows)
    labels = Counter(r["label"] for r in rows)
    avg = sum(r["score"] for r in rows) / n
    stats = aspect_stats(rows)
    emotions = Counter(r["emotion"] for r in rows if r["emotion"] != "Neutral")

    print("=" * 62)
    print(f"SENTIMENT REPORT  ({n} entries)")
    print("=" * 62)
    print(f"Average score: {avg:+.2f}   "
          + "  ".join(f"{k}: {labels[k]} ({labels[k] / n:.0%})"
                      for k in ("Positive", "Neutral", "Negative")))

    print("\nSentiment by aspect (mentions, avg score, negatives)")
    for a, s in sorted(stats.items(), key=lambda x: -x[1]["n"]):
        print(f"  {a:<20} {s['n']:>3} mentions  {s['avg']:+.2f}  {s['neg']} neg")

    print("\nEmotions: " + (", ".join(f"{e} {c}" for e, c in emotions.most_common())
                            or "none detected"))

    print("\nINSIGHTS")
    mood = ("clearly positive" if avg >= .25 else "clearly negative" if avg <= -.25
            else "mildly positive" if avg > .05 else "mildly negative" if avg < -.05
            else "mixed")
    print(f"  - Overall sentiment is {mood}.")
    problems = [(a, s) for a, s in stats.items() if s["neg"] and s["avg"] < 0]
    problems.sort(key=lambda x: x[1]["neg"] * -x[1]["avg"], reverse=True)
    if problems:
        a, s = problems[0]
        print(f"  - Weakest area: {a} ({s['neg']} of {s['n']} mentions negative).")
    strengths = [(a, s) for a, s in stats.items() if s["avg"] >= .25 and s["n"] >= 2]
    for a, s in strengths[:2]:
        print(f"  - Strength to protect: {a} (avg {s['avg']:+.2f}).")
    if emotions:
        print(f"  - Leading emotion: {emotions.most_common(1)[0][0]}.")

    print("\nSUGGESTED SOLUTIONS")
    for a, s in problems[:4]:
        print(f"  [{priority(s)} priority] {a}")
        for fix in ASPECTS[a]["fix"]:
            print(f"      * {fix}")
    if not problems:
        print("  No problem areas found.")
    return stats


def validate(rows, truth):
    """Compare predicted labels with human labels and print accuracy."""
    pairs = [(r["label"], t.strip().capitalize()) for r, t in zip(rows, truth) if t.strip()]
    if not pairs:
        return
    correct = sum(p == t for p, t in pairs)
    print(f"\nVALIDATION: {correct}/{len(pairs)} correct ({correct / len(pairs):.0%})")
    for cls in ("Positive", "Neutral", "Negative"):
        tp = sum(p == t == cls for p, t in pairs)
        pred = sum(p == cls for p, _ in pairs)
        act = sum(t == cls for _, t in pairs)
        prec = tp / pred if pred else 0
        rec = tp / act if act else 0
        print(f"  {cls:<9} precision {prec:.2f}  recall {rec:.2f}")


def save_outputs(rows, stats, prefix):
    with open(f"{prefix}_entries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "score", "label", "emotion", "aspects"])
        for r in rows:
            w.writerow([r["text"], r["score"], r["label"], r["emotion"],
                        "; ".join(r["aspects"])])
    with open(f"{prefix}_aspects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aspect", "mentions", "avg_score", "positive", "neutral", "negative",
                    "priority"])
        for a, s in sorted(stats.items(), key=lambda x: -x[1]["n"]):
            w.writerow([a, s["n"], round(s["avg"], 3), s["pos"], s["neu"], s["neg"],
                        priority(s)])
    print(f"\nSaved {prefix}_entries.csv and {prefix}_aspects.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = sorted(stats, key=lambda a: stats[a]["avg"])
        vals = [stats[a]["avg"] for a in names]
        plt.figure(figsize=(8, 4))
        plt.barh(names, vals, color=["#d4503f" if v < 0 else "#2a9d6f" for v in vals])
        plt.axvline(0, color="grey", lw=0.8)
        plt.title("Average sentiment by aspect")
        plt.tight_layout()
        plt.savefig(f"{prefix}_aspects.png", dpi=150)
        print(f"Saved {prefix}_aspects.png")
    except ImportError:
        print("(Install matplotlib to also get a chart: pip install matplotlib)")


def load_csv(path, column):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        col = column or next((h for h in fields if re.search(
            r"text|review|comment|feedback|content|message|body", h, re.I)), fields[0])
        data = list(reader)
    texts = [r[col].strip() for r in data if r.get(col, "").strip()]
    truth = [r.get("label", "") for r in data if r.get(col, "").strip()]
    print(f"Using column '{col}' ({len(texts)} rows)")
    return texts, truth


def main():
    ap = argparse.ArgumentParser(description="Aspect-based sentiment & emotion tool")
    ap.add_argument("csv", nargs="?", help="CSV file with a text column")
    ap.add_argument("--column", help="name of the text column")
    ap.add_argument("--out", help="prefix for saved output files")
    ap.add_argument("--sample", action="store_true", help="run on built-in sample reviews")
    args = ap.parse_args()
    if args.sample:
        texts, truth = SAMPLE, []
    elif args.csv:
        texts, truth = load_csv(args.csv, args.column)
    else:
        ap.error("provide a CSV file or use --sample")
    rows = analyse(texts)
    stats = print_report(rows)
    if truth:
        validate(rows, truth)
    if args.out:
        save_outputs(rows, stats, args.out)


if __name__ == "__main__":
    main()
