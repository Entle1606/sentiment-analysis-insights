# Aspect-Based Sentiment & Emotion Analysis Tool

A tool that reads customer comments and tells you **how people feel, what they are talking about, which emotions they express, and what to fix first**. It comes as an interactive dashboard and a Python command-line script that share the same logic.

Individual project: AI-assisted data interpretation and insight generation.

## What it does

- **Sentiment scoring** from -1 (very negative) to +1 (very positive), using a word list with negation, intensifiers and context rules
- **Aspect-based sentiment:** scores each topic separately (support, pricing, subscription, marketing and others), so "great product, slow delivery" is not averaged into one number
- **Emotion detection:** Joy, Trust, Anger, Frustration, Disappointment, Worry
- **Insights and suggested solutions:** weakest and strongest areas, priority levels (High, Medium, Low) and three concrete fixes per problem area
- **AI-assisted interpretation:** the dashboard builds a prompt from your results to paste into Claude for root causes and an action plan
- **Self-validation:** add a `label` column of human labels to get accuracy, precision and recall

## Quick start

### Dashboard

Open `dashboard/sentiment_dashboard.html` in any browser (no install needed), then:

1. Click **Load sample reviews**, or paste comments, or upload a CSV.
2. Read the aspect chart, emotions, insights and suggested solutions.
3. Copy the prompt from "Ask an AI to go deeper" into Claude for a fuller analysis.

### Python script

Requires Python 3.8 or newer. No packages needed (matplotlib is optional, for a chart).

```bash
# Try it on the built-in sample reviews
python sentiment_tool.py --sample

# Analyse your own CSV
python sentiment_tool.py reviews.csv --column review

# Save results (entries CSV, aspects CSV and a chart)
python sentiment_tool.py reviews.csv --column review --out results
```

If `--column` is left out, the script looks for a column named like text, review, comment or feedback.

### Check accuracy

Add a `label` column to your CSV with your own labels (positive, neutral or negative) for 20 to 30 rows, then run the script as above. It prints accuracy, precision and recall.

## Example output

```
SENTIMENT REPORT  (20 entries)
Average score: +0.09   Positive: 10 (50%)  Neutral: 2 (10%)  Negative: 8 (40%)

INSIGHTS
  - Overall sentiment is mildly positive.
  - Weakest area: App performance (4 of 5 mentions negative).
  - Leading emotion: Joy.

SUGGESTED SOLUTIONS
  [High priority] App performance
      * Prioritise crash and load-time fixes in the next sprint
      ...
```

## How it works

1. **Tokenise** the text and walk through it sentence by sentence.
2. **Score** each lexicon word, adjusted by intensifiers ("very"), negation ("not good") and context.
3. **Context rules:** later sentences weigh more, hypotheticals ("if...") and reported thoughts ("I thought it was a scam") are discounted, and star ratings and verdict phrases ("5/5", "well worth it") are blended in.
4. **Split into clauses** and score each one separately to get sentiment per aspect.
5. **Detect emotions** from keyword stems, kept consistent with the sentiment label.
6. **Generate insights** with fixed rules, so results are repeatable and every score can be traced to its words.

Full details, including the improvement log and limitations, are in the [technical explanation](docs/technical_explanation.md).

## Results on a real dataset

Run on 149 customer comments from a meditation and sleep app:

| Measure | Result |
|---|---|
| Positive / Neutral / Negative | 46% / 39% / 15% |
| Average score | 0.26 |
| Leading emotion | Joy (60 comments) |
| Most negative aspects | Returns & refunds (4 of 8 mentions negative), Pricing & value (7 of 23) |
| Strongest aspect | Usability & design (+0.39) |

**Main finding:** customers like the product. The complaints concentrate on money (price, trial terms, refunds) and a few strong reactions to marketing and missing search filters.

Recommended actions: make pricing, trial terms and refunds clear and easy; give users control over marketing; and validate demand for search and filters before building them. See the [insights report](docs/insights_report.md) for the full analysis.

## Limitations

- Lexicon-based, so sarcasm is not understood
- Topic words such as "anxiety" can be read as negative sentiment
- Two aspects (Content & features, Subscription & billing) matched too many comments in the real dataset and need tighter keywords
- Several aspects have fewer than ten mentions, which is too few for firm conclusions
- Word lists are tuned for English product and app reviews
- Accuracy against human labels: **[add your result here]**

## Project structure

```
.
├── README.md
├── sentiment_tool.py                 # Python command-line version
├── dashboard/
│   └── sentiment_dashboard.html      # Interactive dashboard
└── docs/
    ├── insights_report.md            # Data insights report
    ├── technical_explanation.md      # Method, improvement log, limitations
    └── screenshots/                  # Dashboard screenshots
```

## Roadmap

- Tighten the aspect keyword lists and re-run
- Validate on a larger hand-labelled sample and tune the word lists
- Compare against a pretrained transformer model and Claude as a classifier
- Embedding-based aspect detection to recognise new vocabulary
- Sentiment trends over time or by app version

## Tech

Python 3 (standard library, optional matplotlib), JavaScript and HTML (single-file dashboard), Claude for AI-assisted interpretation.
