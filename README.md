# EchoCache

Challenge 2: Deliver AI with Less Compute

## The problem

Customer-support chatbots send every question to a large AI model, even when the same question was answered minutes ago in different words. Each answer costs about 1,500 tokens of AI work in our tests (about $0.0006 at Google's paid prices, or $570 per million answers).

A *cache* (an answer memory) can reuse earlier answers for similar questions and skip the AI. But similar wording is not the same meaning: "How do I turn **on** Apple Pay?" and "How do I turn **off** Apple Pay?" look almost identical to a basic cache, so it reuses the wrong answer.

## The solution

EchoCache is an answer cache with three cheap checks before any reuse:

1. **Meaning-flip check:** blocks reuse when one small word changes the meaning (on / off, £100 / £500, Spain / Brazil).
2. **Stricter on risky topics:** a wrong answer about a stolen card matters more than one about delivery times, so risky topics need a much closer match (similarity of 0.92 instead of 0.80).
3. **Privacy protected:** questions about a customer's own account, such as their balance, are always answered fresh and never stored.

If any check fails, the AI writes a fresh answer. Every decision is printed with a plain-English reason.

The checks are simple rules rather than AI, because asking an AI "is this safe to reuse?" for every question would use up the computing power the cache saves.

## How it works

For each question:

1. **Private?** Questions about the customer's own account skip the memory.
2. **Find similar:** a small free model on your computer finds the closest stored question and scores it from 0 (unrelated) to 1 (same meaning).
3. **Risk level:** the score must reach 0.80, 0.86 or 0.92 depending on the topic's risk.
4. **Meaning check:** negation, opposite words, numbers and place names must match.
5. **Reuse or ask the AI:** reuse only if steps 3 and 4 pass.
6. **Store safely:** keep new answers only if they come from the FAQ.

**Models**

- Matching model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (sentence-transformers), free, runs locally.
- Answering model: Google Gemini 3.5 Flash-Lite, which answers only from the 19 entries in `data/faq.json`, so it cannot invent facts about the fictional bank.

## Files

| File | What it is |
|---|---|
| `echocache.py` | The program and a live demo. `--basic` switches the three checks off, giving a basic cache to compare against |
| `test_echocache.py` | Automatic test: runs 47 prepared question pairs through both versions and counts right and wrong reuses |
| `data/faq.json` | The fictional bank's (NovaBank) 19 official answers, each marked low, medium or high risk |
| `data/tests/tricky.json` | 47 test question pairs, each with the correct decision written down in advance |
| `requirements.txt` | Python packages needed |

## How to run

You need Python 3.10 or newer and a free Gemini API key from [Google AI Studio](https://aistudio.google.com).

```
python -m venv venv
venv\Scripts\activate            (Windows)   or   source venv/bin/activate   (Mac/Linux)
python -m pip install -r requirements.txt
```

Create a file called `.env` in the project folder containing:

```
GEMINI_API_KEY=your-key-here
```

Then:

```
python echocache.py              # EchoCache: type questions, see each decision and its reason
python echocache.py --basic      # the basic cache, for comparison
python test_echocache.py         # the 47-pair test (about 8 minutes, about 90 AI calls)
```

Run every command from the project folder. The first run downloads the matching model (about 90 MB).

