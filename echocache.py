"""EchoCache: reuse an AI answer only when it is safe to.

    python echocache.py           EchoCache (safe reuse)
    python echocache.py --basic   basic semantic cache, for comparison

Type questions and see, for each one, whether a stored answer was reused and why.
"""
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

MODEL = "gemini-3.5-flash-lite"
PRICE_IN, PRICE_OUT = 0.30, 2.50            # US$ per million tokens: what a business would pay (secondary measure)
BASIC_THRESHOLD = 0.85                      # basic cache: one threshold for every question
RISK_THRESHOLDS = {"low": 0.80, "medium": 0.86, "high": 0.92}   # EchoCache: stricter when mistakes cost more

NEGATIONS = {"not", "no", "never", "don't", "can't", "cannot", "won't", "didn't", "only", "stop", "cancel", "undo"}
PERSONAL = ["my balance", "my account number", "my sort code", "card ending", "my transactions", "my last",
            "my statement", "how much did i", "what did i spend", "my transfer of", "my refund for order",
            "when does my card expire", "expiry date on my card"]   # phrases that ask about the customer's own account
NOT_NAMES = {"novabank", "bank", "card", "pin", "atm", "app"}   # capitalised words that are not names
OPPOSITES = [("on", "off"), ("freeze", "unfreeze"), ("block", "unblock"), ("lost", "found"),
             ("activate", "deactivate"), ("twice", "once"), ("weekdays", "weekends")]

load_dotenv()
FAQ = json.load(open("data/faq.json", encoding="utf-8"))
RISK = {e["id"]: e["risk"] for e in FAQ}
FAQ_TEXT = "\n".join(f"[{e['id']}] {e['question']} {e['answer']}" for e in FAQ)
PROMPT = """You are NovaBank's support assistant. Answer in 1-3 sentences using ONLY these FAQ entries.
If none answers it, or it needs the customer's own account details, use faq_id "none" and suggest contacting Support.
Reply with JSON only: {{"faq_id": "<id of the entry used, or none>", "answer": "<reply>"}}

FAQ:
{faq}

Question: {question}"""

_embedder = None
_answers = {}       # answers already fetched this run, so tests don't pay twice for the same question


def guard(new_q, old_q):
    """Cheap check for differences that similarity scores miss. Returns a reason, or '' if none."""
    a, b = set(re.findall(r"[a-z0-9']+", new_q.lower())), set(re.findall(r"[a-z0-9']+", old_q.lower()))
    if (a & NEGATIONS) != (b & NEGATIONS):
        return "negation differs"
    if {w for w in a if w.isdigit()} != {w for w in b if w.isdigit()}:
        return "numbers differ"
    for x, y in OPPOSITES:
        if (x in a and y in b) or (y in a and x in b):
            return f"opposite words ({x}/{y})"
    names = lambda q: {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]+\b", q)[1:]} - NOT_NAMES
    if names(new_q) != names(old_q):
        return "names differ"
    return ""


def is_personal(question):
    q = question.lower().replace("\u2019", "'")
    return any(phrase in q for phrase in PERSONAL)


class EchoCache:
    def __init__(self, safe=True):
        global _embedder
        self.safe = safe
        _embedder = _embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.cache = []             # stored answers: question, vector, answer, faq_id
        self.questions = 0          # questions asked
        self.calls = 0              # Gemini calls made
        self.tokens = 0             # tokens the LLM processed: our measure of compute used
        self.cost = 0.0             # estimated bill for the business

    def ask(self, question):
        self.questions += 1
        if self.safe and is_personal(question):       # layer 1: cheap phrase check, before any reuse
            answer, faq_id = self.call_gemini(question)
            return {"decision": "NEW ANSWER", "answer": answer, "faq_id": faq_id,
                    "reason": "personal question: never reused or stored"}
        vec = _embedder.encode(question, normalize_embeddings=True)
        best = max(self.cache, key=lambda c: float(c["vector"] @ vec), default=None)
        reason = "cache is empty"
        if best:
            score = float(best["vector"] @ vec)
            if self.safe:
                risk = RISK[best["faq_id"]]
                limit, problem = RISK_THRESHOLDS[risk], guard(question, best["question"])
                topic = f" for that {risk}-risk topic"
            else:
                limit, problem, topic = BASIC_THRESHOLD, "", ""
            if score >= limit and not problem:
                return {"decision": "REUSED", "answer": best["answer"], "faq_id": best["faq_id"],
                        "reason": f"matches \"{best['question']}\" (similarity {score:.2f}, needs {limit})"}
            if score < limit:
                reason = f"no stored answer close enough (best: \"{best['question']}\" at {score:.2f}, needs {limit}{topic})"
            else:
                reason = f"close match \"{best['question']}\" ({score:.2f}) blocked by guard: {problem}"
        answer, faq_id = self.call_gemini(question)
        if faq_id in RISK:          # layer 2: only store answers backed by an FAQ entry
            self.cache.append({"question": question, "vector": vec, "answer": answer, "faq_id": faq_id})
        else:
            reason += "; not stored: no FAQ entry fits (personal or off-topic)"
        return {"decision": "NEW ANSWER", "answer": answer, "faq_id": faq_id, "reason": reason}

    def request(self, prompt):
        """Send one request to Gemini. If Google is busy (503) or we are going too fast (429),
        wait and try again instead of crashing."""
        for attempt in range(1, 6):
            time.sleep(4.5)         # free tier allows 15 calls a minute
            try:
                return self.client.models.generate_content(model=MODEL, contents=prompt)
            except (errors.ServerError, errors.ClientError) as err:
                if getattr(err, "code", None) not in (429, 500, 503) or attempt == 5:
                    raise           # a real problem (e.g. bad key), or still failing after 5 tries
                wait = 15 * attempt
                print(f"  [Gemini busy ({err.code}), waiting {wait}s then retrying, attempt {attempt + 1} of 5]")
                time.sleep(wait)

    def call_gemini(self, question):
        if question not in _answers:
            resp = self.request(PROMPT.format(faq=FAQ_TEXT, question=question))
            u = resp.usage_metadata
            in_tokens = u.prompt_token_count or 0
            out_tokens = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
            tokens = in_tokens + out_tokens
            cost = (in_tokens * PRICE_IN + out_tokens * PRICE_OUT) / 1_000_000
            found = re.search(r"\{.*\}", resp.text or "", re.DOTALL)
            data = json.loads(found.group(0)) if found else {}
            _answers[question] = (data.get("answer", resp.text), str(data.get("faq_id", "none")).strip("[] "), tokens, cost)
        answer, faq_id, tokens, cost = _answers[question]
        self.calls += 1
        self.tokens += tokens
        self.cost += cost
        return answer, faq_id


if __name__ == "__main__":
    bot = EchoCache(safe="--basic" not in sys.argv)
    print(f"{'Basic cache' if not bot.safe else 'EchoCache'} ready. Type a question, or 'quit'.\n")
    while (q := input("You: ").strip()).lower() != "quit":
        r = bot.ask(q)
        print(f"  {r['decision']}: {r['reason']}")
        print(f"  Answer: {r['answer']}")
        print(f"  So far: {bot.questions} questions, {bot.calls} LLM calls, "
              f"{bot.tokens:,} LLM tokens processed (business cost about ${bot.cost:.5f})\n")
