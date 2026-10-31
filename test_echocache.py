"""Compare the basic cache with EchoCache on the tricky question pairs.

    python test_echocache.py

For each pair, the first question is asked so its answer gets stored, then the second
question is asked. We record whether the stored answer was reused, and whether reusing
it was the right thing to do (data/tests/tricky.json says which).
"""
import json

from echocache import EchoCache

TESTS = json.load(open("data/tests/tricky.json", encoding="utf-8"))
GROUPS = [g for g in TESTS if not g.startswith("_")]

results = {}
for name, safe in [("Basic cache", False), ("EchoCache", True)]:
    bot = EchoCache(safe=safe)
    rows = {}
    for group in GROUPS:
        good = wrong = missed = 0
        for pair in TESTS[group]:
            bot.cache.clear()                              # every pair starts with an empty cache
            bot.ask(pair["seed"])
            reused = bot.ask(pair["probe"])["decision"] == "REUSED"
            if reused and pair["should_reuse"]:
                good += 1
            elif reused:
                wrong += 1
                print(f"  [{name}] wrong reuse: \"{pair['seed']}\" -> \"{pair['probe']}\"")
            elif pair["should_reuse"]:
                missed += 1
        rows[group] = (len(TESTS[group]), good, wrong, missed)
    results[name] = (rows, bot)

print(f"\n{'Group':12}{'Pairs':>6} | {'Basic: saved / WRONG / missed':>30} | {'EchoCache: saved / WRONG / missed':>34}")
for group in GROUPS:
    n, *b = results["Basic cache"][0][group]
    _, *e = results["EchoCache"][0][group]
    print(f"{group:12}{n:>6} | {b[0]:>14} / {b[1]:>5} / {b[2]:>6} | {e[0]:>18} / {e[1]:>5} / {e[2]:>6}")
for name, (rows, bot) in results.items():
    wrong = sum(r[2] for r in rows.values())
    saved = sum(r[1] for r in rows.values())
    per_call = bot.tokens / bot.calls                      # average tokens one LLM call processes
    no_cache = bot.questions * per_call                   # if every question went to the LLM
    avoided = no_cache - bot.tokens
    print(f"\n{name}: {saved} correct reuses, {wrong} wrong reuses")
    print(f"  LLM calls: {bot.calls} of {bot.questions} questions   "
          f"LLM tokens: {bot.tokens:,}   avoided vs no cache: ~{avoided:,.0f} ({avoided / no_cache:.0%})   "
          f"business cost: ${bot.cost:.4f}")
print("\nsaved = reused correctly (no Gemini call), WRONG = reused when it should not, "
      "missed = could have reused but called Gemini")
