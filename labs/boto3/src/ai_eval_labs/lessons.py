"""Scenario adapters and readable Python equivalents of the original assertions."""
from .core import DATA
from .metrics import check, contains


def content_assertions(output, assertions):
    checks = []
    for assertion in assertions:
        kind, value = assertion["type"], assertion.get("value")
        if kind in ("contains", "icontains"):
            score = contains(output, value, kind == "contains")
        elif kind in ("contains-any", "icontains-any"):
            score = float(any(contains(output, text, kind == "contains-any") for text in value))
        else:
            continue
        checks.append(check(f"{kind}: {value}", score))
    return checks


def evaluate_riddle(result, case, judge, *, input_rate=None, output_rate=None):
    if result.errors:
        return result
    result.assertions += content_assertions(result.response, case["assertions"])
    usage_start = len(judge.api.usage)
    for assertion in case["assertions"]:
        if assertion["type"] == "llm-rubric":
            try:
                result.assertions.append(judge.rubric(result.response, assertion["value"]))
            except Exception as exc:
                result.errors.append(f"Rubric: {type(exc).__name__}: {exc}")
    result.grading_usage += judge.api.usage[usage_start:]
    result.estimate_cost(input_rate, output_rate)
    # Synthetic timings and missing prices are unavailable, never measured passes.
    for name, available, value, limit in (
        ("latency-seconds", result.source == "live", result.elapsed_seconds, 60),
        ("response-cost-usd", result.estimated_cost_usd is not None, result.estimated_cost_usd, 0.15),
    ):
        result.assertions.append({"name": name, "value": value if available else None,
                                  "maximum": limit, "passed": value <= limit if available else None,
                                  "reason": "Measured/estimated after completion" if available else "Unavailable"})
    return result


def evaluate_rag(result, case, judge, synthetic):
    if result.errors:
        return result
    context = (DATA / "policies" / case["context"].split("/")[-1]).read_text()
    usage_start = len(judge.api.usage)
    result.assertions += content_assertions(result.response, case["assertions"])
    for assertion in case["assertions"]:
        metric = assertion["type"]
        if metric == "contains":
            continue
        try:
            if metric == "answer-relevance":
                verdict = judge.answer_relevance(case["query"], result.response)
            else:
                evidence, material = {
                    "factuality": ({"answer": result.response}, {"reference": assertion.get("value")}),
                    "context-recall": ({"context": context}, {"reference": assertion.get("value")}),
                    "context-relevance": ({"query": case["query"]}, {"context": context}),
                    "context-faithfulness": ({"context": context}, {"answer": result.response}),
                }[metric]
                verdict = judge.statements(metric, evidence, material, synthetic[metric])
            result.assertions.append(verdict)
        except Exception as exc:
            result.errors.append(f"{metric}: {type(exc).__name__}: {exc}")
    result.grading_usage += judge.api.usage[usage_start:]
    return result


def evaluate_agent(result, case, case_index, judge):
    from .agent import verify_tools
    result.assertions.append(verify_tools(result, case_index))
    if result.errors:
        return result
    output = result.response
    lower = output.casefold()
    result.assertions += content_assertions(output, case["assertions"])
    # Python equivalents of the original final-answer JavaScript checks.
    predicates = {
        0: ("roll" in lower or "d20" in lower) and ("attack" in lower or "hit" in lower),
        1: "hp" in lower or "hit points" in lower,
        2: "gold" in output or "gp" in output,
        3: len(output) > 100 and any(term in lower for term in ("dark", "torch", "shadows")),
        4: "save" in lower or "saving throw" in lower,
    }
    if case_index in predicates:
        result.assertions.append(check("answer-content", float(predicates[case_index])))
    usage_start = len(judge.api.usage)
    for assertion in case["assertions"]:
        if assertion["type"] == "llm-rubric":
            try:
                result.assertions.append(judge.rubric(output, assertion["value"]))
            except Exception as exc:
                result.errors.append(f"Rubric: {type(exc).__name__}: {exc}")
    result.grading_usage += judge.api.usage[usage_start:]
    return result
