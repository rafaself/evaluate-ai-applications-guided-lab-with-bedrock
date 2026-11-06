import csv
import io
import json
from copy import deepcopy

import pytest

from ai_eval_labs.core import Bedrock, DATA, Result, export_results, generate, load_cases
from ai_eval_labs.metrics import Judge, contains, cosine, ratio
from ai_eval_labs.lessons import evaluate_agent, evaluate_rag, evaluate_riddle
from ai_eval_labs.agent import GameTools, ScriptedAgent, run_agent, verify_tools


@pytest.fixture
def synthetic():
    return json.loads((DATA / "synthetic.json").read_text())


def test_scenario_preservation():
    assert [len(load_cases(n)) for n in ("translations", "riddles", "rag", "agents")] == [2, 15, 2, 9]
    cases = load_cases("riddles")
    assert "breath" == cases[10]["assertions"][0]["value"] == cases[14]["assertions"][0]["value"]
    assert "Talk to Fred" in load_cases("rag")[0]["assertions"][3]["value"]
    assert "22 months" in load_cases("rag")[1]["assertions"][2]["value"]


def test_offline_guard(monkeypatch):
    import boto3
    def forbidden(*args, **kwargs):
        pytest.fail("Offline evaluation created an AWS client")
    monkeypatch.setattr(boto3, "client", forbidden)
    api = Bedrock()
    assert generate(api, "test", "ignored", "fixture").response == "fixture"
    with pytest.raises(RuntimeError, match="disabled"):
        api.text("hello")
    with pytest.raises(RuntimeError, match="disabled"):
        api.embed("hello")


def test_deterministic_metrics():
    assert contains("Bonjour", "bonjour") == 0
    assert contains("Bonjour", "bonjour", False) == 1
    assert contains("Straße", "STRASSE", False) == 1
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([1, 2], [1, 2]) == pytest.approx(1)
    assert ratio([True, False, True]) == pytest.approx(2/3)
    for a, b in [([], []), ([0], [1]), ([1], [1, 2]), ([float("nan")], [1])]:
        with pytest.raises(ValueError):
            cosine(a, b)
    for values in ([], [1], ["yes"]):
        with pytest.raises(ValueError):
            ratio(values)


class TextApi:
    live = True
    judge_id = "judge"
    def __init__(self, text):
        self.response = text
    def text(self, *args):
        return self.response, {}, 0


@pytest.mark.parametrize("text", ['bad json', '[]', '{"passed": "true", "reason": "x"}', '{"passed": true}'])
def test_malformed_rubric(text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        Judge(TextApi(text)).rubric("answer", "rubric")


@pytest.mark.parametrize("text", ['{"items": []}', '{"items": [{"statement":"x", "supported":1,"reason":"x"}]}', '{"items": [null]}'])
def test_malformed_statement_grade(text):
    with pytest.raises(ValueError):
        Judge(TextApi(text)).statements("context-recall", {}, {}, [])


@pytest.mark.parametrize("text", ['{"questions": []}', '{"questions": ["x", "y", 3]}'])
def test_malformed_questions(text):
    with pytest.raises(ValueError):
        Judge(TextApi(text)).answer_relevance("q", "a")


def test_exports_and_cost(tmp_path):
    r = Result("case", "model", response='comma, newline\nquoted "text"', source="live",
               token_usage={"inputTokens": 100, "outputTokens": 50},
               assertions=[{"passed": False}], errors=["error"], tool_calls=[{"name": "tool"}])
    assert r.estimate_cost() is None
    assert r.estimate_cost(2, 4) == pytest.approx(0.0004)
    export_results([r], tmp_path / "nested/report")
    data = json.loads((tmp_path / "nested/report.json").read_text())
    assert data[0]["response"] == r.response
    with (tmp_path / "nested/report.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["response"] == r.response
    assert json.loads(row["errors"]) == ["error"]
    assert json.loads(row["tool_calls"]) == r.tool_calls
    assert Result("case", "model").estimate_cost(2, 4) is None


def test_rag_interpretation(synthetic):
    api = Bedrock()
    for case, fixture in zip(load_cases("rag"), synthetic["rag"]):
        result = generate(api, case["id"], "unused", fixture["answer"])
        evaluate_rag(result, case, Judge(api), fixture["grades"])
        by_name = {a["name"]: a for a in result.assertions}
        assert by_name["context-recall"]["score"] == 0.5
        assert by_name["context-faithfulness"]["score"] == 1
        assert not by_name["context-relevance"]["passed"]
        assert not result.errors


def test_riddle_all_cases(synthetic):
    api = Bedrock()
    for case, answer in zip(load_cases("riddles"), synthetic["riddles"]):
        result = generate(api, case["id"], "unused", answer)
        evaluate_riddle(result, case, Judge(api))
        assert not result.errors
        assert all(a["passed"] is not False for a in result.assertions)
        assert result.assertions[-1]["passed"] is None


def test_live_transport_and_embeddings():
    class Client:
        def converse(self, **kwargs):
            assert kwargs["modelId"] == "response-model"
            return {"output": {"message": {"content": [{"text": "hello"}, {"text": "world"}]}},
                    "usage": {"inputTokens": 10, "outputTokens": 2}}
        def invoke_model(self, **kwargs):
            assert json.loads(kwargs["body"]) == {"inputText": "question"}
            return {"body": io.BytesIO(b'{"embedding":[1,0],"inputTextTokenCount":1}')}
    api = Bedrock(live=True, client=Client())
    result = generate(api, "case", "prompt", "unused", "response-model")
    assert result.response == "hello\nworld"
    assert result.token_usage["inputTokens"] == 10
    assert api.embed("question") == [1, 0]
    assert len(api.usage) == 2


def test_grader_error_recorded(synthetic):
    api = Bedrock()
    judge = Judge(api)
    judge.rubric = lambda *args: (_ for _ in ()).throw(ValueError("malformed"))
    case = load_cases("riddles")[0]
    result = Result(case["id"], "model", response=synthetic["riddles"][0])
    evaluate_riddle(result, case, judge)
    assert "malformed" in result.errors[0]


def test_game_tools_validation_and_isolation():
    first = GameTools(seed=7)
    second = GameTools(seed=7)
    assert first.execute("roll_dice", {"sides":20}) == second.execute("roll_dice", {"sides":20})
    inv = first.execute("check_inventory", {})
    inv["gold"] = 0
    assert first.execute("check_inventory", {})["gold"] == 47
    first.state["character"]["name"] = "changed"
    assert second.execute("check_character_stats", {})["name"] == "Thorin Ironforge"
    assert "description" in first.execute("describe_scene", {"location":"crypt", "mood":"ominous"})
    for name, args in [("unknown", {}), ("roll_dice", {"sides":True}),
                       ("roll_dice", {"sides":20,"count":0}), ("roll_dice", {"sides":20,"count":101}),
                       ("roll_dice", {"sides":20,"shell":"x"}), ("roll_dice", {}),
                       ("check_inventory", {"playerId":"player2"}), ("describe_scene", {"location":"crypt"})]:
        with pytest.raises(ValueError):
            first.execute(name, args)


def test_all_agent_scenarios(synthetic):
    for i, case in enumerate(load_cases("agents")):
        transport = ScriptedAgent(i, synthetic["agents"][i])
        result = run_agent(transport, case["id"], case["query"], seed=42)
        evaluate_agent(result, case, i, Judge(Bedrock()))
        assert not result.errors
        assert all(a["passed"] for a in result.assertions)
        if i != 7:
            sent = transport.requests[-1]["messages"][-1]["content"][0]["toolResult"]
            assert sent["toolUseId"] == result.tool_calls[0]["call_id"]
            assert sent["content"][0]["json"] == result.tool_calls[0]["result"]
    fake = Result("case", "model", response="I called roll_dice sides 20 count 1")
    assert not verify_tools(fake, 0)["passed"]
    fake.tool_calls = [{"name":"roll_dice", "arguments":{"sides":20,"count":1}}]
    assert not verify_tools(fake, 7)["passed"]


def message(calls=None, reason=None):
    return {"output": {"message": {"role":"assistant", "content":
            [{"toolUse":c} for c in calls] if calls else [{"text":"done"}]}},
            "stopReason": reason or ("tool_use" if calls else "end_turn"),
            "usage": {"inputTokens": 2, "outputTokens": 1}}


class Sequence:
    live = False
    model_id = "scripted"
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def converse(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.responses)


def test_multiple_calls_and_errors():
    calls = [
        {"toolUseId":"a", "name":"check_inventory", "input":{"playerId":"player1"}},
        {"toolUseId":"b", "name":"check_character_stats", "input":{"playerId":"player1"}},
        {"toolUseId":"c", "name":"not_registered", "input":{}},
        {"toolUseId":"d", "name":"roll_dice", "input":{"sides":20,"count":0}},
    ]
    transport = Sequence([message(calls), message()])
    result = run_agent(transport, "case", "query")
    assert [c["call_id"] for c in result.tool_calls] == ["a", "b"]
    outputs = transport.requests[-1]["messages"][-1]["content"]
    assert [o["toolResult"]["toolUseId"] for o in outputs] == ["a","b","c","d"]
    assert [o["toolResult"]["status"] for o in outputs] == ["success","success","error","error"]
    assert len(result.errors) == 2
    assert result.token_usage == {"inputTokens":4,"outputTokens":2}


def test_turn_exhaustion_duplicate_ids_and_truncation():
    call = {"toolUseId":"a","name":"roll_dice","input":{"sides":20,"count":1}}
    result = run_agent(Sequence([message([call])]), "case", "query", max_turns=1)
    assert "exceeded 1 turns" in result.errors[0]
    assert result.response == ""
    result = run_agent(Sequence([message([call]), message([call])]), "case", "query")
    assert "duplicate" in result.errors[0]
    result = run_agent(Sequence([message(reason="max_tokens")]), "case", "query")
    assert "max_tokens" in result.errors[0]


def test_effective_tool_defaults_are_verified():
    call = {"toolUseId": "a", "name": "roll_dice", "input": {"sides": 20}}
    result = run_agent(Sequence([message([call]), message()]), "case", "query")
    assert verify_tools(result, 0)["passed"]
    assert result.tool_calls[0]["arguments"]["count"] == 1


def test_answer_relevance_live_calculation():
    class Api(TextApi):
        def embed(self, text):
            return {"query": [1, 0], "a": [1, 0], "b": [0, 1], "c": [1, 0]}[text]
    grade = Judge(Api('{"questions": ["a", "b", "c"]}')).answer_relevance("query", "answer")
    assert grade["score"] == pytest.approx(2/3)
    assert not grade["passed"]
