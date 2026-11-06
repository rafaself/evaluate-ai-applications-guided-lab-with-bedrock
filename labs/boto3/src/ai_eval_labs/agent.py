"""A visible, bounded Converse tool loop with completed-execution evidence."""
from copy import deepcopy
import json
import random
import time

from .core import DATA, Result

GAME = json.loads((DATA / "game.json").read_text())
INSTRUCTIONS = (DATA / "agent_instructions.txt").read_text()
SCHEMAS = {
    "roll_dice": {"sides": "integer", "count": "integer", "modifier": "integer", "purpose": "string"},
    "check_inventory": {"playerId": "string"},
    "check_character_stats": {"playerId": "string"},
    "describe_scene": {"location": "string", "mood": "string"},
}
REQUIRED = {"roll_dice": ["sides"], "check_inventory": [], "check_character_stats": [],
            "describe_scene": ["location", "mood"]}
TOOLS = [{"toolSpec": {"name": name, "description": {
    "roll_dice": "Roll dice for attacks, damage, saves, or ability checks.",
    "check_inventory": "Read the fixture character inventory, equipment, and currency.",
    "check_character_stats": "Read the fixture character abilities, HP, AC, and level.",
    "describe_scene": "Read an atmospheric location description and possible actions.",
}[name], "inputSchema": {"json": {"type": "object",
    "properties": {key: {"type": kind} for key, kind in fields.items()},
    "required": REQUIRED[name], "additionalProperties": False}}}}
    for name, fields in SCHEMAS.items()]


class GameTools:
    def __init__(self, seed=None):
        self.state = deepcopy(GAME)
        self.random = random.Random(seed)

    def execute(self, name, arguments):
        if name not in SCHEMAS:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        fields = SCHEMAS[name]
        if set(arguments) - fields.keys() or set(REQUIRED[name]) - arguments.keys():
            raise ValueError("Unexpected or missing tool arguments")
        for key, value in arguments.items():
            expected = int if fields[key] == "integer" else str
            if type(value) is not expected:
                raise ValueError(f"Invalid type for {key}")
        if name == "roll_dice":
            sides, count = arguments["sides"], arguments.get("count", 1)
            modifier = arguments.get("modifier", 0)
            if sides not in (4, 6, 8, 10, 12, 20, 100) or not 1 <= count <= 100 or abs(modifier) > 1000:
                raise ValueError("Dice outside supported bounds")
            rolls = [self.random.randint(1, sides) for _ in range(count)]
            critical = ""
            if sides == 20 and count == 1:
                critical = {20: " (Natural 20! Critical hit!)", 1: " (Natural 1! Critical failure!)"}.get(rolls[0], "")
            return {"rolls": rolls, "modifier": modifier, "total": sum(rolls) + modifier,
                    "notation": f"{count}d{sides}{modifier:+}",
                    "purpose": arguments.get("purpose") or "dice roll", "criticalInfo": critical,
                    "breakdown": f"{' + '.join(map(str, rolls))} {modifier:+} = {sum(rolls)+modifier}{critical}"}
        if name in ("check_inventory", "check_character_stats"):
            if arguments.get("playerId", "player1") != "player1":
                raise ValueError("Only fixture player1 exists")
            return deepcopy(self.state["inventory" if name == "check_inventory" else "character"])
        location, mood = arguments["location"], arguments["mood"]
        return {"location": location, "mood": mood,
                "description": self.state["scenes"].get(location, {}).get(mood, f"You find yourself in {location} with {mood} atmosphere. The details remain unclear."),
                "possibleActions": ["investigate the area", "search for hidden items or passages", "proceed cautiously", "take a short rest", "examine something specific"],
                "lightLevel": "dark (torch required)" if location in ("dungeon", "crypt") else "normal"}


def run_agent(api, case_id, query, *, max_turns=20, seed=None):
    if type(max_turns) is not int or not 1 <= max_turns <= 100:
        raise ValueError("max_turns must be an integer between 1 and 100")
    result = Result(case_id, api.model_id, source="live" if api.live else "synthetic")
    tools = GameTools(seed)
    messages = [{"role": "user", "content": [{"text": query}]}]
    seen_ids = set()
    started = time.perf_counter()
    try:
        for _ in range(max_turns):
            response = api.converse(modelId=api.model_id, messages=deepcopy(messages),
                                    system=[{"text": INSTRUCTIONS}], toolConfig={"tools": TOOLS},
                                    inferenceConfig={"maxTokens": 2048})
            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    result.token_usage[key] = result.token_usage.get(key, 0) + value
            message = response["output"]["message"]
            messages.append(message)
            calls = [b["toolUse"] for b in message["content"] if "toolUse" in b]
            if not calls:
                if response.get("stopReason") != "end_turn":
                    raise RuntimeError(f"Agent stopped without a final answer: {response.get('stopReason')}")
                result.response = "\n".join(b["text"] for b in message["content"] if "text" in b)
                return result
            if response.get("stopReason") != "tool_use":
                raise ValueError("Tool requests require tool_use stop reason")
            outputs = []
            for call in calls:
                call_id, name, arguments = call["toolUseId"], call["name"], call["input"]
                if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
                    raise ValueError("Missing or duplicate tool-use ID")
                seen_ids.add(call_id)
                try:
                    value = tools.execute(name, arguments)
                    defaults = {
                        "roll_dice": {"count": 1, "modifier": 0, "purpose": ""},
                        "check_inventory": {"playerId": "player1"},
                        "check_character_stats": {"playerId": "player1"},
                    }.get(name, {})
                    result.tool_calls.append({"call_id": call_id, "name": name,
                                              "arguments": {**defaults, **arguments}, "result": value})
                    output = {"toolUseId": call_id, "content": [{"json": value}], "status": "success"}
                except ValueError as exc:
                    result.errors.append(f"Tool {name}: {exc}")
                    output = {"toolUseId": call_id, "content": [{"text": str(exc)}], "status": "error"}
                outputs.append({"toolResult": output})
            messages.append({"role": "user", "content": outputs})
        result.errors.append(f"Agent exceeded {max_turns} turns")
    except Exception as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result.elapsed_seconds = time.perf_counter() - started
    return result


def required_call(case_index):
    if case_index in (0, 4, 5, 6):
        return "roll_dice", {"sides": 20, "count": 1}
    return {1: ("check_character_stats", {"playerId": "player1"}),
            2: ("check_inventory", {"playerId": "player1"}),
            3: ("describe_scene", {"location": "crypt"}),
            8: ("check_inventory", {"playerId": "player1"})}.get(case_index)


def verify_tools(result, case_index):
    from .metrics import check
    if case_index == 7:
        passed = not result.errors and not any(c["name"] == "roll_dice" for c in result.tool_calls)
        return check("no-premature-roll", float(passed))
    name, arguments = required_call(case_index)
    passed = any(c["name"] == name and all(c["arguments"].get(k) == v for k, v in arguments.items())
                 for c in result.tool_calls)
    return check("completed-tool-and-arguments", float(passed))


class ScriptedAgent:
    """Synthetic offline Converse transport; never a measured model response."""
    live = False
    model_id = "synthetic-agent"

    def __init__(self, case_index, answer):
        self.case_index, self.answer = case_index, answer
        self.requests = []

    def converse(self, **request):
        self.requests.append(deepcopy(request))
        required = required_call(self.case_index)
        if len(self.requests) == 1 and required:
            name, arguments = required
            if name == "roll_dice":
                arguments = {**arguments, "purpose": "attack roll or saving throw"}
            if name == "describe_scene":
                arguments = {**arguments, "mood": "ominous"}
            content = [{"toolUse": {"toolUseId": "synthetic-call-1", "name": name, "input": arguments}}]
            reason = "tool_use"
        else:
            content, reason = [{"text": self.answer}], "end_turn"
        return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": reason}
