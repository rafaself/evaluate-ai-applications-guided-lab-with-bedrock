"""Configuration, shared results, and offline-safe Bedrock access."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import os
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_cases(name):
    return json.loads((DATA / f"{name}.json").read_text())


@dataclass
class Result:
    scenario_id: str
    model_id: str
    response: str = ""
    assertions: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    token_usage: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    source: str = "synthetic"
    estimated_cost_usd: float | None = None
    grading_usage: list = field(default_factory=list)

    def estimate_cost(self, input_per_million=None, output_per_million=None):
        """Response cost only; graders/embeddings have separate usage records."""
        if input_per_million is None or output_per_million is None:
            self.estimated_cost_usd = None
        elif min(input_per_million, output_per_million) < 0:
            raise ValueError("Rates must be nonnegative")
        elif self.source == "live":
            self.estimated_cost_usd = (
                self.token_usage.get("inputTokens", 0) * input_per_million
                + self.token_usage.get("outputTokens", 0) * output_per_million
            ) / 1_000_000
        return self.estimated_cost_usd


def export_results(results, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(Result.__dataclass_fields__)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


class Bedrock:
    """Constructs no SDK client and performs no credential lookup in offline mode."""
    def __init__(self, live=False, client=None):
        self.live = live
        self._client = client
        self.model_id = os.getenv("RESPONSE_MODEL_ID", "us.openai.gpt-6-luna")
        self.judge_id = os.getenv("JUDGE_MODEL_ID", self.model_id)
        self.embedding_id = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
        self.usage = []

    @property
    def client(self):
        if not self.live:
            raise RuntimeError("Live AWS calls disabled; set LIVE = True explicitly")
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client(
                "bedrock-runtime", region_name=os.getenv("AWS_BEDROCK_REGION", "us-east-1"),
                config=Config(connect_timeout=10, read_timeout=90,
                              retries={"total_max_attempts": 2}),
            )
        return self._client

    def converse(self, **kwargs):
        return self.client.converse(**kwargs)

    def text(self, prompt, model_id=None):
        started = time.perf_counter()
        response = self.converse(
            modelId=model_id or self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2048},
        )
        elapsed = time.perf_counter() - started
        text = "\n".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
        self.usage.append({"model_id": model_id or self.model_id,
                           "tokens": response.get("usage", {}), "elapsed_seconds": elapsed})
        return text, response.get("usage", {}), elapsed

    def embed(self, text):
        """Adapter for Titan Text Embeddings G1/V2-compatible request/response bodies."""
        response = self.client.invoke_model(
            modelId=self.embedding_id, contentType="application/json", accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        body = json.loads(response["body"].read())
        self.usage.append({"model_id": self.embedding_id,
                           "tokens": {"inputTokens": body.get("inputTextTokenCount", 0)}})
        return body["embedding"]


def generate(api, case_id, prompt, synthetic, model_id=None):
    result = Result(case_id, model_id or api.model_id,
                    source="live" if api.live else "synthetic")
    if not api.live:
        result.response = synthetic
        return result
    try:
        result.response, result.token_usage, result.elapsed_seconds = api.text(prompt, model_id)
    except Exception as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
    return result
