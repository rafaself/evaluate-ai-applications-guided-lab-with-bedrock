# Boto3 notebook course · Evaluating AI applications

Five concise labs teach evaluation using Boto3 and plain Python. Work through them in order, or run any notebook independently. No AWS account or model calls are needed for the default offline mode.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

From the repository root:

```bash
cd labs/boto3
uv sync --locked
uv run jupyter lab
```

Open `notebooks/01_setup_and_basics.ipynb` and use the Python kernel from this environment. VS Code can also select `labs/boto3/.venv/bin/python`. Every notebook contains its own setup cell and starts with `LIVE = False`. Run all cells from top to bottom. Dependency installation requires internet access; the offline lessons do not.

| Lesson | What you build |
| --- | --- |
| [01 · Setup and basics](notebooks/01_setup_and_basics.ipynb) | Configuration, shared results, exports |
| [02 · Translation](notebooks/02_translation_evaluation.ipynb) | Case-sensitive/insensitive checks, false positives and negatives |
| [03 · Model comparison](notebooks/03_model_comparison.ipynb) | 15 riddles, rubric judgments, usage and latency comparison |
| [04 · RAG](notebooks/04_rag_evaluation.ipynb) | Five metrics for supplied policy context; retrieval is outside this lab |
| [05 · Agents](notebooks/05_agent_evaluation.ipynb) | Four tools, bounded Converse loop, nine agent scenarios |

Each lesson includes objectives, worked code, an exercise, hints, and interpretation checkpoints. Modify copies of cases and fixtures while experimenting. The shared modules are intentionally small enough to inspect alongside the notebooks.

## Offline evidence versus live results

`data/synthetic.json` contains hand-authored answers and grading annotations. They are synthetic examples, not recordings or measurements. All offline comparison models receive the same answers. Offline embedding vectors demonstrate arithmetic rather than semantic similarity. The offline agent executes real local tools against scripted requests; its final prose is independent of those results. These fixtures exercise evaluator behavior and cannot rank models or establish agent answer grounding.

Outputs identify their source, distinguish unavailable checks from failures, and record errors separately. Each notebook exports JSON and CSV under ignored `results/`. Synthetic token usage is empty, latency checks are unavailable, and unknown prices stay `None`. Agent offline wall time measures local script execution only.

## Optional live AWS experiments

```bash
cp .env.example .env
```

Configure a named AWS profile or the standard AWS credential chain. Temporary keys require `AWS_SESSION_TOKEN`. Never paste credentials into notebooks or commit `.env`. Environment variables take precedence over `.env`. Empty optional variables should be omitted if your credential provider requires that.

Before setting `LIVE = True` in a notebook, verify in your AWS account:

- The configured region, model/inference-profile IDs, and model access. Defaults are inherited examples and may need replacing.
- Converse support for response, comparison, and judge models; tool-use support for the agent model. The Python agent uses Converse, whereas the preserved TypeScript agent uses Responses.
- `bedrock:InvokeModel` permission on all chosen models/profiles, including graders and embeddings, plus any applicable cross-region profile permissions.
- The embedding model accepts the Titan Text Embeddings request shape. Changing to another embedding family requires adapting `Bedrock.embed()`.
- Current prices and the request counts stated in each notebook. SDK retries can add requests. Live execution is explicit per notebook, with no automatic fallback to synthetic answers on failure.

The first setup request is a minimal live connectivity check. No model discovery call can establish all IAM permissions or account-level access without attempting inference. Provider compatibility and generated-answer quality require live validation; the offline test suite does not establish either.

Rate-based costs are optional **response-only estimates**, using user-supplied per-model input/output rates per million tokens. They exclude graders, embeddings, retries, and other billing details. Judge/embedding usage is recorded separately. The 60-second and $0.15 checks run after completion; network timeouts and agent turn limits are independent controls, not dollar budgets. There is no response cache.

## Scoring and fixtures

Deterministic content assertions are ported from all original cases. The repeated breath riddle remains intentionally. Agent assertions use completed calls and selected arguments, plus Python equivalents of the original final-answer checks and rubric grades. They do not verify every game mechanic or that all tool values were used correctly. Game tools read fresh fixtures per scenario; they do not persist game state.

Semantic grading is an explicit teaching implementation, not a Promptfoo compatibility layer. Judges return validated JSON; malformed responses become errors, without inventing a score. Judgment is probabilistic, and treating input as quoted data reduces but does not eliminate judge prompt-injection risk.

| Metric | Formula / rubric |
| --- | --- |
| Factuality | Reference statements supported by answer / all reference statements |
| Answer relevance | Mean cosine similarity between query and three judge-generated questions about answer, floored at zero |
| Context recall | Reference statements supported by context / all reference statements |
| Context relevance | Context paragraphs needed for query / all context paragraphs |
| Context faithfulness | Answer claims supported by context / all answer claims |
| Rubric | Boolean satisfaction of each original scenario rubric, with reason |

The five RAG metrics use a 0.9 threshold; deterministic checks and rubrics require 1. The judge extracts atomic statements or paragraphs and returns per-item reasons. An answer without factual claims receives an unsupported item. Inspect decompositions before interpreting scores. Fred and the elephant leave entitlement are deliberately unsupported reference facts; low recall here diagnoses the reference rather than an unfaithful answer.

## Validation

```bash
uv run pytest
```

Tests check scenario preservation, calculations, invalid grades, exports, tool validation, fresh-state isolation, tool-result correlation, multiple/error calls, turn exhaustion, and final-answer truncation. Every notebook runs in a fresh kernel with AWS client creation and outbound socket connections blocked. Source notebooks remain cleared. This verification makes no live model calls.

## References

- [Boto3 Converse API](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/converse.html)
- [Bedrock client-side tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html)
- [Titan embedding request and response shapes](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html)
- [Promptfoo labs](../promptfoo/README.md)
