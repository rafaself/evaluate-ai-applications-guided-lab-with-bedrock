# Evaluate AI Applications — Guided Labs

Two independent lab tracks explore AI evaluation on Amazon Bedrock:

| Track | Runtime | Start here |
| --- | --- | --- |
| Boto3 notebook course | Python 3.12+, Boto3, Jupyter | [Boto3 labs](labs/boto3/README.md) |
| Promptfoo evaluation labs | Node.js 24, Promptfoo, TypeScript agent | [Promptfoo labs](labs/promptfoo/README.md) |

The Boto3 course contains five concise, executable lessons with exercises and offline teaching fixtures. The Promptfoo track preserves its runtime, configs, dependencies, and agent examples separately. Each track uses its own environment and local `.env`; run commands from the track directory.

Live model calls incur AWS usage. The Boto3 lessons run offline by default. Neither track's cost or latency assertions act as spending limits.

The scenarios were adapted from [Promptfoo examples](https://github.com/promptfoo/promptfoo/tree/main/examples). The Boto3 lab implementation defines its own scoring methods and does not promise equivalent scores.
