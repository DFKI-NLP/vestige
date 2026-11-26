## VESTIGE: What De-identification Leaves Behind in Multilingual Healthcare Communication

This repository provides a pipeline for generating synthetic health texts across eight ambulantory care genres in German, English and French.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your Azure OpenAI endpoint and API key, then set the Azure
deployment name in `config/generate_synthetic_corpus.example.json`.

## Add permitted anchors

The repository ships with placeholders in `anchor_templates/`. From the
repository root, run this command once to create your local `anchors/` folder:

```bash
cp -R anchor_templates anchors
```

In `anchors/`, replace the placeholder text in each file with a permitted,
sanitized anchor for that language and genre. Keep the folder structure and
filenames as provided. If `anchors/` already exists, edit the files there
directly. Git ignores `anchors/`, and `anchor_templates/` stays ready to share
with the repository.

## Generate documents

```bash
python generate_synthetic_corpus.py \
  --config config/generate_synthetic_corpus.example.json \
  --documents-per-genre 10
```

Outputs are written beneath `outputs/` and include generated documents, sampled
frames, quality checks, run metadata, and a small review sample. `outputs/` and
`.env` are ignored by Git.

`--no-write` performs generation and quality checks without saving outputs; it
still makes model calls. Use it only when that behavior is intended.

The script contains the fixed genre schemas, length targets, and the 24 expected
anchor paths. Configuration files override those public defaults; unknown keys
are rejected to prevent stale settings from being silently ignored. Use
`--write-default-config <path>` to inspect or customize the complete effective
configuration.

## Included genres

| Display name | Output identifier |
| --- | --- |
| AI-health assistant message | `ai-health-assistant-message` |
| Consultation assistant dialogue | `consultation-assistant` |
| Consultation assistant summary | `consultation-assistant-summary` |
| Consultation assistant transcript | `consultation-assistant-transcript` |
| Referral letter | `letter-generation` |
| Medical dictation | `medical-dictation` |
| Patient request | `patient-request` |
| Phone assistant | `phone-assistant` |

## Alternative provider

Azure OpenAI is the documented default. The generator also supports an
OpenAI-compatible endpoint for local or self-hosted use. Set
`providers.active` to `openai_compatible` in a custom configuration, set its
model there, and provide `OPENAI_COMPATIBLE_BASE_URL` and (when needed)
`OPENAI_COMPATIBLE_API_KEY` through `.env`.

For local generation with [Ollama](https://docs.ollama.com/api/openai-compatibility),
start Ollama and install a model, for example with `ollama pull qwen2.5:7b`.
Set `OPENAI_COMPATIBLE_BASE_URL=http://localhost:11434/v1` in `.env` and set
`providers.openai_compatible.model` to `qwen2.5:7b` in your configuration.
The blueprint step uses JSON mode. This setup runs locally without Azure
credentials.

## Cite

TBA
