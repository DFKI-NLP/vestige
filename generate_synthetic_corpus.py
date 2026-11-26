#!/usr/bin/env python3
"""
Standalone anchor-guided synthetic corpus generator.

The generator is self-contained and does not import project-local pipeline
helpers from sibling scripts.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXAMPLE_CONFIG_PATH = (
    SCRIPT_DIR / "config" / "generate_synthetic_corpus.example.json"
)


def load_environment_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# Load a local, untracked credential file when present. Public configuration files
# must reference environment-variable names only; never place credentials in them.
load_environment_file(SCRIPT_DIR / ".env")


GENRE_ALIASES = {
    "patient-requests": "patient-request",
    "patient_requests": "patient-request",
    "aiha-anonymized-messages": "ai-health-assistant-message",
    "aiha_anonymized_messages": "ai-health-assistant-message",
    "aiha-message": "ai-health-assistant-message",
}


def _log(message: str) -> None:
    print(message, flush=True)


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 25,
    "run_name": "synthetic-health-texts",
    "output_root": "outputs",
    "providers": {
        "active": "azure",
        "azure": {
            "deployment": "YOUR_AZURE_DEPLOYMENT_NAME",
            "endpoint_env": "AZURE_OPENAI_ENDPOINT",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "api_version_env": "AZURE_OPENAI_API_VERSION",
            "api_version_default": "2024-12-01-preview",
            "pricing": {
                "currency": "USD",
                "input_per_1m": None,
                "cached_input_per_1m": None,
                "output_per_1m": None,
            },
        },
        "openai_compatible": {
            "base_url_env": "OPENAI_COMPATIBLE_BASE_URL",
            "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
            "model": "qwen2.5:14b-instruct",
            "pricing": {
                "currency": "USD",
                "input_per_1m": None,
                "cached_input_per_1m": None,
                "output_per_1m": None,
            },
        },
    },
    "generation": {
        "num_documents": 200,
        "documents_per_genre": None,
        "documents_per_genre_map": {},
        "min_documents_per_genre": 7,
        "sampling_distribution": {
            "medical-dictation": 0.145631,
            "patient-request": 0.466019,
            "phone-assistant": 0.097087,
            "consultation-assistant-summary": 0.097087,
            "consultation-assistant-transcript": 0.048544,
            "letter-generation": 0.048545,
            "consultation-assistant": 0.097087,
        },
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1800,
        },
        "human_review_sample_size_per_group": 5,
    },
    "priors": {
        "age": {
            "fr": {
                "0-19": 0.23,
                "20-39": 0.24,
                "40-64": 0.32,
                ">65": 0.22,
            },
            "de": {
                "<20": 0.19,
                "20-66": 0.61,
                "67+": 0.20,
            },
            "en": {
                "0-15": 0.18,
                "16-64": 0.62,
                "65-84": 0.20,
                ">85": 0.03,
            },
        },
        "gender": {
            "fr": {"male": 0.49, "female": 0.51},
            "de": {"male": 0.49, "female": 0.51},
            "en": {"male": 0.49, "female": 0.51},
        },
        "smoking_status": {
            "fr": {"yes": 0.23, "no": 0.77},
            "de": {"yes": 0.29, "no": 0.71},
            "en": {"yes": 0.11, "no": 0.89},
        },
        "employment_status": {
            "fr": {"employed": 0.69, "unemployed": 0.31},
            "de": {"employed": 0.77, "unemployed": 0.23},
            "en": {"employed": 0.75, "unemployed": 0.25},
        },
    },
    "anchor_documents": {
        "medical-dictation": {
            "fr": "anchors/fr/medical-dictation.txt",
            "de": "anchors/de/medical-dictation.txt",
            "en": "anchors/en/medical-dictation.txt",
        },
        "patient-request": {
            "fr": "anchors/fr/patient-request.txt",
            "de": "anchors/de/patient-request.txt",
            "en": "anchors/en/patient-request.txt",
        },
        "phone-assistant": {
            "fr": "anchors/fr/phone-assistant.txt",
            "de": "anchors/de/phone-assistant.txt",
            "en": "anchors/en/phone-assistant.txt",
        },
        "consultation-assistant-summary": {
            "fr": "anchors/fr/consultation-assistant-summary.txt",
            "de": "anchors/de/consultation-assistant-summary.txt",
            "en": "anchors/en/consultation-assistant-summary.txt",
        },
        "consultation-assistant-transcript": {
            "fr": "anchors/fr/consultation-assistant-transcript.txt",
            "de": "anchors/de/consultation-assistant-transcript.txt",
            "en": "anchors/en/consultation-assistant-transcript.txt",
        },
        "letter-generation": {
            "fr": "anchors/fr/letter-generation.txt",
            "de": "anchors/de/letter-generation.txt",
            "en": "anchors/en/letter-generation.txt",
        },
        "consultation-assistant": {
            "fr": "anchors/fr/consultation-assistant.txt",
            "de": "anchors/de/consultation-assistant.txt",
            "en": "anchors/en/consultation-assistant.txt",
        },
        "ai-health-assistant-message": {
            "fr": "anchors/fr/ai-health-assistant-message.txt",
            "de": "anchors/de/ai-health-assistant-message.txt",
            "en": "anchors/en/ai-health-assistant-message.txt",
        },
    },
    "quality": {
        # Optional external reference corpus for novelty checks. The public release
        # intentionally ships without source texts, so this is disabled by default.
        "reference_glob_template": "",
        "duplicate_hamming_threshold": 3,
        "max_pairwise_checks": 50000,
    },
    "genres": {
        "medical-dictation": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "complaint",
                    "symptoms",
                    "diagnosis",
                    "treatment",
                    "next_steps",
                    "doctor_name",
                    "clinic_name",
                    "appointment_date",
                ],
                "optional": [
                    "medications",
                    "allergies",
                    "city",
                    "visit_type",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 1400,
                    "char_max": 3400,
                    "token_min": 250,
                    "token_max": 650,
                },
                "de": {
                    "char_min": 1800,
                    "char_max": 4600,
                    "token_min": 300,
                    "token_max": 760,
                },
                "en": {
                    "char_min": 1600,
                    "char_max": 4200,
                    "token_min": 280,
                    "token_max": 700,
                },
            },
        },
        "patient-request": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "request_intent",
                ],
                "optional": [
                    "appointment_date",
                    "city",
                    "medications",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 120,
                    "char_max": 420,
                    "token_min": 18,
                    "token_max": 70,
                },
                "de": {
                    "char_min": 100,
                    "char_max": 360,
                    "token_min": 16,
                    "token_max": 60,
                },
                "en": {
                    "char_min": 110,
                    "char_max": 380,
                    "token_min": 18,
                    "token_max": 65,
                },
            },
        },
        "phone-assistant": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "complaint",
                    "symptoms",
                    "next_steps",
                    "doctor_name",
                    "clinic_name",
                    "appointment_date",
                    "phone_number",
                ],
                "optional": [
                    "visit_type",
                    "city",
                    "medications",
                    "allergies",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 1400,
                    "char_max": 3400,
                    "token_min": 220,
                    "token_max": 620,
                },
                "de": {
                    "char_min": 1400,
                    "char_max": 3400,
                    "token_min": 220,
                    "token_max": 620,
                },
                "en": {
                    "char_min": 1400,
                    "char_max": 3400,
                    "token_min": 220,
                    "token_max": 620,
                },
            },
        },
        "consultation-assistant-summary": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "complaint",
                    "symptoms",
                    "diagnosis",
                    "treatment",
                    "next_steps",
                    "doctor_name",
                ],
                "optional": [
                    "medications",
                    "visit_type",
                    "allergies",
                    "appointment_date",
                    "city",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 800,
                    "char_max": 2200,
                    "token_min": 120,
                    "token_max": 340,
                },
                "de": {
                    "char_min": 800,
                    "char_max": 2300,
                    "token_min": 120,
                    "token_max": 340,
                },
                "en": {
                    "char_min": 800,
                    "char_max": 2300,
                    "token_min": 120,
                    "token_max": 340,
                },
            },
        },
        "consultation-assistant-transcript": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "complaint",
                    "symptoms",
                    "diagnosis",
                    "treatment",
                    "next_steps",
                    "doctor_name",
                ],
                "optional": [
                    "medications",
                    "visit_type",
                    "appointment_date",
                    "allergies",
                    "clinic_name",
                    "city",
                    "family_context",
                    "hobbys_or_lifestyle_context",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 1800,
                    "char_max": 6200,
                    "token_min": 300,
                    "token_max": 1250,
                },
                "de": {
                    "char_min": 1800,
                    "char_max": 6200,
                    "token_min": 300,
                    "token_max": 1250,
                },
                "en": {
                    "char_min": 1800,
                    "char_max": 6200,
                    "token_min": 300,
                    "token_max": 1250,
                },
            },
        },
        "letter-generation": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "complaint",
                    "diagnosis",
                    "treatment",
                    "next_steps",
                    "doctor_name",
                    "clinic_name",
                    "city",
                ],
                "optional": [
                    "medications",
                    "appointment_date",
                    "allergies",
                    "visit_type",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 700,
                    "char_max": 2200,
                    "token_min": 120,
                    "token_max": 360,
                },
                "de": {
                    "char_min": 700,
                    "char_max": 2200,
                    "token_min": 120,
                    "token_max": 360,
                },
                "en": {
                    "char_min": 700,
                    "char_max": 2200,
                    "token_min": 120,
                    "token_max": 360,
                },
            },
        },
        "consultation-assistant": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "complaint",
                    "symptoms",
                    "diagnosis",
                    "treatment",
                    "next_steps",
                    "doctor_name",
                    "clinic_name",
                ],
                "optional": [
                    "medications",
                    "appointment_date",
                    "city",
                    "visit_type",
                    "allergies",
                    "family_context",
                    "hobbys_or_lifestyle_context",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 700,
                    "char_max": 2600,
                    "token_min": 100,
                    "token_max": 420,
                },
                "de": {
                    "char_min": 700,
                    "char_max": 2600,
                    "token_min": 100,
                    "token_max": 420,
                },
                "en": {
                    "char_min": 700,
                    "char_max": 2600,
                    "token_min": 100,
                    "token_max": 420,
                },
            },
        },
        "ai-health-assistant-message": {
            "languages": ["fr", "de", "en"],
            "slots": {
                "required": [
                    "message_body",
                ],
                "optional": [
                    "request_intent",
                    "symptoms",
                    "family_context",
                    "appointment_date",
                    "city",
                    "travel_context",
                    "age_context",
                ],
            },
            "length_targets": {
                "fr": {
                    "char_min": 20,
                    "char_max": 700,
                    "token_min": 4,
                    "token_max": 130,
                },
                "de": {
                    "char_min": 20,
                    "char_max": 700,
                    "token_min": 4,
                    "token_max": 130,
                },
                "en": {
                    "char_min": 20,
                    "char_max": 700,
                    "token_min": 4,
                    "token_max": 130,
                },
            },
        },
    },
}


def normalize_genre(genre: str) -> str:
    return GENRE_ALIASES.get(genre, genre)


def timestamp_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config_override(
    override: dict[str, Any], defaults: dict[str, Any], path: tuple[str, ...] = ()
) -> None:
    """Reject stale or misspelled configuration keys before merging overrides."""
    for key, value in override.items():
        current_path = path + (str(key),)
        if key not in defaults:
            dotted_path = ".".join(current_path)
            raise ValueError(f"Unknown configuration key: {dotted_path}")
        default_value = defaults[key]
        if isinstance(value, dict) and isinstance(default_value, dict):
            if current_path == ("generation", "documents_per_genre_map"):
                unknown_genres = set(value) - set(DEFAULT_CONFIG["genres"])
                if unknown_genres:
                    raise ValueError(
                        "Unknown genre in generation.documents_per_genre_map: "
                        f"{sorted(unknown_genres)}"
                    )
                continue
            validate_config_override(value, default_value, current_path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def glob_matches(pattern: str) -> list[str]:
    if not pattern:
        return []
    candidates = [pattern]
    if not os.path.isabs(pattern):
        candidates.append(str((SCRIPT_DIR / pattern).resolve()))
    matches: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for item in glob.glob(candidate):
            resolved = str(Path(item).resolve())
            if resolved not in seen:
                seen.add(resolved)
                matches.append(resolved)
    return matches


def weighted_choice(
    weights: dict[str, float], rng: random.Random, allowed: Optional[set[str]] = None
) -> str:
    filtered_items = []
    for key, value in weights.items():
        canonical = normalize_genre(key)
        if allowed is not None and canonical not in allowed and key not in allowed:
            continue
        weight = float(value)
        if weight > 0:
            filtered_items.append((canonical, weight))
    if not filtered_items:
        raise ValueError("No valid choices available from the provided weights.")
    total = sum(weight for _, weight in filtered_items)
    draw = rng.random() * total
    cumulative = 0.0
    for key, weight in filtered_items:
        cumulative += weight
        if draw <= cumulative:
            return key
    return filtered_items[-1][0]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿ']+", text.lower())


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def git_commit_hash() -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
    return output or None


class BaseProvider:
    def get_usage_summary(self) -> dict[str, int]:
        raise NotImplementedError

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError


class AzureProvider(BaseProvider):
    def __init__(self, cfg: dict[str, Any]):
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency `openai`. Install it before using Azure mode."
            ) from exc

        endpoint_env = cfg.get("endpoint_env", "AZURE_OPENAI_ENDPOINT")
        endpoint = os.getenv(endpoint_env)
        api_key_env = cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        api_version_env = cfg.get("api_version_env", "AZURE_OPENAI_API_VERSION")
        api_version = os.getenv(
            api_version_env, cfg.get("api_version_default", "2024-12-01-preview")
        )

        if not endpoint or not api_key:
            raise RuntimeError(
                "Azure credentials are not set. Set the configured endpoint and API-key "
                "environment variables."
            )
        self.client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self._warned_sampling_compat = False
        self._warned_token_compat = False
        self._usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
        }

    def _accumulate_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        def _value(obj: Any, key: str, default: int = 0) -> int:
            if isinstance(obj, dict):
                raw = obj.get(key, default)
            else:
                raw = getattr(obj, key, default)
            try:
                return int(raw or 0)
            except Exception:
                return default

        prompt_tokens = _value(usage, "prompt_tokens", 0)
        if prompt_tokens == 0:
            prompt_tokens = _value(usage, "input_tokens", 0)

        completion_tokens = _value(usage, "completion_tokens", 0)
        if completion_tokens == 0:
            completion_tokens = _value(usage, "output_tokens", 0)

        total_tokens = _value(usage, "total_tokens", prompt_tokens + completion_tokens)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        prompt_details = (
            usage.get("prompt_tokens_details", {})
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        cached_prompt_tokens = _value(prompt_details or {}, "cached_tokens", 0)

        self._usage["requests"] += 1
        self._usage["prompt_tokens"] += prompt_tokens
        self._usage["completion_tokens"] += completion_tokens
        self._usage["total_tokens"] += total_tokens
        self._usage["cached_prompt_tokens"] += cached_prompt_tokens

    def get_usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        base_params: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            base_params["response_format"] = {"type": "json_object"}

        def _call_with_token_param(
            token_param_name: str,
            *,
            include_temperature: bool,
            include_top_p: bool,
            temperature_value: float,
        ):
            params = dict(base_params)
            params[token_param_name] = max_tokens
            if include_temperature:
                params["temperature"] = temperature_value
            if include_top_p:
                params["top_p"] = top_p
            return self.client.chat.completions.create(**params)

        def _request_with_compat_fallback(token_param_name: str):
            attempts = [
                (True, True, temperature),
                (False, True, temperature),
                (True, False, 1.0),
                (False, False, 1.0),
            ]
            last_exc: Exception | None = None
            for include_temperature, include_top_p, temperature_value in attempts:
                try:
                    return _call_with_token_param(
                        token_param_name,
                        include_temperature=include_temperature,
                        include_top_p=include_top_p,
                        temperature_value=temperature_value,
                    )
                except Exception as exc:
                    msg = str(exc)
                    recoverable = (
                        "Unsupported value: 'temperature'" in msg
                        or "Unsupported parameter: 'temperature'" in msg
                        or "Unsupported parameter: 'top_p'" in msg
                    )
                    if recoverable:
                        if not self._warned_sampling_compat:
                            _log(
                                "[compat] Azure model rejected sampling params "
                                "(temperature/top_p). Retrying with compatible defaults."
                            )
                            self._warned_sampling_compat = True
                        last_exc = exc
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("No compatible request variant succeeded.")

        try:
            response = _request_with_compat_fallback("max_tokens")
        except Exception as exc:
            msg = str(exc)
            if "Unsupported parameter: 'max_tokens'" in msg:
                if not self._warned_token_compat:
                    _log(
                        "[compat] Azure model rejected max_tokens. "
                        "Retrying with max_completion_tokens."
                    )
                    self._warned_token_compat = True
                response = _request_with_compat_fallback("max_completion_tokens")
            else:
                raise
        self._accumulate_usage(response)
        return (response.choices[0].message.content or "").strip()


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, cfg: dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency `openai`. Install it before using local mode."
            ) from exc

        base_url = os.getenv(cfg.get("base_url_env", "OPENAI_COMPATIBLE_BASE_URL"))
        if not base_url:
            raise RuntimeError(
                "OpenAI-compatible base URL is not set. Set the configured base-URL "
                "environment variable."
            )
        api_key = os.getenv(cfg.get("api_key_env", "OPENAI_COMPATIBLE_API_KEY"), "local")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._warned_sampling_compat = False
        self._warned_token_compat = False
        self._usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
        }

    def _accumulate_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        def _value(obj: Any, key: str, default: int = 0) -> int:
            if isinstance(obj, dict):
                raw = obj.get(key, default)
            else:
                raw = getattr(obj, key, default)
            try:
                return int(raw or 0)
            except Exception:
                return default

        prompt_tokens = _value(usage, "prompt_tokens", 0)
        if prompt_tokens == 0:
            prompt_tokens = _value(usage, "input_tokens", 0)

        completion_tokens = _value(usage, "completion_tokens", 0)
        if completion_tokens == 0:
            completion_tokens = _value(usage, "output_tokens", 0)

        total_tokens = _value(usage, "total_tokens", prompt_tokens + completion_tokens)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        prompt_details = (
            usage.get("prompt_tokens_details", {})
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        cached_prompt_tokens = _value(prompt_details or {}, "cached_tokens", 0)

        self._usage["requests"] += 1
        self._usage["prompt_tokens"] += prompt_tokens
        self._usage["completion_tokens"] += completion_tokens
        self._usage["total_tokens"] += total_tokens
        self._usage["cached_prompt_tokens"] += cached_prompt_tokens

    def get_usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        base_params: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            base_params["response_format"] = {"type": "json_object"}

        def _call_with_token_param(
            token_param_name: str,
            *,
            include_temperature: bool,
            include_top_p: bool,
            temperature_value: float,
        ):
            params = dict(base_params)
            params[token_param_name] = max_tokens
            if include_temperature:
                params["temperature"] = temperature_value
            if include_top_p:
                params["top_p"] = top_p
            return self.client.chat.completions.create(**params)

        def _request_with_compat_fallback(token_param_name: str):
            attempts = [
                (True, True, temperature),
                (False, True, temperature),
                (True, False, 1.0),
                (False, False, 1.0),
            ]
            last_exc: Exception | None = None
            for include_temperature, include_top_p, temperature_value in attempts:
                try:
                    return _call_with_token_param(
                        token_param_name,
                        include_temperature=include_temperature,
                        include_top_p=include_top_p,
                        temperature_value=temperature_value,
                    )
                except Exception as exc:
                    msg = str(exc)
                    recoverable = (
                        "Unsupported value: 'temperature'" in msg
                        or "Unsupported parameter: 'temperature'" in msg
                        or "Unsupported parameter: 'top_p'" in msg
                    )
                    if recoverable:
                        if not self._warned_sampling_compat:
                            _log(
                                "[compat] OpenAI-compatible server rejected sampling params "
                                "(temperature/top_p). Retrying with compatible defaults."
                            )
                            self._warned_sampling_compat = True
                        last_exc = exc
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("No compatible request variant succeeded.")

        try:
            response = _request_with_compat_fallback("max_tokens")
        except Exception as exc:
            msg = str(exc)
            if "Unsupported parameter: 'max_tokens'" in msg:
                if not self._warned_token_compat:
                    _log(
                        "[compat] OpenAI-compatible server rejected max_tokens. "
                        "Retrying with max_completion_tokens."
                    )
                    self._warned_token_compat = True
                response = _request_with_compat_fallback("max_completion_tokens")
            else:
                raise
        self._accumulate_usage(response)
        return (response.choices[0].message.content or "").strip()


def create_provider(config: dict[str, Any], provider_name: str) -> BaseProvider:
    providers = config["providers"]
    if provider_name == "azure":
        return AzureProvider(providers["azure"])
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider(providers["openai_compatible"])
    raise ValueError(f"Unsupported provider: {provider_name}")


def simhash_64(tokens: list[str]) -> int:
    if not tokens:
        return 0
    vec = [0] * 64
    for token in set(tokens):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, byteorder="big", signed=False)
        for idx in range(64):
            vec[idx] += 1 if (h >> idx) & 1 else -1
    result = 0
    for idx, score in enumerate(vec):
        if score > 0:
            result |= 1 << idx
    return result


def near_duplicate_pairs(
    texts: list[str],
    *,
    hamming_threshold: int,
    max_pairwise_checks: int,
) -> dict[str, int]:
    signatures = [simhash_64(tokenize(text)) for text in texts]
    buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, sig in enumerate(signatures):
        buckets[sig >> 48].append((idx, sig))
    pairwise_checks = 0
    near_pairs = 0
    stop = False
    for bucket in buckets.values():
        for i in range(len(bucket)):
            _, sig_i = bucket[i]
            for j in range(i + 1, len(bucket)):
                _, sig_j = bucket[j]
                pairwise_checks += 1
                if pairwise_checks > max_pairwise_checks:
                    stop = True
                    break
                if (sig_i ^ sig_j).bit_count() <= hamming_threshold:
                    near_pairs += 1
            if stop:
                break
        if stop:
            break
    return {
        "near_duplicate_pairs": near_pairs,
        "pairwise_checks": pairwise_checks,
        "capped": int(stop),
    }


def load_reference_vocab(
    reference_glob_template: str, genre: str, lang: str
) -> set[str]:
    if not reference_glob_template:
        return set()
    vocab: set[str] = set()
    genre_candidates = [genre]
    if genre == "patient-request":
        genre_candidates.append("patient-requests")
    for candidate_genre in genre_candidates:
        pattern = reference_glob_template.format(genre=candidate_genre, lang=lang)
        for path in glob_matches(pattern):
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            vocab.update(tokenize(text))
    return vocab


def compute_quality_report(
    *,
    documents: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    quality_cfg = config.get("quality", {})
    reference_glob_template = quality_cfg.get("reference_glob_template", "")
    reference_check_enabled = bool(reference_glob_template)
    hamming_threshold = int(quality_cfg.get("duplicate_hamming_threshold", 3))
    max_pairwise_checks = int(quality_cfg.get("max_pairwise_checks", 50000))

    report: dict[str, Any] = {
        "total_documents": len(documents),
        "reference_check_enabled": reference_check_enabled,
        "groups": {},
        "duplicates": {},
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in documents:
        grouped[(row["genre"], row["lang"])].append(row)

    all_texts = [row["text"] for row in documents]
    normalized = [normalize_text(t).lower() for t in all_texts]
    counts = Counter(normalized)
    duplicate_docs = sum(count - 1 for count in counts.values() if count > 1)
    report["duplicates"]["exact_duplicate_docs"] = duplicate_docs
    report["duplicates"]["exact_duplicate_rate"] = (
        (duplicate_docs / len(all_texts)) if all_texts else 0.0
    )
    report["duplicates"].update(
        near_duplicate_pairs(
            all_texts,
            hamming_threshold=hamming_threshold,
            max_pairwise_checks=max_pairwise_checks,
        )
    )

    for (genre, lang), rows in sorted(grouped.items()):
        texts = [row["text"] for row in rows]
        tokens_all: list[str] = []
        lengths_chars = []
        lengths_tokens = []
        target_hits_chars = 0
        target_hits_tokens = 0
        targets = config["genres"][genre]["length_targets"][lang]
        for text in texts:
            tokens = tokenize(text)
            tokens_all.extend(tokens)
            char_count = len(text)
            token_count = len(tokens)
            lengths_chars.append(char_count)
            lengths_tokens.append(token_count)
            if targets["char_min"] <= char_count <= targets["char_max"]:
                target_hits_chars += 1
            if targets["token_min"] <= token_count <= targets["token_max"]:
                target_hits_tokens += 1

        vocab = set(tokens_all)
        ttr = (len(vocab) / len(tokens_all)) if tokens_all else 0.0
        group_metrics = {
            "n_docs": len(rows),
            "char_mean": statistics.fmean(lengths_chars) if lengths_chars else 0.0,
            "char_min": min(lengths_chars) if lengths_chars else 0,
            "char_max": max(lengths_chars) if lengths_chars else 0,
            "token_mean": statistics.fmean(lengths_tokens) if lengths_tokens else 0.0,
            "token_min": min(lengths_tokens) if lengths_tokens else 0,
            "token_max": max(lengths_tokens) if lengths_tokens else 0,
            "type_token_ratio": ttr,
            "char_target_hit_rate": (target_hits_chars / len(rows)) if rows else 0.0,
            "token_target_hit_rate": (target_hits_tokens / len(rows)) if rows else 0.0,
        }
        if reference_check_enabled:
            reference_vocab = load_reference_vocab(reference_glob_template, genre, lang)
            group_metrics["vocab_overlap_with_reference"] = (
                (len(vocab & reference_vocab) / len(vocab)) if vocab else 0.0
            )
        report["groups"][f"{genre}|{lang}"] = group_metrics
    return report


def choose_model_for_provider(config: dict[str, Any], provider_name: str) -> str:
    if provider_name == "azure":
        return config["providers"]["azure"]["deployment"]
    if provider_name == "openai_compatible":
        return config["providers"]["openai_compatible"]["model"]
    raise ValueError(f"Unsupported provider {provider_name}")


def estimate_usage_cost(
    usage: dict[str, int],
    pricing: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        input_per_1m = float(pricing.get("input_per_1m"))
        output_per_1m = float(pricing.get("output_per_1m"))
    except Exception:
        return None

    cached_input_per_1m_raw = pricing.get("cached_input_per_1m")
    cached_input_per_1m: Optional[float]
    try:
        cached_input_per_1m = (
            float(cached_input_per_1m_raw)
            if cached_input_per_1m_raw is not None
            else None
        )
    except Exception:
        cached_input_per_1m = None

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    cached_prompt_tokens = int(usage.get("cached_prompt_tokens", 0))
    cached_prompt_tokens = max(0, min(cached_prompt_tokens, prompt_tokens))
    uncached_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)

    input_cost = (uncached_prompt_tokens / 1_000_000.0) * input_per_1m
    cached_input_cost = 0.0
    if cached_input_per_1m is not None:
        cached_input_cost = (cached_prompt_tokens / 1_000_000.0) * cached_input_per_1m
    else:
        input_cost += (cached_prompt_tokens / 1_000_000.0) * input_per_1m

    output_cost = (completion_tokens / 1_000_000.0) * output_per_1m
    total_cost = input_cost + cached_input_cost + output_cost
    currency = pricing.get("currency", "USD")
    return {
        "currency": currency,
        "input_per_1m": input_per_1m,
        "cached_input_per_1m": cached_input_per_1m,
        "output_per_1m": output_per_1m,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_input_cost, 6),
        "output_cost": round(output_cost, 6),
        "estimated_total_cost": round(total_cost, 6),
    }


def load_config_or_default(path: Path) -> dict[str, Any]:
    if path.exists():
        file_cfg = load_json(path)
        if not isinstance(file_cfg, dict):
            raise ValueError("Configuration file must contain a JSON object.")
        validate_config_override(file_cfg, DEFAULT_CONFIG)
        cfg = deep_merge(DEFAULT_CONFIG, file_cfg)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    return cfg


def load_single_anchor_index(
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, str]]:
    anchor_cfg = config.get("anchor_documents") or {}
    if not anchor_cfg:
        raise ValueError("Config is missing `anchor_documents`.")

    anchors: dict[tuple[str, str], dict[str, str]] = {}
    for raw_genre, by_lang in anchor_cfg.items():
        genre = normalize_genre(raw_genre)
        for target_lang, pattern in (by_lang or {}).items():
            anchor_id = f"{target_lang}/{genre}"
            matches = glob_matches(str(pattern))
            if len(matches) != 1:
                raise ValueError(
                    "Each anchor_documents entry must resolve to exactly one file. "
                    f"{anchor_id} resolved to {len(matches)} files."
                )
            anchor_path = Path(matches[0]).resolve()
            try:
                anchor_text = normalize_text(
                    anchor_path.read_text(encoding="utf-8", errors="ignore")
                )
            except OSError as exc:
                raise ValueError(
                    f"Failed to read anchor document for {anchor_id}"
                ) from exc
            if not anchor_text:
                raise ValueError(f"Anchor document is empty: {anchor_id}")
            if "REPLACE_WITH_YOUR_SANITIZED_ANCHOR_TEXT" in anchor_text:
                raise ValueError(
                    "Anchor document still contains the public-release placeholder: "
                    f"{anchor_id}. Replace it with a permitted sanitized anchor."
                )
            anchors[(genre, target_lang)] = {
                "id": anchor_id,
                "text": anchor_text,
                "anchor_language": target_lang,
            }
    return anchors


def sample_prior_value(
    *,
    rng: random.Random,
    config: dict[str, Any],
    prior_name: str,
    lang: str,
) -> str:
    prior_cfg = config.get("priors", {}).get(prior_name, {})
    by_lang = prior_cfg.get(lang) or prior_cfg.get("fr") or prior_cfg.get("de")
    if not by_lang:
        raise ValueError(f"Missing prior distribution for {prior_name}/{lang}.")
    return weighted_choice(by_lang, rng)


def sample_profile_controls(
    *,
    rng: random.Random,
    config: dict[str, Any],
    lang: str,
) -> dict[str, str]:
    return {
        "age": sample_prior_value(rng=rng, config=config, prior_name="age", lang=lang),
        "gender": sample_prior_value(
            rng=rng, config=config, prior_name="gender", lang=lang
        ),
        "smoking_status": sample_prior_value(
            rng=rng, config=config, prior_name="smoking_status", lang=lang
        ),
        "employment_status": sample_prior_value(
            rng=rng, config=config, prior_name="employment_status", lang=lang
        ),
    }


def schema_slots_for_genre(genre_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    slots_cfg = genre_cfg.get("slots", {})
    required_slots = [str(slot) for slot in slots_cfg.get("required", [])]
    optional_slots = [str(slot) for slot in slots_cfg.get("optional", [])]
    return required_slots, optional_slots


def realism_guidance_for_language(lang: str) -> str:
    guidance = {
        "fr": (
            "For French output, use names, institutions, addresses, dates, abbreviations, "
            "and clinical phrasing that look native to French-speaking medical and administrative usage. "
            "If person or organization slots are present, prefer ordinary contemporary French-style names "
            "and plausible clinic or hospital naming patterns."
        ),
        "de": (
            "For German output, use names, institutions, addresses, dates, abbreviations, "
            "and clinical phrasing that look native to German medical and administrative usage. "
            "If person or organization slots are present, prefer ordinary contemporary German-style names "
            "and plausible Praxis, Klinik, MVZ, or Gesundheitszentrum naming patterns."
        ),
        "en": (
            "For English output, use names, institutions, addresses, dates, abbreviations, "
            "and clinical phrasing that look native to contemporary English-language medical and administrative usage. "
            "If person or organization slots are present, prefer ordinary contemporary English-style names "
            "and plausible clinic or hospital naming patterns."
        ),
    }
    fallback = (
        "Use names, institutions, addresses, and clinical phrasing that are natural for the target language "
        "and locale rather than generic multilingual placeholders."
    )
    return guidance.get(lang, fallback)


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def build_blueprint_prompt(
    *,
    doc_id: str,
    genre: str,
    lang: str,
    genre_cfg: dict[str, Any],
    controls: dict[str, str],
    anchor: dict[str, str],
) -> tuple[str, str]:
    length_cfg = genre_cfg.get("length_targets", {}).get(lang, {})
    required_slots, optional_slots = schema_slots_for_genre(genre_cfg)
    system_prompt = (
        "You create medical document blueprints for synthetic data generation. "
        "Return JSON only. It should read like ordinary real-world data. "
        "Do not copy the anchor text."
    )
    prompt = (
        "Create one JSON blueprint for a synthetic clinical document.\n"
        f"doc_id: {doc_id}\n"
        f"genre: {genre}\n"
        f"target_language: {lang}\n"
        f"target_char_range: {length_cfg.get('char_min')}..{length_cfg.get('char_max')}\n"
        f"target_token_range: {length_cfg.get('token_min')}..{length_cfg.get('token_max')}\n"
        "Use the sampled profile controls exactly as provided. "
        "They come from configurable priors and must remain editable in config.\n"
        f"sampled_profile_controls: {json.dumps(controls, ensure_ascii=False)}\n"
        f"required_schema_slots: {json.dumps(required_slots, ensure_ascii=False)}\n"
        f"optional_schema_slots: {json.dumps(optional_slots, ensure_ascii=False)}\n"
        f"anchor_id: {anchor['id']}\n"
        f"anchor_language: {anchor['anchor_language']}\n"
        f"language_realism_guidance: {realism_guidance_for_language(lang)}\n"
        "Use exactly this one anchor for high-level style, formatting, rhythm, and genre cues only. "
        "Do not reuse names, dates, addresses, phone numbers, clinics, or sentences from it.\n"
        "When filling schema slots, prefer culturally and linguistically plausible values for the target language. "
        "Names of patients, doctors, clinics, hospitals, and cities should feel ordinary and native for that language. "
        "They must not sound cartoonish, placeholder-like, exaggerated, self-consciously synthetic, or obviously machine-made.\n"
        "Do not use labels or markers such as fictive, fictional, fake, dummy, mock, test, sample, synthetic, invented, "
        "placeholder, example hospital, or example patient inside the slot values.\n"
        "Do not use redaction placeholders or angle-bracket markers such as <FIRST_NAME>, <LAST_NAME>, <DATE>, <CITY>, or similar tokens.\n"
        "Invent new values, but make them subtle, naturalistic, and unremarkable.\n"
        "Return a JSON object with exactly these top-level keys:\n"
        "- doc_id\n"
        "- genre\n"
        "- lang\n"
        "- anchor_id\n"
        "- anchor_language\n"
        "- sampled_profile_controls\n"
        "- schema_payload\n"
        "- blueprint\n"
        "The `schema_payload` object must contain every required schema slot and may contain any optional schema slot.\n"
        "Do not invent keys outside the provided required/optional schema slot lists.\n"
        "The `blueprint` object must contain:\n"
        "- visit_context\n"
        "- chief_complaint\n"
        "- key_clinical_facts (list of strings)\n"
        "- diagnostic_or_operational_impression\n"
        "- care_plan (list of strings)\n"
        "- style_notes (list of strings)\n"
        "- mention_controls_naturally (boolean)\n"
        "Anchor text follows:\n"
        f"{anchor['text']}\n"
    )
    return system_prompt, prompt


def validate_blueprint(
    blueprint_row: dict[str, Any],
    *,
    doc_id: str,
    genre: str,
    lang: str,
    genre_cfg: dict[str, Any],
) -> dict[str, Any]:
    row = copy.deepcopy(blueprint_row)
    row["doc_id"] = str(row.get("doc_id") or doc_id)
    row["genre"] = str(row.get("genre") or genre)
    row["lang"] = str(row.get("lang") or lang)
    required_slots, optional_slots = schema_slots_for_genre(genre_cfg)
    controls = row.get("sampled_profile_controls")
    if not isinstance(controls, dict):
        raise ValueError("Blueprint is missing `sampled_profile_controls`.")
    for key in ("age", "gender", "smoking_status", "employment_status"):
        value = controls.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Blueprint control is missing or empty: {key}")
    schema_payload = row.get("schema_payload")
    if not isinstance(schema_payload, dict):
        raise ValueError("Blueprint is missing `schema_payload` object.")
    allowed_schema_keys = set(required_slots) | set(optional_slots)
    unexpected_keys = sorted(set(schema_payload.keys()) - allowed_schema_keys)
    if unexpected_keys:
        raise ValueError(
            f"Schema payload contains keys outside genre schema: {unexpected_keys}"
        )
    for key in required_slots:
        if key not in schema_payload:
            raise ValueError(f"Schema payload missing required slot: {key}")
        value = schema_payload.get(key)
        if value is None:
            raise ValueError(f"Schema payload slot is null: {key}")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Schema payload string slot is empty: {key}")
        if isinstance(value, list) and not value:
            raise ValueError(f"Schema payload list slot is empty: {key}")
    payload = row.get("blueprint")
    if not isinstance(payload, dict):
        raise ValueError("Blueprint is missing `blueprint` object.")
    payload.setdefault("mention_controls_naturally", True)
    required_payload_keys = (
        "visit_context",
        "chief_complaint",
        "key_clinical_facts",
        "diagnostic_or_operational_impression",
        "care_plan",
        "style_notes",
        "mention_controls_naturally",
    )
    for key in required_payload_keys:
        if key not in payload:
            raise ValueError(f"Blueprint payload missing key: {key}")
    row["schema_required_slots"] = required_slots
    row["schema_optional_slots"] = optional_slots
    for key, value in schema_payload.items():
        row[key] = value
    return row


def llm_generate_blueprint(
    *,
    provider: BaseProvider,
    model: str,
    decoding: dict[str, Any],
    doc_id: str,
    genre: str,
    lang: str,
    genre_cfg: dict[str, Any],
    controls: dict[str, str],
    anchor: dict[str, str],
) -> dict[str, Any]:
    system_prompt, prompt = build_blueprint_prompt(
        doc_id=doc_id,
        genre=genre,
        lang=lang,
        genre_cfg=genre_cfg,
        controls=controls,
        anchor=anchor,
    )
    max_attempts = 3
    retry_prompt = prompt
    last_exc: Optional[Exception] = None
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = provider.complete(
            prompt=retry_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=float(decoding.get("temperature", 0.0)),
            top_p=float(decoding.get("top_p", 1.0)),
            max_tokens=int(decoding.get("max_tokens", 1400)),
            json_mode=True,
        )
        last_raw = raw
        try:
            parsed = parse_json_object(raw)
            return validate_blueprint(
                parsed,
                doc_id=doc_id,
                genre=genre,
                lang=lang,
                genre_cfg=genre_cfg,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            _log(
                f"[retry] blueprint {doc_id} attempt {attempt}/{max_attempts} failed: {exc}"
            )
            retry_prompt = (
                f"{prompt}\n"
                "Your previous response was invalid. Repair it and return one complete JSON object only.\n"
                f"Validation error: {exc}\n"
                "Do not omit any required schema slot. Keep all top-level keys and blueprint keys.\n"
                "Previous invalid response follows:\n"
                f"{last_raw}\n"
            )
    raise ValueError(
        f"Blueprint generation failed for {doc_id} after {max_attempts} attempts: {last_exc}"
    )


def build_render_prompt(
    *,
    blueprint_row: dict[str, Any],
    genre_cfg: dict[str, Any],
    anchor: dict[str, str],
) -> tuple[str, str]:
    lang = str(blueprint_row["lang"])
    length_cfg = genre_cfg.get("length_targets", {}).get(lang, {})
    system_prompt = (
        "You write medical documents for synthetic data generation. "
        "Return plain text only. Do not mention that the document is synthetic. "
        "The result should read like an ordinary in-domain document written by a native user of the target language."
    )
    prompt = (
        "Write one document from the provided blueprint.\n"
        f"genre: {blueprint_row['genre']}\n"
        f"target_language: {lang}\n"
        f"target_char_range: {length_cfg.get('char_min')}..{length_cfg.get('char_max')}\n"
        f"target_token_range: {length_cfg.get('token_min')}..{length_cfg.get('token_max')}\n"
        f"language_realism_guidance: {realism_guidance_for_language(lang)}\n"
        "Use the anchor only for style, register, structure, and rhythm. "
        "Do not copy wording, named entities, dates, numbers, addresses, or phone numbers.\n"
        f"anchor_id: {anchor['id']}\n"
        f"anchor_language: {anchor['anchor_language']}\n"
        "Use the blueprint semantics faithfully. Keep profile controls editable by reflecting the sampled values "
        "naturally when the genre supports it, but do not force awkward demographic exposition.\n"
        "If the schema includes names of patients, doctors, clinics, hospitals, or cities, render them as natural, "
        "plausible entities for the target language and setting. They should feel realistic and routine, not theatrical, "
        "playful, obviously fabricated, or visibly synthetic.\n"
        "Do not insert labels such as fictional, fake, dummy, mock, synthetic, invented, placeholder, or example into the text.\n"
        "Do not output redaction placeholders or angle-bracket markers such as <FIRST_NAME>, <LAST_NAME>, <DATE>, <CITY>, or similar tokens.\n"
        f"blueprint_json: {json.dumps(blueprint_row, ensure_ascii=False, indent=2)}\n"
        "Anchor text follows:\n"
        f"{anchor['text']}\n"
        "Output text only."
    )
    return system_prompt, prompt


def llm_render_document(
    *,
    provider: BaseProvider,
    model: str,
    decoding: dict[str, Any],
    blueprint_row: dict[str, Any],
    genre_cfg: dict[str, Any],
    anchor: dict[str, str],
) -> str:
    system_prompt, prompt = build_render_prompt(
        blueprint_row=blueprint_row,
        genre_cfg=genre_cfg,
        anchor=anchor,
    )
    text = provider.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=float(decoding.get("temperature", 0.0)),
        top_p=float(decoding.get("top_p", 1.0)),
        max_tokens=int(decoding.get("max_tokens", 1800)),
        json_mode=False,
    )
    clean = normalize_text(text)
    if not clean:
        raise ValueError("LLM returned empty document text.")
    return clean


def build_generation_plan(
    *,
    cfg: dict[str, Any],
    rng: random.Random,
    genre_cfgs: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    generation_plan: list[tuple[str, str]] = []

    def configured_languages(genre: str) -> list[str]:
        allowed_langs = sorted(
            str(lang) for lang in genre_cfgs[genre].get("languages", [])
        )
        if not allowed_langs:
            raise ValueError(f"Genre {genre} has no configured languages.")
        return allowed_langs

    def extend_round_robin(genre: str, count: int) -> None:
        allowed_langs = configured_languages(genre)
        for idx in range(count):
            generation_plan.append((genre, allowed_langs[idx % len(allowed_langs)]))

    def append_weighted_draw(genre: str) -> None:
        allowed_langs = configured_languages(genre)
        generation_plan.append((genre, rng.choice(allowed_langs)))

    per_genre_map = cfg["generation"].get("documents_per_genre_map") or {}
    per_genre = cfg["generation"].get("documents_per_genre")
    if per_genre_map:
        for raw_genre, count in per_genre_map.items():
            genre = normalize_genre(raw_genre)
            if genre not in genre_cfgs:
                raise ValueError(
                    f"Unknown genre in documents_per_genre_map: {raw_genre}"
                )
            repetitions = int(count)
            if repetitions < 0:
                raise ValueError(
                    f"documents_per_genre_map has negative count for {raw_genre}"
                )
            allowed_langs = configured_languages(genre)
            for lang in allowed_langs:
                generation_plan.extend([(genre, lang)] * repetitions)
    elif per_genre is not None:
        repetitions = int(per_genre)
        if repetitions < 0:
            raise ValueError("documents_per_genre cannot be negative")
        for genre in sorted(genre_cfgs.keys()):
            allowed_langs = configured_languages(genre)
            for lang in allowed_langs:
                generation_plan.extend([(genre, lang)] * repetitions)
    else:
        num_documents = int(cfg["generation"]["num_documents"])
        min_per_genre = int(cfg["generation"].get("min_documents_per_genre", 0) or 0)
        if min_per_genre < 0:
            raise ValueError("min_documents_per_genre cannot be negative")
        genre_weights: dict[str, float] = {}
        raw_genre_weights = cfg["generation"]["sampling_distribution"]
        for raw_genre, weight in raw_genre_weights.items():
            genre = normalize_genre(raw_genre)
            if genre not in genre_cfgs:
                continue
            try:
                configured_languages(genre)
            except ValueError:
                continue
            genre_weights[genre] = float(weight)
        if not genre_weights:
            raise ValueError("No valid genre/language pairs available for generation.")
        floor_total = min_per_genre * len(genre_weights)
        if num_documents < floor_total:
            raise ValueError(
                f"num_documents={num_documents} is too small for "
                f"min_documents_per_genre={min_per_genre} across {len(genre_weights)} genres."
            )
        for genre in sorted(genre_weights):
            extend_round_robin(genre, min_per_genre)
        remaining_documents = num_documents - floor_total
        genre_items = list(genre_weights.items())
        total_weight = sum(weight for _, weight in genre_items)
        for _ in range(remaining_documents):
            draw = rng.random() * total_weight
            cumulative = 0.0
            selected_genre = genre_items[-1][0]
            for genre, weight in genre_items:
                cumulative += weight
                if draw <= cumulative:
                    selected_genre = genre
                    break
            append_weighted_draw(selected_genre)
    rng.shuffle(generation_plan)
    return generation_plan


def pipeline_run(
    *,
    config: dict[str, Any],
    output_root_override: Optional[str] = None,
    provider_override: Optional[str] = None,
    num_documents_override: Optional[int] = None,
    documents_per_genre_override: Optional[int] = None,
    no_write: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)

    if provider_override:
        cfg["providers"]["active"] = provider_override
    if num_documents_override is not None:
        cfg["generation"]["num_documents"] = int(num_documents_override)
        cfg["generation"]["documents_per_genre"] = None
        cfg["generation"]["documents_per_genre_map"] = {}
    if documents_per_genre_override is not None:
        cfg["generation"]["documents_per_genre"] = int(documents_per_genre_override)
        cfg["generation"]["documents_per_genre_map"] = {}
    if output_root_override:
        cfg["output_root"] = output_root_override

    rng = random.Random(int(cfg.get("seed", 1337)))
    provider_name = cfg["providers"]["active"]
    decoding = cfg["generation"]["decoding"]
    model = choose_model_for_provider(cfg, provider_name)
    anchors = load_single_anchor_index(cfg)
    genre_cfgs = {normalize_genre(k): v for k, v in cfg["genres"].items()}
    _log(
        f"[setup] provider={provider_name} model={model} "
        "pipeline_mode=llm_anchor_only"
    )
    _log("[setup] initializing model provider...")
    provider = create_provider(cfg, provider_name)
    _log("[setup] provider initialized.")

    run_id = f"{cfg.get('run_name', 'run')}_{timestamp_utc()}"
    run_dir = Path(cfg["output_root"]) / run_id
    if not no_write:
        run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[setup] run_id={run_id}")
    _log(f"[setup] output_dir={run_dir}")
    config_used_path = run_dir / "config_used.json"
    run_metadata_path = run_dir / "run_metadata.json"
    frames_path = run_dir / "frames.jsonl"
    documents_path = run_dir / "documents.jsonl"
    failures_path = run_dir / "failed_documents.jsonl"
    quality_report_path = run_dir / "quality_report.json"
    review_sample_path = run_dir / "human_review_sample.jsonl"

    generation_plan = build_generation_plan(cfg=cfg, rng=rng, genre_cfgs=genre_cfgs)
    total_documents = len(generation_plan)
    plan_counts = Counter(generation_plan)
    plan_str = ", ".join(
        f"{genre}/{lang}:{count}"
        for (genre, lang), count in sorted(plan_counts.items())
    )
    _log(f"[plan] total_documents={total_documents}")
    _log(f"[plan] by_genre_lang={plan_str}")

    frames: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metadata = {
        "run_id": run_id,
        "created_at_utc": timestamp_utc(),
        "git_commit": git_commit_hash(),
        "num_documents": total_documents,
        "generated_documents_so_far": 0,
        "failed_documents_so_far": 0,
        "documents_per_genre": cfg["generation"].get("documents_per_genre"),
        "documents_per_genre_map": cfg["generation"].get("documents_per_genre_map", {}),
        "provider": provider_name,
        "model": model,
        "pipeline_mode": "llm_anchor_only",
        "status": "running",
        "decoding": decoding,
        "usage_summary": None,
        "cost_summary": None,
    }

    if not no_write:
        _log("[save] initializing output files...")
        save_json(config_used_path, cfg)
        save_json(run_metadata_path, metadata)
        frames_path.write_text("", encoding="utf-8")
        documents_path.write_text("", encoding="utf-8")
        failures_path.write_text("", encoding="utf-8")
        _log("[save] initialized config_used.json")
        _log("[save] initialized run_metadata.json")
        _log("[save] initialized frames.jsonl")
        _log("[save] initialized documents.jsonl")
        _log("[save] initialized failed_documents.jsonl")

    try:
        for idx, (genre, lang) in enumerate(generation_plan):
            genre_cfg = genre_cfgs[genre]
            if (genre, lang) not in anchors:
                raise ValueError(f"Missing anchor document for {genre}/{lang}.")
            doc_id = f"{run_id}_{idx:05d}"
            anchor = anchors[(genre, lang)]
            controls = sample_profile_controls(rng=rng, config=cfg, lang=lang)
            stage = "blueprint"
            try:
                blueprint_row = llm_generate_blueprint(
                    provider=provider,
                    model=model,
                    decoding=decoding,
                    doc_id=doc_id,
                    genre=genre,
                    lang=lang,
                    genre_cfg=genre_cfg,
                    controls=controls,
                    anchor=anchor,
                )
                stage = "render"
                text = llm_render_document(
                    provider=provider,
                    model=model,
                    decoding=decoding,
                    blueprint_row=blueprint_row,
                    genre_cfg=genre_cfg,
                    anchor=anchor,
                )
            except Exception as exc:
                failure_row = {
                    "doc_id": doc_id,
                    "genre": genre,
                    "lang": lang,
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "sampled_profile_controls": controls,
                    "anchor_id": anchor["id"],
                }
                failures.append(failure_row)
                metadata["failed_documents_so_far"] = len(failures)
                if not no_write:
                    append_jsonl_row(failures_path, failure_row)
                    if (idx + 1) % 10 == 0 or idx + 1 == total_documents:
                        save_json(run_metadata_path, metadata)
                _log(
                    f"[skip] {idx + 1}/{total_documents} genre={genre} lang={lang} "
                    f"id={doc_id} stage={stage} error={exc}"
                )
                continue

            blueprint_row["frame_source"] = "llm"
            blueprint_row["text_source"] = "llm"
            blueprint_row["char_count"] = len(text)
            blueprint_row["token_count"] = len(tokenize(text))
            blueprint_row["generation_parameters"] = {
                "provider": provider_name,
                "model": model,
                "pipeline_mode": "llm_anchor_only",
                "temperature": decoding.get("temperature"),
                "top_p": decoding.get("top_p"),
                "max_tokens": decoding.get("max_tokens"),
            }
            document_row = {
                "doc_id": doc_id,
                "genre": genre,
                "lang": lang,
                "text": text,
                "frame": blueprint_row,
            }

            frames.append(blueprint_row)
            docs.append(document_row)

            if not no_write:
                append_jsonl_row(frames_path, blueprint_row)
                append_jsonl_row(documents_path, document_row)
                metadata["generated_documents_so_far"] = len(docs)
                if (idx + 1) % 10 == 0 or idx + 1 == total_documents:
                    save_json(run_metadata_path, metadata)

            if idx < 5:
                _log(
                    f"[doc] {idx + 1}/{total_documents} genre={genre} lang={lang} id={doc_id}"
                )
            if (idx + 1) % 10 == 0 or idx + 1 == total_documents:
                _log(f"[progress] generated {idx + 1}/{total_documents}")
    except Exception as exc:
        if not no_write:
            metadata["status"] = "failed"
            metadata["generated_documents_so_far"] = len(docs)
            metadata["failed_documents_so_far"] = len(failures)
            metadata["error"] = str(exc)
            save_json(run_metadata_path, metadata)
        raise

    _log("[quality] computing quality report...")
    quality_report = compute_quality_report(documents=docs, config=cfg)
    _log("[quality] completed.")

    usage_summary = provider.get_usage_summary()
    cost_summary: Optional[dict[str, Any]] = None
    if usage_summary is not None:
        _log(
            "[usage] "
            f"requests={usage_summary.get('requests', 0)} "
            f"prompt_tokens={usage_summary.get('prompt_tokens', 0)} "
            f"completion_tokens={usage_summary.get('completion_tokens', 0)} "
            f"cached_prompt_tokens={usage_summary.get('cached_prompt_tokens', 0)} "
            f"total_tokens={usage_summary.get('total_tokens', 0)}"
        )
        provider_cfg = cfg.get("providers", {}).get(provider_name, {})
        pricing_cfg = provider_cfg.get("pricing", {})
        cost_summary = estimate_usage_cost(usage_summary, pricing_cfg)
        if cost_summary is not None:
            _log(
                "[cost] "
                f"estimated_total_{cost_summary['currency']}="
                f"{cost_summary['estimated_total_cost']:.6f} "
                f"(input={cost_summary['input_cost']:.6f}, "
                f"cached_input={cost_summary['cached_input_cost']:.6f}, "
                f"output={cost_summary['output_cost']:.6f})"
            )

    review_n = int(cfg["generation"].get("human_review_sample_size_per_group", 5))
    review_pack: list[dict[str, Any]] = []
    grouped_docs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in docs:
        grouped_docs[(row["genre"], row["lang"])].append(row)
    for _, rows in sorted(grouped_docs.items()):
        review_pack.extend(rows[:review_n])

    metadata["status"] = "completed"
    metadata["generated_documents_so_far"] = len(docs)
    metadata["failed_documents_so_far"] = len(failures)
    metadata["usage_summary"] = usage_summary
    metadata["cost_summary"] = cost_summary

    if not no_write:
        _log("[save] finalizing output files...")
        save_json(run_metadata_path, metadata)
        save_json(quality_report_path, quality_report)
        save_jsonl(review_sample_path, review_pack)
        _log("[save] updated run_metadata.json")
        _log("[save] wrote quality_report.json")
        _log("[save] wrote human_review_sample.jsonl")
    else:
        _log("[save] no-write mode enabled, outputs not written.")

    return {
        "run_dir": str(run_dir),
        "generated_documents": len(docs),
        "quality_report": quality_report,
        "metadata": metadata,
        "usage_summary": usage_summary,
        "cost_summary": cost_summary,
        "no_write": no_write,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an anchored synthetic health-text corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXAMPLE_CONFIG_PATH,
        help="Path to LLM-anchor pipeline config JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root directory.",
    )
    parser.add_argument(
        "--provider",
        choices=["azure", "openai_compatible"],
        default=None,
        help="Override provider.",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=None,
        help="Override number of documents to generate.",
    )
    parser.add_argument(
        "--documents-per-genre",
        type=int,
        default=None,
        help="Override fixed number of documents to generate for each configured genre.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the pipeline without writing outputs; model calls still occur.",
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        default=None,
        help="Write default config to this path and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        save_json(args.write_default_config, DEFAULT_CONFIG)
        print(f"Wrote default config to {args.write_default_config}")
        return

    config = load_config_or_default(args.config)
    _log(f"[setup] using config={args.config}")
    summary = pipeline_run(
        config=config,
        output_root_override=args.output_root,
        provider_override=args.provider,
        num_documents_override=args.num_documents,
        documents_per_genre_override=args.documents_per_genre,
        no_write=args.no_write,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
