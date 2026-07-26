from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .schema import EvalExample


@dataclass(slots=True)
class JudgeConfig:
    provider: str
    task: str
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_retries: int
    retry_sleep: float


@dataclass(slots=True)
class JudgeResult:
    prediction: str | None
    correct: bool | None
    raw_response: str | None
    error: str | None = None


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge",
        choices=("none", "mimo", "seed"),
        default=default_judge_provider(),
        help="Optional OpenAI-compatible judge for open-ended responses.",
    )
    parser.add_argument(
        "--judge-task",
        choices=("correctness",),
        default=os.environ.get("JUDGE_TASK", "correctness"),
        help="Ask the judge to compare model response with the ground-truth answer.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("JUDGE_BASE_URL"),
        help="OpenAI-compatible base URL or /chat/completions URL for the judge.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL"),
        help="Judge model name.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=os.environ.get("JUDGE_API_KEY_ENV"),
        help="Environment variable name that stores the judge API key.",
    )
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-retry-sleep", type=float, default=2.0)


def build_judge_config(args: argparse.Namespace) -> JudgeConfig | None:
    if args.judge == "none":
        return None

    base_url = args.judge_base_url or default_judge_base_url(args.judge)
    model = args.judge_model or default_judge_model(args.judge)
    api_key_env = args.judge_api_key_env or default_judge_api_key_env(args.judge)
    api_key = os.environ.get(api_key_env)
    missing = []
    if not base_url:
        missing.append(f"{default_judge_base_url_env(args.judge)} or --judge-base-url")
    if not model:
        missing.append(f"{default_judge_model_env(args.judge)} or --judge-model")
    if not api_key:
        missing.append(api_key_env)
    if missing:
        raise ValueError(
            f"{args.judge} judge is enabled but required configuration is missing: "
            + ", ".join(missing)
        )

    return JudgeConfig(
        provider=args.judge,
        task=args.judge_task,
        base_url=chat_completions_url(base_url),
        api_key=api_key,
        model=model,
        timeout=args.judge_timeout,
        max_retries=args.judge_max_retries,
        retry_sleep=args.judge_retry_sleep,
    )


def default_judge_provider() -> str:
    if os.environ.get("JUDGE"):
        return os.environ["JUDGE"]
    if os.environ.get("ARK_API_KEY") or os.environ.get("ARK_BASE_URL") or os.environ.get("ARK_MODEL"):
        return "seed"
    return "mimo"


def default_judge_base_url(provider: str) -> str | None:
    if provider == "seed":
        return os.environ.get("ARK_BASE_URL")
    if provider == "mimo":
        return os.environ.get("MIMO_BASE_URL")
    return None


def default_judge_model(provider: str) -> str | None:
    if provider == "seed":
        return os.environ.get("ARK_MODEL")
    if provider == "mimo":
        return os.environ.get("MIMO_MODEL")
    return None


def default_judge_api_key_env(provider: str) -> str:
    if provider == "seed":
        return os.environ.get("ARK_API_KEY_ENV", "ARK_API_KEY")
    if provider == "mimo":
        return os.environ.get("MIMO_API_KEY_ENV", "MIMO_API_KEY")
    return "JUDGE_API_KEY"


def default_judge_base_url_env(provider: str) -> str:
    if provider == "seed":
        return "ARK_BASE_URL"
    if provider == "mimo":
        return "MIMO_BASE_URL"
    return "JUDGE_BASE_URL"


def default_judge_model_env(provider: str) -> str:
    if provider == "seed":
        return "ARK_MODEL"
    if provider == "mimo":
        return "MIMO_MODEL"
    return "JUDGE_MODEL"


def should_judge_example(config: JudgeConfig | None, example: EvalExample) -> bool:
    if config is None:
        return False
    return not example.choices and example.dataset != "pope"


def should_judge_mcq_parse_failure(
    config: JudgeConfig | None,
    example: EvalExample,
    strict_prediction: str | None,
) -> bool:
    if config is None:
        return False
    return bool(example.choices) and strict_prediction is None


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def build_correctness_judge_prompt(example: EvalExample, response: str) -> str:
    return (
        "You are grading a vision-language model answer.\n"
        "Compare the model answer with the ground-truth answer and make a forced "
        "binary decision.\n"
        "Return true only if the model answer is semantically equivalent to the "
        "ground-truth answer.\n"
        "Return false if the model answer contradicts the ground truth, is too "
        "vague, is unrelated, or does not answer.\n"
        "Ignore harmless formatting differences, capitalization, punctuation, "
        "and extra explanatory words when the final meaning is clearly correct.\n"
        "You must return exactly one lowercase boolean token: true or false. "
        "Never return null, unknown, or an empty message.\n"
        "Do not explain. Do not reason step by step. Do not include Markdown. "
        "Do not include any text before or after the boolean token.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Ground-truth answer:\n{example.answer}\n\n"
        f"Model answer:\n{response}\n\n"
        "Return exactly one of these two tokens and nothing else:\n"
        "true\n"
        "false"
    )


def build_mcq_correctness_judge_prompt(
    example: EvalExample,
    response: str,
    expected_answer: str,
) -> str:
    labels = sorted(example.choices)
    choices = "\n".join(
        f"{label}. {example.choices[label]}"
        for label in labels
        if label in example.choices
    )
    label_text = "/".join(labels) if labels else "the option"
    expected_option = example.choices.get(expected_answer)
    ground_truth = (
        f"{expected_answer}. {expected_option}"
        if expected_option
        else expected_answer
    )
    return (
        "You are grading a vision-language model answer to a multiple-choice question.\n"
        f"The local rule-based parser could not extract a {label_text} choice from the "
        "model answer, so you must judge semantic correctness against the ground truth.\n"
        "Return true only if the model answer clearly matches the ground-truth choice "
        "or its meaning, even if it does not explicitly say the option letter.\n"
        "Return false if the model answer chooses another option, is ambiguous, "
        "is too vague, refuses to answer, or does not answer.\n"
        "You must return exactly one lowercase boolean token: true or false.\n"
        "Do not explain. Do not reason step by step. Do not include Markdown.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Ground-truth answer:\n{ground_truth}\n\n"
        f"Model answer:\n{response}\n\n"
        "Return exactly one of these two tokens and nothing else:\n"
        "true\n"
        "false"
    )


def judge_open_response(
    config: JudgeConfig,
    example: EvalExample,
    response: str,
) -> JudgeResult:
    if config.task != "correctness":
        raise ValueError(f"unsupported judge task: {config.task}")

    return request_judge(
        config,
        build_correctness_judge_prompt(example, response),
    )


def judge_mcq_response(
    config: JudgeConfig,
    example: EvalExample,
    response: str,
    expected_answer: str,
) -> JudgeResult:
    return request_judge(
        config,
        build_mcq_correctness_judge_prompt(
            example,
            response,
            expected_answer,
        ),
    )


def request_judge(config: JudgeConfig, user_prompt: str) -> JudgeResult:
    payload = build_judge_payload(config, user_prompt)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "api-key": config.api_key,
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    last_raw_response: str | None = None

    for attempt in range(config.max_retries + 1):
        request = urllib.request.Request(
            config.base_url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as handle:
                response_payload = json.loads(handle.read().decode("utf-8"))
            content = extract_judge_content(response_payload)
            last_raw_response = content
            correct = parse_judge_correctness(content)
            if correct is None:
                raise ValueError("judge response missing explicit correct boolean")
            return JudgeResult(
                prediction=None,
                correct=correct,
                raw_response=content,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {body}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)

        if attempt < config.max_retries:
            time.sleep(config.retry_sleep * (attempt + 1))

    return JudgeResult(
        prediction=None,
        correct=None,
        raw_response=last_raw_response,
        error=last_error,
    )


def build_judge_payload(config: JudgeConfig, user_prompt: str) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict binary evaluator. Reasoning is forbidden. "
                "Explanations are forbidden. Markdown is forbidden. Output "
                "only one lowercase boolean token in the assistant message "
                "content, with no surrounding text."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    if config.provider == "seed":
        return {
            "model": config.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 16,
            "stream": False,
        }
    return {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "top_p": 0.95,
        "max_completion_tokens": 16,
        "stream": False,
        "stop": None,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }


def extract_judge_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("judge response does not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content:
            return content
    if isinstance(content, list):
        content_text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ).strip()
        if content_text:
            return content_text
    for fallback_key in ("reasoning_content", "reasoning"):
        fallback = message.get(fallback_key)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    raise ValueError("judge response content is not text")


def normalize_correctness(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "correct", "yes", "y", "1", "right"}:
        return True
    if text in {"false", "incorrect", "no", "n", "0", "wrong"}:
        return False
    return None


def parse_judge_correctness(content: str) -> bool | None:
    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return normalize_correctness(stripped)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return normalize_correctness(stripped)
    if isinstance(parsed, dict):
        for key in ("correct", "is_correct", "score", "judgement", "judgment"):
            if key in parsed:
                return normalize_correctness(parsed[key])
    return normalize_correctness(parsed)
