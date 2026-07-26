from __future__ import annotations

import argparse
import os
import time


def call_ark(args: argparse.Namespace, image_data_url: str, prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Ark API key environment variable: {args.api_key_env}")
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Ark returned an empty response.")
            return str(content).strip()
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"Ark request failed: {last_error}") from last_error
