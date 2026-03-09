"""Unified LLM client supporting Gemini and Claude."""

from __future__ import annotations

import json
import logging
import re
from typing import Type, TypeVar

from pydantic import BaseModel

from alphalab.config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified LLM client with retry and provider fallback."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._gemini_client = None
        self._anthropic_client = None

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client(api_key=self.config.gemini_api_key)
        return self._gemini_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(
                api_key=self.config.anthropic_api_key,
            )
        return self._anthropic_client

    def generate(
        self,
        prompt: str,
        system: str = "",
        provider: str | None = None,
    ) -> str:
        provider = provider or self.config.primary_provider

        for attempt in range(self.config.max_retries):
            try:
                if provider == "gemini":
                    return self._generate_gemini(prompt, system)
                elif provider == "claude":
                    return self._generate_claude(prompt, system)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except Exception as e:
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                if attempt == self.config.max_retries - 1:
                    fallback = "claude" if provider == "gemini" else "gemini"
                    logger.info("Falling back to %s", fallback)
                    try:
                        if fallback == "gemini":
                            return self._generate_gemini(prompt, system)
                        return self._generate_claude(prompt, system)
                    except Exception as fb_err:
                        raise RuntimeError(
                            f"Both providers failed. "
                            f"Primary ({provider}): {e}, Fallback ({fallback}): {fb_err}"
                        ) from fb_err

        raise RuntimeError("LLM generation failed after all retries")

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str = "",
        provider: str | None = None,
    ) -> T:
        schema_hint = (
            "\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(response_model.model_json_schema(), indent=2)}\n"
        )
        raw = self.generate(prompt + schema_hint, system, provider)
        json_str = self._extract_json(raw)
        return response_model.model_validate_json(json_str)

    def _generate_gemini(self, prompt: str, system: str) -> str:
        client = self._get_gemini_client()
        config: dict = {"temperature": self.config.temperature}
        if system:
            config["system_instruction"] = system
        response = client.models.generate_content(
            model=self.config.gemini_model,
            contents=prompt,
            config=config,
        )
        return response.text

    def _generate_claude(self, prompt: str, system: str) -> str:
        client = self._get_anthropic_client()
        kwargs: dict = {
            "model": self.config.anthropic_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    @staticmethod
    def _extract_json(text: str) -> str:
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                return text[start : end + 1]
        return text.strip()
