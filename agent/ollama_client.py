"""Ollama API client for local LLM inference."""

import json
import requests
from typing import Optional


class OllamaClient:
    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                # Try partial match
                matched = [m for m in models if self.model.split(":")[0] in m]
                if matched:
                    self.model = matched[0]
                else:
                    raise RuntimeError(f"Model {self.model} not found. Available: {models}")
        except requests.ConnectionError:
            raise RuntimeError("Ollama not running. Start with: ollama serve")

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """Generate a completion from the local LLM."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if system:
            payload["system"] = system

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096) -> str:
        """Chat completion from the local LLM."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
