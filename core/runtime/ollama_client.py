"""Tiny Ollama HTTP client wrapper using requests.

This is intentionally small and defensive: it tries a couple of common Ollama endpoints
and returns clear statuses instead of raising for connectivity issues.
"""
from typing import Any, Dict, List, Optional
import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        urls = [f"{self.base}/health", f"{self.base}/api/health"]
        for u in urls:
            try:
                r = requests.get(u, timeout=2)
                if r.status_code == 200:
                    return {"ok": True, "url": u, "raw": r.json() if r.headers.get("content-type",""
                                                                                           ).startswith("application/json") else r.text}
            except Exception:
                continue
        return {"ok": False, "url": None}

    def list_models(self) -> Dict[str, Any]:
        urls = [f"{self.base}/v1/models", f"{self.base}/models"]
        for u in urls:
            try:
                r = requests.get(u, timeout=3)
                if r.status_code == 200:
                    try:
                        return {"ok": True, "models": r.json()}
                    except Exception:
                        return {"ok": True, "models": r.text}
            except Exception:
                continue
        return {"ok": False, "models": []}

    def chat(self, model: str, messages: List[Dict[str, str]], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        # Try a couple of common endpoints
        endpoints = [f"{self.base}/v1/chat", f"{self.base}/chat", f"{self.base}/api/chat"]
        payload = {"model": model, "messages": messages}
        payload.update(params)
        for ep in endpoints:
            try:
                r = requests.post(ep, json=payload, timeout=10)
                if r.status_code in (200, 201):
                    try:
                        return {"ok": True, "response": r.json()}
                    except Exception:
                        return {"ok": True, "response": r.text}
                # if 404 or other, try next
            except Exception:
                continue
        return {"ok": False, "error": "unreachable"}
