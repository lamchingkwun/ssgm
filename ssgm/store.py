from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import MemoryRecord
from .versioning import VersionIndex

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text-v2-moe"


class OllamaEmbeddingModel:
    _EMBED_CACHE: Dict[Tuple[str, str, str], List[float]] = {}

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _cache_key(self, text: str) -> Tuple[str, str, str]:
        return (self.base_url, self.model, text)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama embedding response was not valid JSON") from exc

    def _embed_one(self, text: str) -> List[float]:
        cache_key = self._cache_key(text)
        cached = self._EMBED_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            data = self._post_json("/api/embed", {"model": self.model, "input": text})
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                first = embeddings[0]
                if isinstance(first, list):
                    embedding = [float(value) for value in first]
                    self._EMBED_CACHE[cache_key] = embedding
                    return list(embedding)
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                result = [float(value) for value in embedding]
                self._EMBED_CACHE[cache_key] = result
                return list(result)
        except RuntimeError:
            pass

        legacy = self._post_json("/api/embeddings", {"model": self.model, "prompt": text})
        legacy_embedding = legacy.get("embedding")
        if isinstance(legacy_embedding, list):
            result = [float(value) for value in legacy_embedding]
            self._EMBED_CACHE[cache_key] = result
            return list(result)
        raise RuntimeError("Ollama embedding response did not contain an embedding vector")

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for index, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached = self._EMBED_CACHE.get(cache_key)
            if cached is not None:
                results[index] = list(cached)
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        if not missing_texts:
            return [row for row in results if row is not None]

        try:
            data = self._post_json("/api/embed", {"model": self.model, "input": missing_texts})
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings and all(isinstance(row, list) for row in embeddings):
                for index, text, row in zip(missing_indices, missing_texts, embeddings):
                    embedding = [float(value) for value in row]
                    self._EMBED_CACHE[self._cache_key(text)] = embedding
                    results[index] = list(embedding)
                return [row for row in results if row is not None]
            embedding = data.get("embedding")
            if isinstance(embedding, list) and embedding and all(isinstance(row, list) for row in embedding):
                for index, text, row in zip(missing_indices, missing_texts, embedding):
                    embedding_row = [float(value) for value in row]
                    self._EMBED_CACHE[self._cache_key(text)] = embedding_row
                    results[index] = list(embedding_row)
                return [row for row in results if row is not None]
        except RuntimeError:
            pass

        # Legacy / partial compatibility path: fall back to per-item requests.
        for index, text in zip(missing_indices, missing_texts):
            results[index] = self._embed_one(text)
        return [row for row in results if row is not None]

    def encode(self, text: Any, convert_to_numpy: bool = True):
        if isinstance(text, (list, tuple)):
            embeddings = self._embed_many([str(item) for item in text])
            if not convert_to_numpy:
                return embeddings
            import numpy as np

            return np.array(embeddings)

        embedding = self._embed_one(str(text))
        if not convert_to_numpy:
            return embedding
        import numpy as np

        return np.array(embedding)


class SemanticStore:
    _MODEL_CACHE: Dict[Tuple[str, str], OllamaEmbeddingModel] = {}

    def __init__(
        self,
        embedding_model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        use_embeddings: bool = True,
        base_url: Optional[str] = None,
    ) -> None:
        self.records: Dict[str, MemoryRecord] = {}
        self.version_index = VersionIndex()
        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model
        self.embedding_backend = "ollama"
        self.embedding_base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self._model: Any = None
        self._model_load_failed = False
        self._record_embeddings: Dict[str, Any] = {}

    def _get_model(self) -> OllamaEmbeddingModel:
        cache_key = (self.embedding_base_url, self.embedding_model_name)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        model = OllamaEmbeddingModel(model=self.embedding_model_name, base_url=self.embedding_base_url)
        self._MODEL_CACHE[cache_key] = model
        return model

    def _maybe_encode(self, text: str):
        if not self.use_embeddings or self._model_load_failed:
            return None
        if self._model is None:
            try:
                self._model = self._get_model()
            except Exception:
                self._model_load_failed = True
                return None
        try:
            return self._model.encode(str(text), convert_to_numpy=True)
        except Exception:
            self._model_load_failed = True
            return None

    def get(self, key: str) -> Optional[MemoryRecord]:
        return self.records.get(key)

    def upsert(self, record: MemoryRecord) -> None:
        self.records[record.key] = record
        self.version_index.append(record)
        emb = self._maybe_encode(record.value)
        if emb is not None:
            self._record_embeddings[record.key] = emb

    def list_all(self) -> List[MemoryRecord]:
        return list(self.records.values())

    def list_visible(self, tenant_id: str) -> List[MemoryRecord]:
        return [r for r in self.records.values() if r.tenant_id == tenant_id and r.status == 'active']

    def rollback(self, key: str, version: int) -> Optional[MemoryRecord]:
        target = self.version_index.rollback_target(key, version)
        if target is None:
            return None
        self.records[key] = target
        emb = self._maybe_encode(target.value)
        if emb is not None:
            self._record_embeddings[key] = emb
        return target

    def semantic_search(self, query: str, tenant_id: Optional[str] = None, top_k: int = 5) -> List[MemoryRecord]:
        candidates = list(self.records.values())
        if tenant_id is not None:
            candidates = [r for r in candidates if r.tenant_id == tenant_id and r.status == 'active']
        else:
            candidates = [r for r in candidates if r.status == 'active']

        if not candidates:
            return []

        query_emb = self._maybe_encode(query)
        if query_emb is None:
            return sorted(candidates, key=lambda x: x.timestamp, reverse=True)[:top_k]

        import numpy as np
        norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        scores = []
        for record in candidates:
            emb = self._record_embeddings.get(record.key)
            if emb is not None:
                norm_emb = emb / (np.linalg.norm(emb) + 1e-10)
                sim = max(0.0, np.dot(norm_q, norm_emb))
            else:
                sim = 0.0
            scores.append((sim, record))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scores[:top_k]]
