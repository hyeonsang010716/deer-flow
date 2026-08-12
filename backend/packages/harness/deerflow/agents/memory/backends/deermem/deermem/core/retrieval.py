"""DeerMem용 FTS5 기반 retrieval 엔진.

저장된 fact에 대해 BM25 전문 검색을 제공하며 다음을 지원한다.

- jieba 중국어 tokenization (선택적, 없으면 공백 분리로 fallback)
- FTS5 MATCH 문법(AND/OR/NOT/구문/접두어) 지원과 fallback
- 시간 감쇠 + confidence 가중 랭킹
- category 필터링
- scope(user_id) 격리

``FTS5Retrieval``은 저수준 SQLite 엔진이다. storage 연동은
``FTS5RetrievalAdapter``가 맡으며, storage를 import하지 않고
``storage.RetrievalPort``를 구현해 순환 의존을 피한다.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 점수 가중치 ──────────────────────────────────────────────────
#
# SQLite FTS5의 ``bm25(memory_fts)``는 위치 인자 없이 쓰면 기본값 K1=1.2, B=0.75가
# 적용되며, 문서 관련도에 비례한 크기의 음수를 반환한다. 중요한 점은 이 함수의 인자
# 순서가 ``(table, k1, b, *column_weights)``라는 것이다. 원래 코드는
# ``bm25(..., 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)``을 넘겨 **k1 = 0**으로 만들었고,
# 그 결과 BM25 점수 전체가 조용히 0이 되어(tf saturation 비활성화) 랭킹이
# ``confidence * _CONFIDENCE_WEIGHT``로 붕괴했다. 인자 없는 형태를 써야 SQLite 기본값이
# 적용되고 BM25가 실제로 점수에 반영된다.
_CONFIDENCE_WEIGHT = 0.2


# ── jieba (선택적) ──────────────────────────────────────────────────

try:
    import jieba

    _jieba_available = True
except ImportError:
    _jieba_available = False


def _tokenize(text: str) -> list[str]:
    """텍스트를 토큰화한다. 중국어는 jieba, 영어는 공백 분리를 쓴다."""
    if not text or not text.strip():
        return []
    if _jieba_available:
        return [t for t in jieba.cut(text) if t.strip()]
    return [t for t in text.split() if t.strip()]


# ── FTS5 질의 전처리 ──────────────────────────────────────────

_FTS5_ADVANCED_RE = re.compile(
    r"(\bAND\b|\bOR\b|\bNOT\b|\bNEAR\b"
    r'|"\w.*?"'  # 구문 "..."
    r"|\w+\*"  # 접두어 prefix*
    r"|\(.*?\))"  # 그룹 (...)
)


def _is_advanced_query(query: str) -> bool:
    """질의가 FTS5 고급 문법을 쓰는지 판별한다."""
    return bool(_FTS5_ADVANCED_RE.search(query))


def _build_fallback_query(query: str) -> str:
    """자연어 질의를 FTS5 OR 질의로 변환한다(fallback 전략)."""
    tokens = [token for token in _tokenize(query) if any(char.isalnum() for char in token)]
    if not tokens:
        return ""
    # 토큰마다 따옴표를 씌워, 자연어 입력의 문장부호가 FTS5 연산자가 되거나 문법 오류를
    # 내지 않게 한다. 토큰 안의 큰따옴표 두 개는 리터럴 따옴표를 뜻하는 FTS5 escape다.
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


# ── 핵심 retrieval 엔진 ─────────────────────────────────────────────


class FTS5Retrieval:
    """SQLite FTS5 기반 retrieval 엔진.

    질의 전략:
      1. FTS5 고급 문법이면 MATCH로 그대로 넘긴다.
      2. 자연어면 jieba로 토큰화한 뒤 OR로 잇는다.
      3. 문법 오류가 나면 토큰화 OR 질의로 fallback한다.
      4. 그래도 실패하면 빈 결과를 반환한다.

    랭킹:
      BM25 점수 × time_decay + confidence × 0.2
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        # Gateway는 asyncio.to_thread / ThreadPoolExecutor로 도구 호출을 실행한다.
        # 최상위 인스턴스가 단일 프로세스여도 SQLite 연결을 스레드 간에 공유하는 것은
        # 안전하지 않아 방어 계층을 둘 둔다.
        #   1. ``check_same_thread=False``로 스레드 A에서 만든 연결을 스레드 B에서도
        #      쓸 수 있게 한다(libsqlite 자체는 직렬화 래퍼 아래서 재진입 가능하다.
        #      #4208의 hot path 논의 참고).
        #   2. ``_lock``이 모든 변경성 sqlite 호출을 감싸 동시 호출자를 직렬화한다
        #      (쓰기 교차와 FTS5 인덱스 재정렬을 막는다). 인스턴스 메서드 밖의 호출자는
        #      반드시 공개 API를 통해 lock을 거쳐야 하며 ``self._conn``에 직접 접근하면
        #      안 된다.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = threading.RLock()
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        conn = self._conn
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                doc_id UNINDEXED,
                content,
                raw_content UNINDEXED,
                category UNINDEXED,
                scope_user UNINDEXED,
                scope_agent UNINDEXED,
                created_at UNINDEXED,
                confidence UNINDEXED,
                source UNINDEXED,
                fact_json UNINDEXED,
                tokenize='unicode61'
            )
            """
        )
        conn.commit()

    # ── 인덱스 연산 ───────────────────────────────────────────────

    def _preprocess_content(self, content: str) -> str:
        """색인용으로 content를 전처리한다. 중국어는 jieba로 토큰화한다."""
        if not content:
            return ""
        if _jieba_available:
            tokens = _tokenize(content)
            return " ".join(tokens)
        return content

    def _row_from_document(self, document: dict[str, Any]) -> tuple[Any, ...]:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return (
            document["fact_id"],
            self._preprocess_content(document["content"]),
            document["content"],
            document["category"],
            document["scope_user"],
            document["scope_agent"],
            document.get("created_at") or now,
            document.get("confidence", 0.5),
            document.get("source"),
            json.dumps(document.get("fact_data"), ensure_ascii=False, default=str),
        )

    def index_fact(
        self,
        fact_id: str,
        content: str,
        category: str = "context",
        confidence: float = 0.5,
        created_at: str | None = None,
        scope_user: str | None = None,
        scope_agent: str | None = None,
        source: str | None = None,
        fact_data: dict[str, Any] | None = None,
    ) -> None:
        """FTS5 인덱스에 fact를 삽입하거나 갱신한다."""
        with self._lock:
            conn = self._conn
            # 같은 doc_id의 기존 항목을 지운다(FTS5에서의 INSERT OR REPLACE 대용)
            conn.execute("DELETE FROM memory_fts WHERE doc_id = ?", (fact_id,))
            conn.execute(
                """
                INSERT INTO memory_fts(
                    doc_id, content, raw_content, category, scope_user, scope_agent,
                    created_at, confidence, source, fact_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._row_from_document(
                    {
                        "fact_id": fact_id,
                        "content": content,
                        "category": category,
                        "scope_user": scope_user or "",
                        "scope_agent": scope_agent or "",
                        "created_at": created_at,
                        "confidence": confidence,
                        "source": source,
                        "fact_data": fact_data,
                    }
                ),
            )
            conn.commit()

    def replace_documents(self, documents: list[dict[str, Any]], *, scopes: list[tuple[str, str]] | None = None) -> None:
        """전체 또는 지정한 scope의 행을 한 트랜잭션에서 원자적으로 교체한다."""
        with self._lock:
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                if scopes is None:
                    conn.execute("DELETE FROM memory_fts")
                else:
                    conn.executemany(
                        "DELETE FROM memory_fts WHERE scope_user = ? AND scope_agent = ?",
                        scopes,
                    )
                conn.executemany(
                    """
                    INSERT INTO memory_fts(
                        doc_id, content, raw_content, category, scope_user, scope_agent,
                        created_at, confidence, source, fact_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [self._row_from_document(document) for document in documents],
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def remove_fact(self, fact_id: str) -> None:
        """FTS5 인덱스에서 fact를 제거한다."""
        with self._lock:
            conn = self._conn
            conn.execute("DELETE FROM memory_fts WHERE doc_id = ?", (fact_id,))
            conn.commit()

    def clear_index(self) -> None:
        """FTS5 인덱스 전체를 비운다."""
        with self._lock:
            conn = self._conn
            conn.execute("DELETE FROM memory_fts")
            conn.commit()

    def clear_scope(self, *, scope_user: str, scope_agent: str) -> None:
        """다른 사용자에 영향을 주지 않고 지정한 adapter scope 하나만 비운다."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_fts WHERE scope_user = ? AND scope_agent = ?",
                (scope_user, scope_agent),
            )
            self._conn.commit()

    def rebuild_from_facts(
        self,
        facts: list[dict[str, Any]],
        *,
        scope_user: str | None = None,
        scope_agent: str | None = None,
    ) -> None:
        """fact dict 목록으로 인덱스 전체를 다시 만든다."""
        with self._lock:
            conn = self._conn
            try:
                conn.execute("BEGIN")
                conn.execute("DELETE FROM memory_fts")
                for fact in facts:
                    fact_id = fact.get("id", "")
                    content = fact.get("content", "")
                    if not fact_id or not isinstance(content, str) or not content:
                        continue
                    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                    conn.execute(
                        """
                        INSERT INTO memory_fts(
                            doc_id, content, raw_content, category, scope_user, scope_agent,
                            created_at, confidence, source, fact_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fact_id,
                            self._preprocess_content(content),
                            content,
                            fact.get("category", "context"),
                            scope_user or "",
                            scope_agent or "",
                            fact.get("createdAt") or now,
                            fact.get("confidence", 0.5),
                            fact.get("source"),
                            json.dumps(fact, ensure_ascii=False, default=str),
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── 검색 ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        scope_user: str | None = None,
        scope_agent: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """category와 scope 필터를 적용한 FTS5 BM25 검색을 수행한다.

        관련도 순으로 정렬된 fact dict 목록(id, content, category, confidence,
        createdAt, source, score, bm25_score)을 반환한다.
        """
        import time

        _t0 = time.perf_counter()
        if not query or not query.strip() or top_k <= 0:
            logger.debug("FTS5Retrieval.search: skipped (empty/invalid) query=%r top_k=%d", query, top_k)
            return []

        query = query.strip()

        # 질의 전략 결정
        if _is_advanced_query(query):
            fts5_query = query
            strategy = "advanced"
        else:
            fts5_query = _build_fallback_query(query)
            strategy = "tokenized"
        logger.debug(
            "FTS5Retrieval.search: query=%r strategy=%r fts5_query=%r scope_user=%r scope_agent=%r category=%r top_k=%d",
            query,
            strategy,
            fts5_query,
            scope_user,
            scope_agent,
            category,
            top_k,
        )

        # 검색 시도
        results = self._execute_search(fts5_query, scope_user, scope_agent, category, top_k)
        if results is not None:
            logger.debug(
                "FTS5Retrieval.search: strategy=%r returned %d results in %.1fms",
                strategy,
                len(results),
                (time.perf_counter() - _t0) * 1000,
            )
            return results

        # fallback: 고급 문법 오류면 토큰화 OR 질의로 재시도한다
        if strategy == "advanced":
            fallback = _build_fallback_query(query)
            if fallback and fallback != fts5_query:
                logger.debug("FTS5Retrieval.search: advanced syntax failed, retrying with tokenized fts5_query=%r", fallback)
                results = self._execute_search(fallback, scope_user, scope_agent, category, top_k)
                if results is not None:
                    logger.debug(
                        "FTS5Retrieval.search: tokenized fallback returned %d results in %.1fms",
                        len(results),
                        (time.perf_counter() - _t0) * 1000,
                    )
                    return results

        logger.debug(
            "FTS5Retrieval.search: returning [] (no path produced results) in %.1fms",
            (time.perf_counter() - _t0) * 1000,
        )
        return []

    def _execute_search(
        self,
        fts5_query: str,
        scope_user: str | None,
        scope_agent: str | None,
        category: str | None,
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        """FTS5 질의를 실행한다. 문법 오류면 None을 반환한다."""
        if not fts5_query:
            return []

        conditions = ["memory_fts MATCH ?"]
        params: list[Any] = [fts5_query]

        if scope_user:
            conditions.append("scope_user = ?")
            params.append(scope_user)
        if scope_agent:
            conditions.append("scope_agent = ?")
            params.append(scope_agent)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT doc_id, content, raw_content, category, scope_user, scope_agent,
                   created_at, confidence, source, fact_json,
                   bm25(memory_fts) AS bm25_score
            FROM memory_fts
            WHERE {where_clause}
            ORDER BY bm25_score
            LIMIT ?
        """
        params.append(top_k * 2)

        with self._lock:
            try:
                conn = self._conn
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                logger.debug("FTS5 query syntax error: %s (query: %s)", e, fts5_query)
                return None

        results: list[dict[str, Any]] = []
        for row in rows:
            (
                doc_id,
                indexed_content,
                raw_content,
                cat,
                s_user,
                s_agent,
                created_at,
                confidence,
                source,
                fact_json,
                bm25_score,
            ) = row

            score = self._compute_final_score(
                bm25_score=-bm25_score,  # FTS5는 음수를 반환한다
                confidence=confidence,
                created_at=created_at,
            )

            fact: dict[str, Any] = {}
            if fact_json:
                try:
                    decoded = json.loads(fact_json)
                    if isinstance(decoded, dict):
                        fact = decoded
                except (TypeError, ValueError):
                    logger.debug("FTS5 fact metadata was not valid JSON for doc_id=%r", doc_id)
            fact.setdefault("id", doc_id)
            fact.setdefault("content", raw_content if raw_content is not None else indexed_content)
            fact.setdefault("category", cat)
            fact.setdefault("confidence", confidence)
            fact.setdefault("createdAt", created_at)
            fact.setdefault("source", source if source is not None else "fts5")
            fact["score"] = score
            fact["bm25_score"] = -bm25_score
            results.append(fact)

        logger.debug(
            "FTS5 raw SQL: fts5_query=%r scope_user=%r scope_agent=%r category=%r -> %d rows. bm25 raw: %s",
            fts5_query,
            scope_user,
            scope_agent,
            category,
            len(rows),
            [(r["id"], r["bm25_score"]) for r in results[:10]],
        )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _compute_final_score(
        self,
        bm25_score: float,
        confidence: float,
        created_at: str,
    ) -> float:
        """BM25 × time_decay + confidence 가중치를 합친 최종 점수를 계산한다.

        SQLite FTS5 관례상 ``bm25_score``는 관련 문서일수록 음수다. 호출자가 결과 dict에
        저장하기 전에 부호를 뒤집으므로(``-bm25_score``) 여기서는 양수 관련도 크기로
        취급한다.
        """
        score = bm25_score

        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = (datetime.now(UTC) - dt).days
            time_decay = 1.0 if age_days < 30 else math.exp(-0.01 * (age_days - 30))
            score *= time_decay
        except (AttributeError, ValueError, TypeError):
            time_decay = -1.0  # "파싱 불가"를 뜻하는 sentinel. 감쇠를 건너뛴다
            age_days = -1

        try:
            normalized_confidence = float(confidence)
            if not math.isfinite(normalized_confidence):
                raise ValueError
        except (TypeError, ValueError):
            normalized_confidence = 0.5
        score += normalized_confidence * _CONFIDENCE_WEIGHT

        logger.debug(
            "_compute_final_score: bm25_in=%.4f time_decay=%s age_days=%s conf=%.2f -> final=%.4f",
            bm25_score,
            time_decay,
            age_days,
            normalized_confidence,
            score,
        )
        return score

    # ── 통계 ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """인덱스 통계를 반환한다."""
        with self._lock:
            conn = self._conn
            total = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        return {
            "total_docs": total,
            "jieba": _jieba_available,
            "db_path": self._db_path,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _scope_value(value: str | None) -> str:
    """``None``이 user id와 충돌하지 않도록 scope 값을 타입까지 담아 인코딩한다."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _scope_key(scope: dict[str, str | None]) -> tuple[str, str]:
    user_id = scope.get("userId")
    agent_name = scope.get("agentName")
    if user_id is not None and not isinstance(user_id, str):
        raise ValueError("retrieval scope userId must be a string or null")
    if agent_name is not None and not isinstance(agent_name, str):
        raise ValueError("retrieval scope agentName must be a string or null")
    return _scope_value(user_id), _scope_value(agent_name)


class FTS5RetrievalAdapter:
    """SQLite FTS5 DB 하나를 쓰는 scope 인지 ``RetrievalPort`` adapter.

    인덱스는 파생 데이터다. 정본 fact는 Markdown에 남고 storage 알림은 지정된 행만
    갱신한다. 결정적인 복합 document id를 써서 서로 다른 user/agent scope의 동일한
    fact id가 서로를 덮어쓰지 않게 한다.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._engine = FTS5Retrieval(db_path)

    @staticmethod
    def _document_id(fact_id: str, scope: dict[str, str | None]) -> str:
        scope_user, scope_agent = _scope_key(scope)
        return json.dumps([scope_user, scope_agent, fact_id], ensure_ascii=False, separators=(",", ":"))

    def _document(self, fact: dict[str, Any], scope: dict[str, str | None]) -> dict[str, Any]:
        fact_id = fact.get("id")
        content = fact.get("content")
        if not isinstance(fact_id, str) or not fact_id:
            raise ValueError("retrieval fact.id must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("retrieval fact.content must be a non-empty string")
        scope_user, scope_agent = _scope_key(scope)
        payload = dict(fact)
        payload["scope"] = {"userId": scope.get("userId"), "agentName": scope.get("agentName")}
        source = payload.get("source")
        return {
            "fact_id": self._document_id(fact_id, scope),
            "content": content,
            "category": str(payload.get("category") or "context"),
            "confidence": float(payload.get("confidence") or 0.5),
            "created_at": payload.get("createdAt") if isinstance(payload.get("createdAt"), str) else None,
            "scope_user": scope_user,
            "scope_agent": scope_agent,
            "source": source if isinstance(source, str) else json.dumps(source, ensure_ascii=False, default=str),
            "fact_data": payload,
        }

    def upsert(self, fact: dict[str, Any], *, scope: dict[str, str | None], path: str) -> None:
        del path  # 정본 위치는 storage 소관이고 인덱스는 언제든 다시 만들 수 있다.
        document = self._document(fact, scope)
        self._engine.index_fact(**document)

    def rebuild(self, records: list[tuple[dict[str, Any], dict[str, str | None], str]], *, scopes: list[dict[str, str | None]] | None) -> None:
        """storage 재구축이 지정한 레코드를 원자적으로 교체한다."""
        documents = [self._document(fact, scope) for fact, scope, _path in records]
        encoded_scopes = None
        if scopes is not None:
            encoded_scopes = [_scope_key(scope) for scope in scopes]
        self._engine.replace_documents(documents, scopes=encoded_scopes)

    def remove(self, fact_id: str, *, scope: dict[str, str | None]) -> None:
        self._engine.remove_fact(self._document_id(fact_id, scope))

    def search(
        self,
        query: str,
        *,
        scopes: list[dict[str, str | None]],
        top_k: int,
        mode: str,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        if mode not in {"hybrid", "fts5", "lexical"}:
            raise ValueError(f"unsupported FTS5 retrieval mode: {mode}")

        filters = filters or {}
        category = filters.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError("retrieval category filter must be a string")

        results: list[dict[str, Any]] = []
        per_scope_limit = top_k * 4
        for scope in scopes:
            scope_user, scope_agent = _scope_key(scope)
            for candidate in self._engine.search(
                query,
                scope_user=scope_user,
                scope_agent=scope_agent,
                category=category,
                top_k=per_scope_limit,
            ):
                fact = dict(candidate)
                score = float(fact.pop("score", 0.0))
                bm25_score = float(fact.pop("bm25_score", 0.0))
                if any(fact.get(key) != value for key, value in filters.items()):
                    continue
                results.append(
                    {
                        "fact": fact,
                        "score": score,
                        "matchType": "fts5",
                        "retrieval": {"bm25": bm25_score},
                    }
                )

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]

    def clear(self, *, scopes: list[dict[str, str | None]] | None = None) -> None:
        if scopes is None:
            self._engine.clear_index()
            return
        for scope in scopes:
            scope_user, scope_agent = _scope_key(scope)
            self._engine.clear_scope(scope_user=scope_user, scope_agent=scope_agent)

    def stats(self) -> dict[str, Any]:
        return self._engine.stats()

    def close(self) -> None:
        self._engine.close()


def create_fts5_retrieval(config: Any) -> FTS5RetrievalAdapter | None:
    """영속 파생 인덱스를 쓰는 DeerMem 번들 adapter를 만든다.

    storage 루트가 설정되지 않은 단독 ``DeerMem`` 인스턴스는 in-memory 인덱스를 쓴다.
    host factory는 항상 절대 storage 루트를 주입하므로, 일반 Gateway 인스턴스는 그
    아래에 재구축 가능한 인덱스를 영속화한다.
    """
    storage_path = str(getattr(config, "storage_path", "") or "")
    if not storage_path:
        db_path: str | Path = ":memory:"
    else:
        index_dir = Path(storage_path) / ".retrieval"
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "memory-fts5.sqlite3"
    try:
        return FTS5RetrievalAdapter(db_path)
    except sqlite3.DatabaseError as exc:
        if db_path == ":memory:":
            logger.warning("SQLite FTS5 is unavailable; DeerMem will use substring retrieval: %s", exc)
            return None

        logger.warning("Derived memory retrieval index is invalid; recreating it: %s", exc)
        try:
            for path in (Path(db_path), Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
                path.unlink(missing_ok=True)
            return FTS5RetrievalAdapter(db_path)
        except (OSError, sqlite3.DatabaseError) as retry_exc:
            logger.warning(
                "SQLite FTS5 index recreation failed; DeerMem will use substring retrieval: %s",
                retry_exc,
            )
            return None
