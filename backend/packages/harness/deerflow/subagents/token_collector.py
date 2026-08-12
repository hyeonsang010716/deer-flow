"""subagent 안에서 LLM token 사용량을 수집하는 callback handler.

subagent 실행마다 자체 collector를 만든다. subagent가 끝나면 수집된 레코드를
:meth:`RunJournal.record_external_llm_usage_records`를 통해 부모 RunJournal로 넘긴다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class SubagentTokenCollector(BaseCallbackHandler):
    """subagent 안의 LLM token 사용량을 수집하는 가벼운 callback handler."""

    def __init__(self, caller: str):
        super().__init__()
        self.caller = caller
        self._records: list[dict[str, int | str | None]] = []
        self._counted_run_ids: set[str] = set()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid in self._counted_run_ids:
            return

        for generation in response.generations:
            for gen in generation:
                if not hasattr(gen, "message"):
                    continue
                usage = getattr(gen.message, "usage_metadata", None)
                usage_dict = dict(usage) if usage else {}
                input_tk = usage_dict.get("input_tokens", 0) or 0
                output_tk = usage_dict.get("output_tokens", 0) or 0
                total_tk = usage_dict.get("total_tokens", 0) or 0
                if total_tk <= 0:
                    total_tk = input_tk + output_tk
                if total_tk <= 0:
                    continue
                # prompt cache 적중(cache를 반영한 비용 계산에 필요하다)
                details = usage_dict.get("input_token_details") or {}
                cache_read_tk = 0
                if isinstance(details, Mapping):
                    try:
                        cache_read_tk = max(int(details.get("cache_read") or 0), 0)
                    except (TypeError, ValueError):
                        cache_read_tk = 0
                # 이 응답을 실제로 만든 모델을 기록한다. 그래야 부모 journal이 lead agent가
                # 해석한 모델이 아니라 실제 모델 기준으로 token을 분류할 수 있다
                response_metadata = getattr(gen.message, "response_metadata", None) or {}
                model_name: str | None = None
                if isinstance(response_metadata, Mapping):
                    model_name = response_metadata.get("model_name") or response_metadata.get("model")
                self._counted_run_ids.add(rid)
                record: dict[str, int | str | None] = {
                    "source_run_id": rid,
                    "caller": self.caller,
                    "model_name": model_name,
                    "input_tokens": input_tk,
                    "output_tokens": output_tk,
                    "total_tokens": total_tk,
                }
                # journal의 모델별 bucket과 같이 희소하게 둔다. provider가 실제로 cache 적중을
                # 보고했을 때만 키가 존재한다.
                if cache_read_tk > 0:
                    record["cache_read_tokens"] = cache_read_tk
                self._records.append(record)
                return

    def snapshot_records(self) -> list[dict[str, int | str | None]]:
        """누적된 사용량 레코드의 복사본을 반환한다."""
        return list(self._records)
