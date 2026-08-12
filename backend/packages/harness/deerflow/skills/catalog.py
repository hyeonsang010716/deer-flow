"""skill catalog — runtime의 deferred skill discovery.

``tool_search.py``의 ``DeferredToolCatalog``와 같은 구조다. 모든 skill의 전체 설명을
system prompt에 박아 넣는 대신, LLM이 필요할 때 skill metadata를 발견할 수 있는
불변 검색 catalog를 제공한다.

agent는 ``<skill_index>``에서 skill 이름만 보고, ``describe_skill``을 호출하기 전에는
metadata를 읽을 수 없다. 덕분에 system prompt는 짧고 prefix-cache에 유리하면서도
모델은 자율적으로 skill을 탐색할 수 있다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property

from deerflow.skills.types import Skill

logger = logging.getLogger(__name__)

MAX_RESULTS = 5


def _compile_catalog_regex(pattern: str) -> re.Pattern[str]:
    """``pattern``을 대소문자 무시로 컴파일하고, 실패하면 리터럴 매칭으로 fallback한다.

    검색 쿼리는 모델이 만들기 때문에, 잘못된 regex(예: 괄호 불일치)는 예외를 던지지 말고
    리터럴 substring 매칭으로 낮춰야 한다.
    """
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


# NOTE: slots=True 없이 frozen=True를 쓰면 __dict__가 남고, 그래야 아래 @cached_property
# 필드가 캐싱된다(frozen __setattr__을 우회해 instance.__dict__에 직접 쓴다).
# slots=True를 추가하면 안 된다. runtime에 hash/names가 깨진다.
@dataclass(frozen=True)
class SkillCatalog:
    """skill의 불변 catalog. 검색만 하고 변경은 하지 않는다.

    쿼리 형태(``DeferredToolCatalog.search``와 동일):

    - ``"select:data-analysis,deep-research"`` — 이름 정확 일치.
    - ``"+podcast gen"`` — 이름에 *podcast*를 요구하고, *gen*으로 랭킹한다.
    - ``"chart visualization"`` — 이름 + 설명에 대한 regex 매칭.
    """

    skills: tuple[Skill, ...]

    @cached_property
    def names(self) -> frozenset[str]:
        """삽입 순서를 따르는 전체 skill 이름."""
        return frozenset(s.name for s in self.skills)

    def search(self, query: str) -> list[Skill]:
        """*query*를 skill 이름과 설명에 매칭한다.

        관련도 순으로 최대 ``MAX_RESULTS``개의 skill을 반환한다.
        """
        query = query.strip()
        if not query:
            return []

        # ── 정확 선택 ──────────────────────────────────────────────────
        if query.startswith("select:"):
            wanted = {n.strip() for n in query[7:].split(",")}
            return [s for s in self.skills if s.name in wanted]

        # ── 필수 prefix 검색 ───────────────────────────────────────────
        if query.startswith("+"):
            parts = query[1:].split(None, 1)
            if not parts:
                return []  # 필수 토큰 없이 "+"만 있는 경우
            required = parts[0].lower()
            candidates = [s for s in self.skills if required in s.name.lower()]
            if len(parts) > 1:
                pattern = _compile_catalog_regex(parts[1])
                candidates.sort(
                    key=lambda s: _catalog_regex_score(pattern, s),
                    reverse=True,
                )
            return candidates[:MAX_RESULTS]

        # ── 자유 텍스트 regex 검색 ─────────────────────────────────────
        regex = _compile_catalog_regex(query)
        scored: list[tuple[int, Skill]] = []
        for s in self.skills:
            searchable = f"{s.name} {s.description or ''}"
            if regex.search(searchable):
                # 이름 매칭이 설명만 매칭된 경우보다 높은 점수를 받는다.
                scored.append((2 if regex.search(s.name) else 1, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored][:MAX_RESULTS]


def _catalog_regex_score(pattern: re.Pattern[str], s: Skill) -> int:
    """랭킹을 위해 이름 + 설명에서 regex 히트 수를 센다."""
    return len(pattern.findall(f"{s.name} {s.description or ''}"))
