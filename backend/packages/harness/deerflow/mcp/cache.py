"""MCP 도구를 반복 로딩하지 않도록 캐싱한다."""

import asyncio
import logging
from pathlib import Path

from langchain_core.tools import BaseTool

from deerflow.config.file_signature import ConfigSignature as _ConfigSignature
from deerflow.config.file_signature import get_config_signature as _get_config_signature

logger = logging.getLogger(__name__)

_mcp_tools_cache: list[BaseTool] | None = None
_cache_initialized = False
_initialization_lock = asyncio.Lock()

# resolve된 extensions config 파일에 대한 캐시 무효화 키. mtime만 보지 않고 resolve된 경로와
# ``(mtime, size, sha256)`` 콘텐츠 signature를 함께 추적한다 — signature는
# ``deerflow.config.app_config``가 런타임 편집 가능한 형제 config 파일에 쓰는 것과 같은 공용
# ``deerflow.config.file_signature`` 헬퍼로 계산한다. mtime ``>`` 비교만으로는 같은 초 안의 편집과
# mtime이 그대로거나 뒤로 가는 경우(object-store / network 마운트, ``git checkout``,
# ``cp -p`` / 백업 복원, 타임스탬프를 보존하는 ``tar`` / ``rsync``)를 놓치고, 경로를 전혀 추적하지
# 않으면 mtime이 같거나 더 오래된 다른 config 파일로 전환한 것을 구조적으로 감지할 수 없다.
_config_path: Path | None = None  # init 시점에 resolve된 extensions config 경로
_config_signature: _ConfigSignature | None = None  # init 시점의 (mtime, size, sha256)


def _resolve_config_path() -> Path | None:
    """extensions config 파일 경로를 resolve한다. 설정되지 않았으면 ``None``을 반환한다.

    ``ExtensionsConfig.resolve_config_path()``는 명시적 `config_path`나
    `DEER_FLOW_EXTENSIONS_CONFIG_PATH`가 존재하지 않는 파일을 가리키면
    ``FileNotFoundError``를 던진다. config를 실제로 사용하려고 로드하는 호출자
    (예: ``get_mcp_tools()``를 통한 ``ExtensionsConfig.from_file()``)에게는 의도된 동작이다.
    운영자가 명시한 경로가 사라진 것은 실제 설정 오류이므로 크게 드러내야 한다.

    이 헬퍼는 그런 호출자가 아니다. 캐시 자체의 staleness 검사
    (``_current_config_state``를 통한 ``_is_cache_stale``)만 뒷받침하며, 이 검사는
    ``get_cached_mcp_tools()`` 호출마다 실행되어 이전에 로드한 config가 여전히 최신인지만
    확인한다. 이전에 유효했던 명시적/env-var 경로의 파일이 나중에 읽을 수 없게 되면
    (실행 중 삭제, Docker 마운트 문제 등) 여기서 예외를 던지는 것은 캐시가 마지막 정상 MCP
    도구를 계속 제공하게 두는 대신, 요청마다 도는 hot path의 이후 모든 호출을 실패시킨다.
    그래서 이 wrapper는 그 특정 실패만 잡아 "설정되지 않음"과 동일하게 취급하며, 이는
    ``_is_cache_stale()``이 ``None`` config state를 fail-soft로 다루는 기존 방식과 일치한다
    (해당 docstring 참고). ``resolve_config_path()`` 자체가 모든 호출자에게 ``None``을
    반환하게 만들지 않고 여기서만 catch 범위를 좁힘으로써, 파일이 실제로 필요한 호출자에게는
    시끄러운 실패가 그대로 유지된다.
    """
    from deerflow.config.extensions_config import ExtensionsConfig

    try:
        return ExtensionsConfig.resolve_config_path()
    except FileNotFoundError:
        logger.debug(
            "Extensions config path could not be resolved while checking MCP cache staleness; treating as unconfigured for this check.",
            exc_info=True,
        )
        return None


def _current_config_state() -> tuple[Path | None, _ConfigSignature | None]:
    """현재 resolve된 extensions config 경로와 그 signature를 반환한다."""
    config_path = _resolve_config_path()
    if config_path is None:
        return None, None
    return config_path, _get_config_signature(config_path)


def _is_cache_stale() -> bool:
    """config 파일 변경으로 캐시가 stale해졌는지 확인한다.

    resolve된 extensions config 경로가 바뀌었거나, ``(mtime, size, sha256)`` 콘텐츠
    signature가 초기화 시점에 기록한 값과 다르면 캐시는 stale이다. 엄격한 mtime ``>`` 비교
    대신 콘텐츠 동등성(``!=``)을 쓰면 같은 초 안의 편집과 mtime이 뒤로 가는 경우를 감지하고,
    resolve된 경로를 추적하면 다른 config 파일로의 전환을 감지한다.

    Returns:
        캐시를 무효화해야 하면 True, 아니면 False.
    """
    if not _cache_initialized:
        return False  # 아직 초기화 전이므로 stale이 아니다

    current_path, current_signature = _current_config_state()

    # 기존의 "config 없음 / 아직 기록 안 됨" 동작을 유지한다. 캐시를 채울 때 읽을 수 있는
    # config가 없었거나 지금 없으면 무효화하지 않는다. init 성공 후 config가 완전히 삭제된
    # 경우(current_signature가 None이 되는 경우)도 여기에 포함된다. 캐시는 설정되지 않은 상태로
    # 무효화되는 대신 마지막 정상 MCP 도구를 계속 제공하며, 이는 수정 전 mtime-only 계약
    # (파일을 stat할 수 없게 되면 마찬가지로 False를 반환했다)과 일치한다. 실수가 아니라 의도된
    # fail-soft 선택이다. "config 삭제"로 MCP 도구를 내리고 싶은 향후 변경은 추론이 아니라
    # 여기에 자체 명시적 신호를 두어야 한다.
    if _config_signature is None or current_signature is None:
        return False

    if current_path != _config_path:
        logger.info("MCP config path changed (%s -> %s), cache is stale", _config_path, current_path)
        return True

    if current_signature != _config_signature:
        logger.info("MCP config content changed (signature %s -> %s), cache is stale", _config_signature, current_signature)
        return True

    return False


async def initialize_mcp_tools() -> list[BaseTool]:
    """MCP 도구를 초기화하고 캐싱한다.

    애플리케이션 startup에서 한 번만 호출해야 한다.

    Returns:
        활성화된 모든 MCP 서버의 LangChain 도구 목록.
    """
    global _mcp_tools_cache, _cache_initialized, _config_path, _config_signature

    async with _initialization_lock:
        if _cache_initialized:
            logger.info("MCP tools already initialized")
            return _mcp_tools_cache or []

        from deerflow.mcp.tools import get_mcp_tools

        logger.info("Initializing MCP tools...")
        _mcp_tools_cache = await get_mcp_tools()
        _cache_initialized = True
        _config_path, _config_signature = _current_config_state()  # config 경로 + 콘텐츠 signature 기록
        logger.info("MCP tools initialized: %d tool(s) loaded (config path: %s)", len(_mcp_tools_cache), _config_path)

        return _mcp_tools_cache


def get_cached_mcp_tools() -> list[BaseTool]:
    """캐싱된 MCP 도구를 lazy initialization과 함께 반환한다.

    도구가 초기화되지 않았으면 자동으로 초기화한다. 덕분에 FastAPI와 LangGraph Studio
    양쪽 context에서 MCP 도구가 동작한다.

    마지막 초기화 이후 config 파일이 수정되었는지도 확인해 필요하면 재초기화한다. 덕분에
    Gateway API로 변경한 내용이 Gateway 내장 LangGraph runtime에 반영된다.

    Returns:
        캐싱된 MCP 도구 목록.
    """
    global _cache_initialized

    # config 파일 변경으로 캐시가 stale해졌는지 확인
    if _is_cache_stale():
        logger.info("MCP cache is stale, resetting for re-initialization...")
        reset_mcp_tools_cache()

    if not _cache_initialized:
        logger.info("MCP tools not initialized, performing lazy initialization...")
        try:
            # 현재 event loop에서 초기화를 시도한다
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # loop가 이미 실행 중이면(예: LangGraph Studio) 별도 thread에서
                # 새 loop를 만들어야 한다
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, initialize_mcp_tools())
                    future.result()
            else:
                # 실행 중인 loop가 없으면 현재 loop를 그대로 쓴다
                loop.run_until_complete(initialize_mcp_tools())
        except RuntimeError:
            # event loop가 없으므로 새로 만든다
            try:
                asyncio.run(initialize_mcp_tools())
            except Exception:
                logger.exception("Failed to lazy-initialize MCP tools")
                return []
        except Exception:
            logger.exception("Failed to lazy-initialize MCP tools")
            return []

    return _mcp_tools_cache or []


def reset_mcp_tools_cache() -> None:
    """MCP 도구 캐시를 리셋한다.

    테스트나 MCP 도구를 다시 로드하고 싶을 때 쓴다. 지속 MCP session도 모두 닫아서
    다음 도구 로드 때 재생성되게 한다.
    """
    global _mcp_tools_cache, _cache_initialized, _config_path, _config_signature
    _mcp_tools_cache = None
    _cache_initialized = False
    _config_path = None
    _config_signature = None

    # 지속 session을 닫는다. 다음 get_mcp_tools() 호출이 (갱신되었을 수 있는) 연결 config로
    # 다시 만든다.
    #
    # close_all_sync()는 이미 소유 loop별로 올바른 전략을 고른다:
    #   * *현재* 실행 중인 loop가 소유한 session은 *신호만* 보낸다
    #     (loop가 제어권을 되찾으면 소유 task가 __aexit__을 실행한다 — loop가 task를
    #     살려두므로 올바르고 leak도 없다),
    #   * 다른 thread의 loop에 있는 session은 결정적으로 정리한다,
    #   * idle/닫힌 loop는 처리하거나 건너뛴다.
    # 현재 실행 중인 loop의 teardown 완료를 여기서 동기적으로 기다리지는 않는다. 그것은
    # self-deadlock이다(loop는 이 동기 호출이 제어권을 돌려준 뒤에야 teardown을 실행할 수 있다).
    try:
        from deerflow.mcp.session_pool import get_session_pool

        get_session_pool().close_all_sync()
    except Exception:
        logger.debug("Could not close MCP session pool on cache reset", exc_info=True)

    from deerflow.mcp.session_pool import reset_session_pool

    reset_session_pool()
    logger.info("MCP tools cache reset")
