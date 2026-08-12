"""런타임에 수정 가능한 설정 파일용 공용 content-signature 헬퍼.

``config/app_config.py``(``config.yaml``)와 ``mcp/cache.py``
(``extensions_config.json``) 모두 설정 파일이 실제로 바뀌었는지 감지해야 한다.
단순 mtime 비교로는 놓치는 경우가 많다. 같은 초 안의 수정, mtime이 그대로거나
과거로 되돌아가는 경우(``git checkout``, ``cp -p``/백업 복원, 타임스탬프를 보존하는
``tar``/``rsync``, object-store 및 network mount), mtime이 이전 기록보다 작거나 같은
다른 파일로 교체되는 경우가 그렇다.

이 모듈은 그 ``(mtime, size, sha256)`` signature의 유일한 구현체다. 두 호출부가 같은
동작을 공유하게 해서, 그대로 복제된 사본이 시간이 지나며 조용히 어긋나는 것을 막는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 설정 파일에 대해 기록한 (mtime, size, sha256-hexdigest), 또는 이전 기록과 비교하려고
# 다시 계산한 현재 값이다. digest(세 번째 요소)가 ``None``이면 stat은 성공했지만 내용을
# 읽지 못했다는 뜻이고, 튜플 전체가 ``None``이면 stat 자체가 실패했다는 뜻이다
# (예: 파일이 존재하지 않음).
ConfigSignature = tuple[float | None, int | None, str | None]


def get_config_signature(config_path: Path) -> ConfigSignature | None:
    """*config_path*의 캐시 메타데이터를 content digest와 함께 반환한다.

    stat이 실패하면(예: 파일 없음) ``None``을 반환한다. 그래야 호출자가 "파일 없음"과
    "내용을 읽을 수 없는 파일"(아래에서 부분 signature를 반환)을 구분할 수 있다.
    """
    try:
        stat_result = config_path.stat()
    except OSError:
        return None

    # mtime/size가 이전 signature와 같더라도 단축하지 않고 항상 파일 전체를 해시한다.
    # 같은 초 안에 바이트 길이가 같은 다른 내용으로 교체되면 mtime과 size가 *둘 다*
    # 그대로여서 sha256만이 그 교체를 잡아낸다. mtime/size가 같다고 해시를 건너뛰면
    # 이 signature가 막으려던 좁은 틈이 다시 열린다.
    digest = hashlib.sha256()
    try:
        with config_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return (stat_result.st_mtime, stat_result.st_size, None)

    return (stat_result.st_mtime, stat_result.st_size, digest.hexdigest())
