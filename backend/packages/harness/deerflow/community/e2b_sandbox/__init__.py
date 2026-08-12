"""DeerFlow용 E2B 클라우드 sandbox provider.

이 패키지는 `e2b` / `e2b_code_interpreter` 클라우드 sandbox SDK 위에서 DeerFlow의
:class:`Sandbox` / :class:`SandboxProvider` 계약을 구현한다.

설정 예시(``config.yaml``)::

    sandbox:
      use: deerflow.community.e2b_sandbox:E2BSandboxProvider
      # E2B 전용 옵션 (SandboxConfig의 ``extra="allow"``로 읽는다):
      api_key: $E2B_API_KEY            # 없으면 E2B_API_KEY 환경 변수로 대체된다
      template: code-interpreter-v1     # e2b template id. 기본값은 e2b code-interpreter
      domain: e2b.dev                  # 선택적 e2b 도메인 (예: 자체 호스팅)
      idle_timeout: 600                # e2b ``set_timeout``으로 전달된다 (초)
      replicas: 3                      # ownership이 Redis일 때 공유되는 하드 정원
      ownership:                       # 다중 worker ownership + 정원 조정
        type: redis
        redis_url: $REDIS_URL
      reconciliation_interval_seconds: 60
      reconciliation_grace_seconds: 120
      reconciliation_orphan_ttl_seconds: 3600
      reconciliation_max_pages: 10
      reconciliation_max_items: 200
      reconciliation_max_seconds: 15
      mounts:                          # 호스트 파일을 sandbox로 1회 업로드한다
        - host_path: /path/on/host
          container_path: /path/in/sandbox
          read_only: false
      environment:                      # 생성 시 e2b ``envs``로 전달된다
        OPENAI_API_KEY: $OPENAI_API_KEY
"""

from .e2b_sandbox import E2BSandbox
from .e2b_sandbox_provider import E2BSandboxProvider

__all__ = [
    "E2BSandbox",
    "E2BSandboxProvider",
]
