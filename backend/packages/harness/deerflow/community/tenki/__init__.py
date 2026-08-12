"""DeerFlow용 Tenki 클라우드 sandbox provider.

`Tenki <https://tenki.cloud>`_ 클라우드 sandbox를 DeerFlow의
:class:`Sandbox` / :class:`SandboxProvider` 계약 뒤에 통합한다. 각 sandbox는 표준 base
image에서 만들어진 격리된 클라우드 microVM이며, 계약 전체를 구현한다. 즉
``execute_command`` 와 ``read_file`` / ``write_file`` / ``update_file`` /
``download_file`` / ``list_dir`` / ``glob`` / ``grep`` 을 모두 제공한다.
파일 전송은 Tenki의 네이티브 ``sandbox.fs`` API를 쓰고, 검색만 ``find`` / ``grep`` 을 shell로 호출한다.

설정 예시(``config.yaml``)::

    sandbox:
      use: deerflow.community.tenki:TenkiSandboxProvider
      # Tenki 전용 옵션 (SandboxConfig의 ``extra="allow"`` 로 읽힌다):
      api_key: $TENKI_API_KEY          # 없으면 TENKI_API_KEY / TENKI_AUTH_TOKEN 환경변수로 fallback
      base_url: https://tenki.cloud    # 선택, 생략 시 SDK 기본값
      image: my-base-image             # 선택, 생략 시 Tenki 계정의 기본 base image
      project_id: proj_...             # 선택, 계정에 프로젝트가 하나뿐이면 자동 선택
      workspace_id: ws_...             # 선택, 계정에 workspace가 하나뿐이면 자동 선택
      cpu_cores: 2                     # 선택, sandbox당 vCPU 수
      memory_mb: 2048                  # 선택, sandbox당 메모리
      replicas: 3                      # gateway 프로세스당 active + warm microVM 상한
      idle_timeout: 600                # warm microVM을 종료하기까지의 유휴 초. 0이면 비활성
      max_duration: 14400              # Tenki sandbox 수명(초). 0이면 계정 기본값
      sticky: false                    # microVM을 호스트에 고정 (pause/resume에서만 의미 있음)
      home_dir: /home/tenki            # /mnt/user-data를 뒷받침하는 쓰기 가능 디렉터리
      environment:                     # 모든 명령에 주입 (생성 시점 env로도 사용)
        PYTHONUNBUFFERED: "1"

이 provider를 선택하기 전에 선택적 SDK를 설치한다::

    pip install "deerflow-harness[tenki]"

Tenki의 안정적인 표면(sandbox 생성/종료, exec/shell/filesystem)만 사용한다. volume,
snapshot, template build는 의도적으로 쓰지 않으므로 미리 구운 image나 불안정한 Tenki 기능이 필요 없다.
"""

from .provider import TenkiSandboxProvider
from .sandbox import TenkiSandbox

__all__ = [
    "TenkiSandbox",
    "TenkiSandboxProvider",
]
