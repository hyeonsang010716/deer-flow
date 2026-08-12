"""DeerFlow sandbox용 BoxLite micro-VM backend.

`BoxLite <https://github.com/boxlite-ai/boxlite>`_ — daemon이 없는 OCI 네이티브
micro-VM runtime(Linux는 libkrun/KVM, macOS는 Hypervisor.framework) — 를 DeerFlow의
:class:`Sandbox` / :class:`SandboxProvider` 계약 뒤에 통합한다. 각 sandbox는 자체
커널을 가진 하드웨어 격리 VM이며 어떤 OCI 이미지든 수정 없이 실행한다.
https://github.com/bytedance/deer-flow/issues/3936 참고.

계약 전체를 구현한다. ``execute_command``와 ``read_file`` / ``write_file`` /
``update_file`` / ``download_file`` / ``list_dir`` / ``glob`` / ``grep``이며,
파일 연산은 box 안에서 shell 명령으로 실행된다.

설정 예시 (``config.yaml``)::

    sandbox:
      use: deerflow.community.boxlite:BoxliteProvider
      image: python:3.12-slim      # 어떤 OCI 이미지든 수정 없이 실행된다
      memory_mib: 1024             # box당 메모리 상한(선택)
      cpus: 2                      # box당 vCPU(선택)
      replicas: 3                  # gateway 프로세스당 active + warm VM 상한
      idle_timeout: 600            # warm VM을 멈추기까지의 유휴 시간(초). 0이면 비활성
      environment:                 # 모든 명령에 주입된다
        PYTHONUNBUFFERED: "1"

이 provider를 선택하기 전에 optional runtime을 설치한다::

    pip install "deerflow-harness[boxlite]"

호스트 요구사항: BoxLite는 micro-VM을 부팅하므로 Linux 호스트에는 KVM이 필요하다
(DeerFlow 자체가 클라우드 VM 안에서 돈다면 중첩 가상화가 필요하다). macOS는
Hypervisor.framework를 쓴다.
"""

from .box import BoxliteBox
from .provider import BoxliteProvider

__all__ = [
    "BoxliteBox",
    "BoxliteProvider",
]
