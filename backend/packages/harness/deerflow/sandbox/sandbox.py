import re
from abc import ABC, abstractmethod

from deerflow.sandbox.search import GrepMatch

# POSIX 환경변수 이름 규칙: 글자나 밑줄로 시작하고 이어서 글자/숫자/밑줄. ``env`` 키가 sandbox
# 구현에 도달하기 전에 검증하는 데 쓴다. 현재 어떤 구현도 키를 shell 문자열에 끼워 넣지 않는다.
# local sandbox는 dict를 ``subprocess.run(env=...)``에 넘기고(shell 미사용), AIO sandbox는
# ``bash.exec``의 구조화된 ``env`` 필드로 전달하며, e2b는 SDK의 ``envs``로 전달한다. 이 검사는
# 계약에 대한 defense-in-depth다. 앞으로 shell에 끼워 넣는 구현이 생기더라도 자체 규칙을 다시
# 만들 필요가 없어야 한다.
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_extra_env(extra_env: dict[str, str] | None) -> None:
    """유효한 POSIX 환경변수 이름이 아닌 ``env`` 키를 거부한다.

    :meth:`Sandbox.execute_command` 계약은 임의의 ``str`` 키를 받는다. 현재는 어떤 구현도 키를
    shell 문자열에 끼워 넣지 않는다. local sandbox는 dict를 ``subprocess.run(env=...)``에
    넘기고(shell 미사용), AIO sandbox는 ``bash.exec``의 구조화된 ``env`` 필드로 전달하며
    (명령 문자열에 끼워 넣지 않는다), e2b는 SDK의 ``envs``로 전달한다. 추상 레이어에서 POSIX
    환경변수 이름 규칙을 강제하는 것은 계약에 대한 defense-in-depth다. 앞으로 키를 shell로
    보내는 구현이 생겨도 자체 검증 규칙을 다시 만들 필요가 없고, config / payload / 사용자
    입력에서 유도한 키를 넘기는 호출자는 나중에 구현이 끼워 넣기로 퇴행했을 때 조용히 취약점을
    만드는 대신 ``ValueError``로 즉시 실패한다.

    Raises:
        ValueError: ``extra_env``가 None이 아니고 어떤 키가
            ``^[A-Za-z_][A-Za-z0-9_]*$``에 맞지 않을 때. ``None``과 빈 dict는 그대로 통과한다.
    """
    if not extra_env:
        return
    for key in extra_env:
        if not isinstance(key, str) or not _ENV_NAME_PATTERN.fullmatch(key):
            raise ValueError(f"extra_env key {key!r} is not a valid POSIX environment variable name (must match ^[A-Za-z_][A-Za-z0-9_]*$). This protects shell-using sandbox implementations from command injection via the key.")


class Sandbox(ABC):
    """sandbox 환경의 추상 base class"""

    _id: str

    def __init__(self, id: str):
        self._id = id

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def execute_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """sandbox에서 bash 명령을 실행한다.

        Args:
            command: 실행할 명령.
            env: 명령 프로세스에 주입할 호출별 환경변수(선택). request-scoped secret
                (예: skill 스크립트용 단기 최종 사용자 토큰(이슈 #3861), ``git push`` / ``gh``용
                GitHub App installation 토큰)을 prompt, 도구 인자, 명령 문자열에 넣지 않고
                전달하는 데 쓴다. ``None``이면 sandbox의 기본 환경을 쓴다. 키는 유효한 POSIX
                환경변수 이름(``^[A-Za-z_][A-Za-z0-9_]*$``)이어야 하며, 구현은 사용 전
                :func:`_validate_extra_env`로 검증한다. 값은 임의 문자열이고, shell을 쓰는
                구현은 끼워 넣을 때 ``shlex.quote``를 적용한다.
            timeout: 호출별 wall-clock timeout(초, 선택). local sandbox는 이걸로 host bash
                명령을 제한해 오래 도는 foreground 프로세스가 turn을 무한정 붙잡지 못하게 한다.
                remote/AIO 구현은 backend가 자체 API timeout과 별개인 명령 timeout 제어를
                제공하지 않으면 무시할 수 있다.

        Returns:
            명령의 표준 출력 또는 에러 출력.

        Raises:
            ValueError: ``env`` 키가 유효한 환경변수 이름이 아닐 때.
        """
        pass

    @abstractmethod
    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """파일 내용을 읽는다.

        Args:
            path: 읽을 파일의 절대 경로.
            start_line: 시작 줄 번호(1부터, 포함, 선택).
            end_line: 끝 줄 번호(1부터, 포함, 선택).

        Returns:
            파일 내용.
        """
        pass

    @abstractmethod
    def download_file(self, path: str) -> bytes:
        """파일의 바이너리 내용을 내려받는다.

        Args:
            path: 내려받을 파일의 절대 경로.

        Returns:
            원본 파일 bytes.

        Raises:
            PermissionError: path traversal이 감지되거나 경로가 허용된 가상 prefix 밖일 때.
            OSError: 파일을 읽을 수 없거나 존재하지 않을 때. 호출자가 하나의 예외 타입만
                처리하면 되도록 local과 remote 구현 모두 ``OSError``를 던져야 한다.
        """
        pass

    @abstractmethod
    def list_dir(self, path: str, max_depth=2) -> list[str]:
        """디렉터리 내용을 나열한다.

        Args:
            path: 나열할 디렉터리의 절대 경로.
            max_depth: 탐색할 최대 깊이. 기본값은 2.

        Returns:
            디렉터리 내용.
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """파일에 내용을 쓴다.

        Args:
            path: 쓸 파일의 절대 경로.
            content: 파일에 쓸 텍스트 내용.
            append: 내용을 파일에 덧붙일지 여부. False면 파일을 새로 만들거나 덮어쓴다.
        """
        pass

    @abstractmethod
    def glob(self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200) -> tuple[list[str], bool]:
        """루트 디렉터리 아래에서 glob 패턴에 맞는 경로를 찾는다."""
        pass

    @abstractmethod
    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        """텍스트 파일 하나 또는 디렉터리 아래 파일들에서 일치하는 부분을 검색한다."""
        pass

    @abstractmethod
    def update_file(self, path: str, content: bytes) -> None:
        """파일을 바이너리 내용으로 갱신한다.

        Args:
            path: 갱신할 파일의 절대 경로.
            content: 파일에 쓸 바이너리 내용.
        """
        pass
