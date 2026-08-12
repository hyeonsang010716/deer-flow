"""read-before-write 파일 gate middleware 설정(issue #3857)."""

from pydantic import BaseModel, Field


class ReadBeforeWriteConfig(BaseModel):
    """파일을 수정하는 도구에 대한 결정적 버전 gate.

    활성화하면 ``write_file``(기존 파일에 append 또는 덮어쓰기)과 ``str_replace``는 해당
    파일이 마지막 수정 이후 ``read_file``로 읽힌 적이 없으면 차단된다. 에이전트가 파일을
    바꾸기 전에 현재 상태를 반드시 보게 만든다.
    """

    enabled: bool = Field(
        default=True,
        description="Whether to block writes to existing files that were not read at their current version",
    )
