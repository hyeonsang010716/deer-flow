"""``langgraph.types.Overwrite``로 감싸인 channel 값을 다루는 헬퍼."""

from langgraph.types import Overwrite


def unwrap_sandbox(sandbox: object) -> tuple[object, bool]:
    """sandbox channel 값이 ``Overwrite``로 감싸여 있으면 벗겨 낸다.

    fork로 복원된 checkpoint는 sandbox channel을 여전히 ``langgraph.types.Overwrite``로 감싼 채
    전달할 수 있다(delta checkpoint mode에서 rollback 복원은 state mutation graph를 통해
    replace 방식의 쓰기를 수행한다). wrapper 자체에 ``sandbox["sandbox_id"]``나
    ``sandbox.get("sandbox_id")``를 하면 터지므로 쓰기 전에 벗겨야 한다.

    ``(value, fork_restored)``를 반환한다. 감싸인 형태는 부모 thread의 sandbox state를
    재생하는 것이므로, 호출자는 그 sandbox를 이 run이 소유한 것처럼 다뤄서는 안 된다(예: release).
    """
    if isinstance(sandbox, Overwrite):
        return sandbox.value, True
    return sandbox, False
