"""user 저장소 서브패키지.

``users`` 테이블의 ORM model을 담는다. 구체적인 repository 구현
(``SQLiteUserRepository``)은 app 계층(``app.gateway.auth.repositories.sqlite``)에 있는데,
ORM row와 auth 모듈의 pydantic ``User`` 클래스 사이를 변환하기 때문이다. 덕분에 harness
패키지는 app 코드에 전혀 의존하지 않는다.
"""

from deerflow.persistence.user.model import UserRow

__all__ = ["UserRow"]
