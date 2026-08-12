"""공유 inbound webhook dedupe 테이블 (issue #4120).

ORM model만 둔다. row 읽기/쓰기는 ``app.channels.dedupe_store.PostgresInboundDedupeStore``의
raw SQL을 통해 이뤄진다.
"""
