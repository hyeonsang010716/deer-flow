"""사용자 소유 IM channel connection용 SQL repository."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.channel_connections.model import (
    ChannelConnectionRow,
    ChannelConversationRow,
    ChannelCredentialRow,
    ChannelOAuthStateRow,
)
from deerflow.utils.time import coerce_iso

logger = logging.getLogger(__name__)

# 동시 writer가 충돌하는 row(같은 owner identity, 또는 partial unique index가 보호하는
# 같은 active external identity)를 먼저 커밋했을 때 upsert_connection이 수행하는 제한된 retry
# 횟수. 각 retry는 이제 보이는 state를 다시 읽으므로, 현실적인 경합에서는 작은 값으로 수렴한다.
_UPSERT_MAX_ATTEMPTS = 3


class ChannelCredentialCipher:
    """provider credential을 저장하기 전에 암호화한다."""

    def __init__(self, fernet: Fernet) -> None:
        self._fernet = fernet

    @classmethod
    def from_key(cls, key: str) -> ChannelCredentialCipher:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return cls(Fernet(base64.urlsafe_b64encode(digest)))

    def encrypt_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        return "fernet:v1:" + self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        token = value.removeprefix("fernet:v1:")
        return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")


class ChannelConnectionRepository:
    """channel connection, credential, conversation에 대한 persistence facade."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        cipher: ChannelCredentialCipher | None = None,
    ) -> None:
        self.session_factory = session_factory
        self._cipher = cipher

    async def close(self) -> None:
        from deerflow.persistence.engine import close_engine

        await close_engine()

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _normalize_optional_identity(value: str | None) -> str:
        return value or ""

    @staticmethod
    def _coerce_datetime(value: datetime | None) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=UTC)

    def _encrypt_optional_secret(self, value: str | None) -> str | None:
        if value is None:
            return None
        if self._cipher is None:
            raise RuntimeError("channel connection encryption key is required")
        return self._cipher.encrypt_text(value)

    @staticmethod
    def _connection_to_dict(row: ChannelConnectionRow) -> dict[str, Any]:
        data = row.to_dict()
        data["external_account_id"] = data["external_account_id"] or None
        data["workspace_id"] = data["workspace_id"] or None
        data["scopes"] = data.pop("scopes_json") or []
        data["capabilities"] = data.pop("capabilities_json") or {}
        data["metadata"] = data.pop("metadata_json") or {}
        for key in ("created_at", "updated_at", "last_seen_at", "last_error_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = coerce_iso(value)
        return data

    async def upsert_connection(
        self,
        *,
        owner_user_id: str,
        provider: str,
        external_account_id: str | None = None,
        external_account_name: str | None = None,
        workspace_id: str | None = None,
        workspace_name: str | None = None,
        bot_user_id: str | None = None,
        scopes: list[str] | None = None,
        capabilities: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "connected",
    ) -> dict[str, Any]:
        external_account_id_value = self._normalize_optional_identity(external_account_id)
        workspace_id_value = self._normalize_optional_identity(workspace_id)

        def _apply(row: ChannelConnectionRow) -> None:
            row.status = status
            row.external_account_name = external_account_name
            row.workspace_name = workspace_name
            row.bot_user_id = bot_user_id
            row.scopes_json = list(scopes or [])
            row.capabilities_json = dict(capabilities or {})
            row.metadata_json = dict(metadata or {})

        async def _revoke_other_active_owners(session: AsyncSession) -> None:
            if status != "connected":
                return
            with session.no_autoflush:
                result = await session.execute(
                    select(ChannelConnectionRow.id).where(
                        ChannelConnectionRow.provider == provider,
                        ChannelConnectionRow.external_account_id == external_account_id_value,
                        ChannelConnectionRow.workspace_id == workspace_id_value,
                        ChannelConnectionRow.owner_user_id != owner_user_id,
                        ChannelConnectionRow.status != "revoked",
                    )
                )
            transferred_ids = [row_id for row_id in result.scalars()]
            if not transferred_ids:
                return
            await session.execute(update(ChannelConnectionRow).where(ChannelConnectionRow.id.in_(transferred_ids)).values(status="revoked"))
            await session.execute(delete(ChannelCredentialRow).where(ChannelCredentialRow.connection_id.in_(transferred_ids)))

        stmt = select(ChannelConnectionRow).where(
            ChannelConnectionRow.owner_user_id == owner_user_id,
            ChannelConnectionRow.provider == provider,
            ChannelConnectionRow.external_account_id == external_account_id_value,
            ChannelConnectionRow.workspace_id == workspace_id_value,
        )

        async with self.session_factory() as session:
            last_error: IntegrityError | None = None
            for _ in range(_UPSERT_MAX_ATTEMPTS):
                try:
                    row = (await session.execute(stmt)).scalar_one_or_none()
                    # 이 external identity에 대한 다른 owner의 active row를 우리 connected row가
                    # flush되기 *전에* 회수한다. 그래야 commit 시점에 active identity에 걸린
                    # partial unique index를 만족한다.
                    await _revoke_other_active_owners(session)
                    if row is None:
                        row = ChannelConnectionRow(
                            id=self._new_id(),
                            owner_user_id=owner_user_id,
                            provider=provider,
                            external_account_id=external_account_id_value,
                            workspace_id=workspace_id_value,
                        )
                        session.add(row)
                    _apply(row)
                    await session.commit()
                    await session.refresh(row)
                    return self._connection_to_dict(row)
                except IntegrityError as exc:
                    # 동시 writer가 충돌하는 row(이 owner의 identity, 또는 같은 active external
                    # identity)를 먼저 커밋했다. rollback 후 retry한다. 다음 회차는 이제 보이는
                    # state를 다시 읽고, 새로 커밋된 owner를 회수한 뒤 우리 row를 쓴다.
                    last_error = exc
                    await session.rollback()
            raise last_error  # type: ignore[misc]  # loop runs at least once

    async def list_connections(self, owner_user_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(select(ChannelConnectionRow).where(ChannelConnectionRow.owner_user_id == owner_user_id).order_by(ChannelConnectionRow.updated_at.desc(), ChannelConnectionRow.id.desc()))
            return [self._connection_to_dict(row) for row in result.scalars()]

    async def disconnect_connection(self, *, connection_id: str, owner_user_id: str) -> bool:
        async with self.session_factory() as session:
            row = await session.get(ChannelConnectionRow, connection_id)
            if row is None or row.owner_user_id != owner_user_id:
                return False

            row.status = "revoked"
            credential = await session.get(ChannelCredentialRow, connection_id)
            if credential is not None:
                await session.delete(credential)
            await session.commit()
            return True

    async def disconnect_provider_connections(self, *, provider: str) -> int:
        """instance 전역 provider 제거를 위해 모든 active 사용자 connection을 회수한다."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChannelConnectionRow.id).where(
                    ChannelConnectionRow.provider == provider,
                    ChannelConnectionRow.status != "revoked",
                )
            )
            connection_ids = [row_id for row_id in result.scalars()]
            if not connection_ids:
                return 0

            await session.execute(update(ChannelConnectionRow).where(ChannelConnectionRow.id.in_(connection_ids)).values(status="revoked"))
            await session.execute(delete(ChannelCredentialRow).where(ChannelCredentialRow.connection_id.in_(connection_ids)))
            await session.commit()
            return len(connection_ids)

    async def store_credentials(
        self,
        connection_id: str,
        *,
        access_token: str | None,
        refresh_token: str | None = None,
        token_type: str | None = None,
        expires_at: datetime | None = None,
        refresh_expires_at: datetime | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._cipher is None:
            raise RuntimeError("channel connection encryption key is required")
        async with self.session_factory() as session:
            row = await session.get(ChannelCredentialRow, connection_id)
            if row is None:
                row = ChannelCredentialRow(connection_id=connection_id)
                session.add(row)
            row.encrypted_access_token = self._cipher.encrypt_text(access_token)
            row.encrypted_refresh_token = self._cipher.encrypt_text(refresh_token)
            row.token_type = token_type
            row.expires_at = expires_at
            row.refresh_expires_at = refresh_expires_at
            row.encrypted_extra_json = self._cipher.encrypt_text(json.dumps(extra or {}, ensure_ascii=False))
            row.version = (row.version or 0) + 1
            await session.commit()

    async def get_credentials(self, connection_id: str) -> dict[str, Any] | None:
        if self._cipher is None:
            return None
        async with self.session_factory() as session:
            row = await session.get(ChannelCredentialRow, connection_id)
            if row is None:
                return None
            try:
                extra_raw = self._cipher.decrypt_text(row.encrypted_extra_json)
                return {
                    "connection_id": row.connection_id,
                    "access_token": self._cipher.decrypt_text(row.encrypted_access_token),
                    "refresh_token": self._cipher.decrypt_text(row.encrypted_refresh_token),
                    "token_type": row.token_type,
                    "expires_at": self._coerce_datetime(row.expires_at),
                    "refresh_expires_at": self._coerce_datetime(row.refresh_expires_at),
                    "extra": json.loads(extra_raw) if extra_raw else {},
                }
            except (InvalidToken, UnicodeError, json.JSONDecodeError):
                logger.warning(
                    "Unable to decrypt channel connection credentials; treating credentials as unavailable",
                    exc_info=True,
                )
                return None

    @staticmethod
    def hash_state(state: str) -> str:
        return hashlib.sha256(state.encode("utf-8")).hexdigest()

    async def create_oauth_state(
        self,
        *,
        owner_user_id: str,
        provider: str,
        state: str,
        expires_at: datetime,
        code_verifier: str | None = None,
        nonce_hash: str | None = None,
        redirect_after: str | None = None,
        requested_scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = ChannelOAuthStateRow(
            state_hash=self.hash_state(state),
            owner_user_id=owner_user_id,
            provider=provider,
            code_verifier_encrypted=self._encrypt_optional_secret(code_verifier),
            nonce_hash=nonce_hash,
            redirect_after=redirect_after,
            requested_scopes_json=list(requested_scopes or []),
            metadata_json=dict(metadata or {}),
            expires_at=expires_at,
        )
        async with self.session_factory() as session:
            session.add(row)
            await session.commit()

    async def create_oauth_state_within_cap(
        self,
        *,
        owner_user_id: str,
        provider: str,
        state: str,
        expires_at: datetime,
        max_pending: int,
        now: datetime | None = None,
        code_verifier: str | None = None,
        nonce_hash: str | None = None,
        redirect_after: str | None = None,
        requested_scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """(owner, provider)별 pending cap을 원자적으로 강제한 뒤 insert한다.

        만료 삭제 + count + insert가 (owner, provider) 단위로 직렬화된 단일 transaction에서
        실행되므로, 동시 connect 요청들이 각각 ``count < max_pending``을 관찰하고 모두
        insert하는(= cap을 넘겨버리는) 일이 생기지 않는다. PostgreSQL은 transaction 범위
        advisory lock을 잡고, SQLite는 선행 DELETE가 획득하는 write lock으로 writer를 직렬화한다.

        row를 넣었으면 ``True``, 이미 cap에 도달했으면 ``False``를 반환한다.
        """
        current_time = now or datetime.now(UTC)
        async with self.session_factory() as session:
            await self._serialize_oauth_owner_scope(session, owner_user_id, provider)
            # 모든 사용자가 아니라 이 owner/provider의 만료 code(이 cap에 영향을 주는 것)만
            # 정리한다. connect POST마다 전역 DELETE가 도는 것을 피하기 위해서다. 이 write를
            # 먼저 내면 SQLite database write lock도 잡히므로, 아래 count가 count와 commit
            # 사이에서 동시 inserter와 경합하지 않는다. 다른 owner의 오래된 code는
            # consume_oauth_state / delete_expired_oauth_states가 전역으로 정리한다.
            await session.execute(
                delete(ChannelOAuthStateRow).where(
                    ChannelOAuthStateRow.owner_user_id == owner_user_id,
                    ChannelOAuthStateRow.provider == provider,
                    ChannelOAuthStateRow.expires_at < current_time,
                )
            )
            pending = await session.execute(
                select(func.count())
                .select_from(ChannelOAuthStateRow)
                .where(
                    ChannelOAuthStateRow.owner_user_id == owner_user_id,
                    ChannelOAuthStateRow.provider == provider,
                    ChannelOAuthStateRow.consumed_at.is_(None),
                    ChannelOAuthStateRow.expires_at >= current_time,
                )
            )
            if int(pending.scalar_one()) >= max_pending:
                await session.rollback()
                return False
            session.add(
                ChannelOAuthStateRow(
                    state_hash=self.hash_state(state),
                    owner_user_id=owner_user_id,
                    provider=provider,
                    code_verifier_encrypted=self._encrypt_optional_secret(code_verifier),
                    nonce_hash=nonce_hash,
                    redirect_after=redirect_after,
                    requested_scopes_json=list(requested_scopes or []),
                    metadata_json=dict(metadata or {}),
                    expires_at=expires_at,
                )
            )
            await session.commit()
            return True

    async def _serialize_oauth_owner_scope(self, session: AsyncSession, owner_user_id: str, provider: str) -> None:
        """한 (owner, provider)에 대한 동시 pending-cap transaction을 직렬화한다.

        PostgreSQL에서는 transaction 범위 advisory lock을 잡아 동시 발급자들이 count+insert를
        한 번에 하나씩 수행하게 한다. SQLite에서는 호출자 transaction의 선행 DELETE가 이미
        database write lock을 획득해 writer를 직렬화하므로 추가 lock이 필요 없다.
        """
        try:
            dialect = session.bind.dialect.name if session.bind is not None else ""
        except Exception:
            dialect = ""
        if dialect == "postgresql":
            await session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": self._oauth_scope_lock_key(owner_user_id, provider)})

    @staticmethod
    def _oauth_scope_lock_key(owner_user_id: str, provider: str) -> int:
        digest = hashlib.sha256(f"{owner_user_id}\x00{provider}".encode()).digest()
        # pg_advisory_xact_lock(bigint)용 63비트 음이 아닌 key.
        return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF

    async def delete_expired_oauth_states(self, *, now: datetime | None = None) -> int:
        current_time = now or datetime.now(UTC)
        async with self.session_factory() as session:
            result = await session.execute(delete(ChannelOAuthStateRow).where(ChannelOAuthStateRow.expires_at < current_time))
            await session.commit()
            return int(result.rowcount or 0)

    async def count_oauth_states(
        self,
        *,
        owner_user_id: str,
        provider: str,
        active_only: bool = False,
        now: datetime | None = None,
    ) -> int:
        current_time = now or datetime.now(UTC)
        conditions = [
            ChannelOAuthStateRow.owner_user_id == owner_user_id,
            ChannelOAuthStateRow.provider == provider,
        ]
        if active_only:
            conditions.extend(
                [
                    ChannelOAuthStateRow.consumed_at.is_(None),
                    ChannelOAuthStateRow.expires_at >= current_time,
                ]
            )

        async with self.session_factory() as session:
            result = await session.execute(select(func.count()).select_from(ChannelOAuthStateRow).where(*conditions))
            return int(result.scalar_one())

    async def consume_oauth_state(
        self,
        *,
        provider: str,
        state: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current_time = now or datetime.now(UTC)
        state_hash = self.hash_state(state)
        async with self.session_factory() as session:
            await session.execute(delete(ChannelOAuthStateRow).where(ChannelOAuthStateRow.expires_at < current_time))
            row = await session.get(ChannelOAuthStateRow, state_hash)
            if row is None or row.provider != provider or row.consumed_at is not None:
                await session.commit()
                return None
            expires_at = self._coerce_datetime(row.expires_at)
            if expires_at is not None and expires_at < current_time:
                await session.commit()
                return None

            # 조건부 UPDATE라서 두 동시 worker가 같은 binding code를 함께 소비할 수 없다.
            # consumed_at을 NULL에서 뒤집는 writer만 이긴다.
            result = await session.execute(
                update(ChannelOAuthStateRow)
                .where(
                    ChannelOAuthStateRow.state_hash == state_hash,
                    ChannelOAuthStateRow.consumed_at.is_(None),
                )
                .values(consumed_at=current_time)
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            return {
                "owner_user_id": row.owner_user_id,
                "provider": row.provider,
                "requested_scopes": row.requested_scopes_json or [],
                "metadata": row.metadata_json or {},
                "redirect_after": row.redirect_after,
            }

    async def find_connection_by_external_identity(
        self,
        *,
        provider: str,
        external_account_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChannelConnectionRow)
                .where(
                    ChannelConnectionRow.provider == provider,
                    ChannelConnectionRow.external_account_id == self._normalize_optional_identity(external_account_id),
                    ChannelConnectionRow.workspace_id == self._normalize_optional_identity(workspace_id),
                    ChannelConnectionRow.status == "connected",
                )
                .order_by(ChannelConnectionRow.updated_at.desc(), ChannelConnectionRow.id.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return self._connection_to_dict(row) if row is not None else None

    async def set_thread_id(
        self,
        *,
        connection_id: str,
        owner_user_id: str,
        provider: str,
        external_conversation_id: str,
        thread_id: str,
        external_topic_id: str | None = None,
    ) -> None:
        topic_id = external_topic_id or ""
        async with self.session_factory() as session:
            stmt = select(ChannelConversationRow).where(
                ChannelConversationRow.connection_id == connection_id,
                ChannelConversationRow.external_conversation_id == external_conversation_id,
                ChannelConversationRow.external_topic_id == topic_id,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = ChannelConversationRow(
                    id=self._new_id(),
                    connection_id=connection_id,
                    owner_user_id=owner_user_id,
                    provider=provider,
                    external_conversation_id=external_conversation_id,
                    external_topic_id=topic_id,
                    thread_id=thread_id,
                )
                session.add(row)
            else:
                row.thread_id = thread_id
                row.owner_user_id = owner_user_id
                row.provider = provider
            await session.commit()

    async def get_thread_id(
        self,
        connection_id: str,
        external_conversation_id: str,
        external_topic_id: str | None = None,
    ) -> str | None:
        async with self.session_factory() as session:
            stmt = select(ChannelConversationRow.thread_id).where(
                ChannelConversationRow.connection_id == connection_id,
                ChannelConversationRow.external_conversation_id == external_conversation_id,
                ChannelConversationRow.external_topic_id == (external_topic_id or ""),
            )
            return (await session.execute(stmt)).scalar_one_or_none()
