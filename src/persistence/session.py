import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, List

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Mapped, mapped_column

from src.persistence.orm_base import BaseORM, BaseDB, engine, Session, AsyncSessionLocal
from src.persistence.user import UserORM
from src.agent.production.graph import Agent


class SessionORM(BaseORM):
    __tablename__ = "session"
    __table_args__ = {"schema": "public"}
    session_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    service_user_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column()
    username: Mapped[str] = mapped_column()
    user_role: Mapped[str] = mapped_column()
    model_name: Mapped[str] = mapped_column()
    embedding_name: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()
    message_count: Mapped[int] = mapped_column()


class SessionSchema(BaseModel):
    session_id: uuid.UUID
    service_user_id: uuid.UUID
    title: str
    username: str
    user_role: Literal["user", "admin", "assistant"]
    model_name: str
    embedding_name: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class SessionDB(BaseDB):
    def _create_session(self, user: UserORM, agent: Agent) -> SessionORM:
        self.logger.info("Creating agent session")
        session_db = SessionORM(
            session_id=uuid.uuid4(),
            service_user_id=user.user_id,
            username=user.username,
            user_role=user.user_role,
            # model_name=agent.llm.model_name,  # TODO: change if appropriate
            # embedding_name=agent.embeddings.model_name, # TODO: change if appropriate
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.session.add(session_db)
        return session_db

    def create_session(self, user: UserORM, agent: Agent) -> SessionORM:
        session_db = self._create_session(user, agent)
        self.session.commit()
        self.session.refresh(session_db)
        return session_db

    async def async_create_session(self, user: UserORM, agent: Agent) -> SessionORM:
        session_db = self._create_session(user, agent)
        await self.session.commit()
        await self.session.refresh(session_db)
        return session_db

    @staticmethod
    def _get_session_info_query(session_id: Optional[uuid.UUID] = None, user_id: Optional[uuid.UUID] = None):
        assert (session_id or user_id)
        if session_id is not None:
            return select(SessionORM).where(SessionORM.session_id == session_id)
        if user_id is not None:
            return select(SessionORM).where(SessionORM.service_user_id == user_id)
        return None

    def sync_get_session(self, session_id: uuid.UUID) -> Optional[SessionORM]:
        session_info_query = self._get_session_info_query(session_id=session_id)
        result = self.session.execute(session_info_query)
        if result.scalar() is not None:
            return result.scalars().first()
        return None

    async def async_get_session(self, session_id: uuid.UUID) -> Optional[SessionORM]:
        session_info_query = self._get_session_info_query(session_id=session_id)
        result = await self.session.execute(session_info_query)
        if result.scalar() is not None:
            return result.scalars().first()
        return None

    def sync_delete_session_by_id(self, session_id: uuid.UUID):
        self.logger.info(f"Deleting session {session_id}")
        session_db = self.sync_get_session(session_id)
        if session_db is not None:
            self.session.delete(session_db)
            self.session.commit()
            return session_db
        else:
            self.logger.info(f"Session {session_id} was not found.")
            return None

    async def async_delete_session_by_id(self, session_id: uuid.UUID):
        self.logger.info(f"Deleting session {session_id}")
        session_db = await self.async_get_session(session_id)
        if session_db is not None:
            await self.session.delete(session_db)
            await self.session.commit()
            return session_db
        else:
            self.logger.info(f"Session {session_id} was not found.")
            return None

    def sync_update_session_title(self, session_id: uuid.UUID, title: str) -> SessionORM | None:
        session_info = self.sync_get_session(session_id)
        if session_info is None:
            self.logger.info(f"Session {session_id} was not found.")
            return None
        else:
            session_info.title = title
            self.session.add(session_info)
            self.session.commit()
            self.session.refresh(session_info)
            return session_info

    async def async_update_session_title(self, session_id: uuid.UUID, title: str) -> SessionORM | None:
        session_info = await self.async_get_session(session_id)
        if session_info is None:
            self.logger.info(f"Session {session_id} was not found.")
            return None
        else:
            session_info.title = title
            self.session.add(session_info)
            await self.session.commit()
            await self.session.refresh(session_info)
            return session_info

    def sync_update_session_activity(self, session_id: uuid.UUID) -> SessionORM | None:
        session_info = self.sync_get_session(session_id)
        if session_info is None:
            self.logger.info(f"Session {session_id} was not found.")
            return None
        else:
            session_info.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.session.add(session_info)
            self.session.commit()
            self.session.refresh(session_info)
            return session_info

    async def async_update_session_activity(self, session_id: uuid.UUID) -> SessionORM | None:
        session_info = await self.async_get_session(session_id)
        if session_info is None:
            self.logger.info(f"Session {session_id} was not found.")
            return None
        else:
            session_info.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.session.add(session_info)
            await self.session.commit()
            await self.session.refresh(session_info)
            return session_info

    def sync_get_user_sessions(self, user_id: uuid.UUID) -> List[SessionORM] | None:
        session_info_query = self._get_session_info_query(user_id=user_id)
        result = self.session.execute(session_info_query)
        if result.scalars() is not None:
            return list(result.scalars().all())
        self.logger.info(f"User {user_id} was not found.")
        return None

    async def async_get_user_sessions(self, user_id: uuid.UUID) -> List[SessionORM] | None:
        session_info_query = self._get_session_info_query(user_id=user_id)
        result = await self.session.execute(session_info_query)
        if result.scalars() is not None:
            return list(result.scalars().all())
        self.logger.info(f"User {user_id} was not found.")
        return None