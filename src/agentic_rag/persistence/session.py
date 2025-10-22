import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, List, Union

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Mapped, mapped_column, Session
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.persistence.orm_base import BaseORM, BaseDB
from agentic_rag.persistence.user import UserORM
from agentic_rag.agent.graph import Agent


class SessionORM(BaseORM):
    """ORM mapping for the ``public.session`` table."""
    __tablename__ = "session"
    __table_args__ = {"schema": "public"}
    session_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4) #: :class:`~sqlalchemy.orm.Mapped`[:class:`uuid.UUID`] : Primary key; generated via :func:`uuid.uuid4`.
    service_user_id: Mapped[uuid.UUID] = mapped_column(nullable=False) #: :class:`~sqlalchemy.orm.Mapped`[:class:`uuid.UUID`] : Identifier of the owning service user.
    title: Mapped[str] = mapped_column(nullable=True) #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Optional human-readable session title.
    username: Mapped[str] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Username associated with the session.
    user_role: Mapped[str] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`\[:class:`str`\] : Role of the user.
    model_name: Mapped[str] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Name of the LLM model deployment used in the session.
    embedding_name: Mapped[str] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Name of the embedding model deployment used in the session.
    created_at: Mapped[datetime] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`datetime`] : Creation timestamp (stored as naive UTC).
    updated_at: Mapped[datetime] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`datetime`] : Last update timestamp (stored as naive UTC).
    message_count: Mapped[int] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`int`] : Number of messages exchanged in the session.


class SessionSchema(BaseModel):
    """Pydantic schema mirroring :class:`SessionORM` for I/O and validation."""
    session_id: uuid.UUID #: :class:`uuid.UUID` : Primary key of the session.
    service_user_id: uuid.UUID #: :class:`uuid.UUID` : Identifier of the current user.
    title: str #: :class:`str` : Human-readable session title.
    username: str #: :class:`str` : Username associated with the session.
    user_role: Literal["user", "admin", "assistant"] #: :class:`typing.Literal`\[{``user``, ``admin``, ``assistant``}\] : Role of the user for this session.
    model_name: str #: :class:`str` : Name of the LLM model deployment used in the session.
    embedding_name: str #: :class:`str` : Name of the embedding model deployment used in the session.
    created_at: datetime #: :class:`datetime.datetime` : Creation timestamp (naive UTC).
    updated_at: datetime #: :class:`datetime.datetime` : Last update timestamp (naive UTC).
    message_count: int #: :class:`int` : Number of messages exchanged in the session.


class SessionDB(BaseDB):
    """Session DataBase CRUD (Create-Read-Update-Delete), sync and async, backed by :sqlalchemy:`SQLAlchemy <>`.

    On construction, this class ensures that the :class:`~src.agentic_rag.persistence.session.SessionORM` table exists
    by calling :sqlalchemy:`sqlalchemy.MetaData.create_all <core/metadata.html#sqlalchemy.schema.MetaData.create_all>` for that table only.

    .. rubric:: Notes

    Timestamps are stored as *naive* UTC datetimes using ``datetime.now(timezone.utc).replace(tzinfo=None)``
    to avoid timezone-aware values in the database.

    Parameters
    ----------
    session : :sqlalchemy:`AsyncSession <orm/extensions/asyncio.html#sqlalchemy.ext.asyncio.AsyncSession>` or :sqlalchemy:`Session </orm/session_api.html#sqlalchemy.orm.Session>`
        Bound SQLAlchemy session used for all operations.
    """

    def __init__(self, session: Union[AsyncSession, Session]):
        super().__init__(session=session)
        engine = session.bind
        BaseORM.metadata.create_all(engine, tables=[SessionORM.__table__])

    def _create_session(self, user: UserORM, agent: Agent) -> SessionORM:
        """Construct a new :class:`~src.agentic_rag.persistence.session.SessionORM` (not committed).

        Parameters
        ----------
        user : :class:`~src.agentic_rag.persistence.user.UserORM`
            ORM instance for the current user; must expose ``user_id``, ``username``, and ``user_role``.
        agent : :class:`~src.agentic_rag.agent.graph.Agent`
            Agent instance providing LLM/embedding configuration.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.session.SessionORM`
            The newly constructed session object.
        """
        self.logger.info("Creating agent session")
        session_db = SessionORM(
            session_id=uuid.uuid4(),
            service_user_id=user.user_id,
            username=user.username,
            user_role=user.user_role,
            model_name=agent.llm_manager.config.model_name,
            embedding_name=agent.config.embeddings.model_name,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            message_count=0,
        )
        self.session.add(session_db)
        return session_db

    def create_session(self, user: UserORM, agent: Agent) -> SessionORM:
        """Create and persist a new session (synchronous).

        Parameters
        ----------
        user : :class:`~src.agentic_rag.persistence.user.UserORM`
            The current user ORM instance.
        agent : :class:`~src.agentic_rag.agent.graph.Agent`
            The agent instance used to populate model/embedding info.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.session.SessionORM`
            The committed and refreshed session row.
        """
        session_db = self._create_session(user, agent)
        self.session.commit()
        self.session.refresh(session_db)
        return session_db

    async def async_create_session(self, user: UserORM, agent: Agent) -> SessionORM:
        """Create and persist a new session (asynchronous).

        Parameters
        ----------
        user : :class:`~src.agentic_rag.persistence.user.UserORM`
            The current user ORM instance.
        agent : :class:`~src.agentic_rag.agent.graph.Agent`
            The agent instance used to populate model/embedding info.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.session.SessionORM`
            The committed and refreshed session row.
        """
        session_db = self._create_session(user, agent)
        await self.session.commit()
        await self.session.refresh(session_db)
        return session_db

    @staticmethod
    def _get_session_info_query(session_id: Optional[uuid.UUID] = None, user_id: Optional[uuid.UUID] = None):
        """Build a ``SELECT`` for session lookup.

        Exactly one of ``session_id`` or ``user_id`` must be provided.

        Parameters
        ----------
        session_id : :class:`uuid.UUID`, optional
            Filter by primary key.
        user_id : :class:`uuid.UUID`, optional
            Filter by ``user_id``.

        Returns
        -------
        :sqlalchemy:`Select <core/selectable.html#sqlalchemy.sql.expression.Select>`
            A select operator over :class:`~src.agentic_rag.persistence.session.SessionORM`.

        Raises
        ------
        AssertionError
            If both identifiers are missing.
        """
        assert (session_id or user_id)
        if session_id is not None:
            return select(SessionORM).where(SessionORM.session_id == session_id)
        if user_id is not None:
            return select(SessionORM).where(SessionORM.service_user_id == user_id)
        return None

    def sync_get_session(self, session_id: uuid.UUID) -> Optional[SessionORM]:
        """Fetch a session by id (synchronous)."""
        session_info_query = self._get_session_info_query(session_id=session_id)
        result = self.session.execute(session_info_query)
        return result.scalars().first()

    async def async_get_session(self, session_id: uuid.UUID) -> Optional[SessionORM]:
        """Fetch a session by id (asynchronous)."""
        session_info_query = self._get_session_info_query(session_id=session_id)
        result = await self.session.execute(session_info_query)
        return result.scalars().first()

    def sync_delete_session_by_id(self, session_id: uuid.UUID) -> Optional[SessionORM]:
        """Delete a session by id (synchronous)."""
        self.logger.info(f"Deleting session {session_id}")
        session_db = self.sync_get_session(session_id)
        if session_db is not None:
            self.session.delete(session_db)
            self.session.commit()
            return session_db
        else:
            self.logger.info(f"Session {session_id} was not found.")
            return None

    async def async_delete_session_by_id(self, session_id: uuid.UUID) -> Optional[SessionORM]:
        """Delete a session by id (asynchronous)."""
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
        """Update the title of a session (synchronous)."""
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
        """Update the title of a session (asynchronous)."""
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

    def sync_update_session_activity(self, session_id: uuid.UUID,
                                     increment_messages: bool = True) -> SessionORM | None:
        """Updates ``updated_at`` and optionally increment ``message_count`` (synchronous)."""
        session_info = self.sync_get_session(session_id)
        if session_info is None:
            self.logger.info(f"Session {session_id} was not found.")
            return None
        else:
            session_info.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if increment_messages:
                session_info.message_count += 1
            self.session.add(session_info)
            self.session.commit()
            self.session.refresh(session_info)
            return session_info

    async def async_update_session_activity(self, session_id: uuid.UUID,
                                            increment_messages: bool = True) -> SessionORM | None:
        """Updates ``updated_at`` and optionally increment ``message_count`` (asynchronous)."""
        session_info = await self.async_get_session(session_id)
        if session_info is None:
            self.logger.info(f"Session {session_id} was not found.")
            return None
        else:
            session_info.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session_info.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if increment_messages:
                session_info.message_count += 1
            self.session.add(session_info)
            await self.session.commit()
            await self.session.refresh(session_info)
            return session_info

    def sync_get_user_sessions(self, user_id: uuid.UUID) -> List[SessionORM] | None:
        """List all sessions for a given user (synchronous)."""
        session_info_query = self._get_session_info_query(user_id=user_id)
        result = self.session.execute(session_info_query)
        if result.scalars() is not None:
            return list(result.scalars().all())
        self.logger.info(f"User {user_id} was not found.")
        return None

    async def async_get_user_sessions(self, user_id: uuid.UUID) -> List[SessionORM] | None:
        """List all sessions for a given user (asynchronous)."""
        session_info_query = self._get_session_info_query(user_id=user_id)
        result = await self.session.execute(session_info_query)
        if result.scalars() is not None:
            return list(result.scalars().all())
        self.logger.info(f"User {user_id} was not found.")
        return None