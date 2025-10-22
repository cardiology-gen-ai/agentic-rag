import uuid
from datetime import datetime, timezone
from typing import Optional, Literal, Union
import asyncio

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, Session

from agentic_rag.persistence.orm_base import BaseORM, BaseDB
from agentic_rag.persistence.db import get_sync_db, get_async_db, ensure_database


class UserORM(BaseORM):
    """ORM mapping for the ``public.user`` table."""
    __tablename__ = "user"
    __table_args__ = {"schema": "public", "extend_existing": True}
    user_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4) #: :class:`~sqlalchemy.orm.Mapped`[:class:`uuid.UUID`] : Primary key; generated via :func:`uuid.uuid4`.
    username: Mapped[str] = mapped_column(nullable=False) #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] :  Unique username for the user.
    user_role: Mapped[str] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Role of the user (e.g. ``"user"``, ``"admin"``, ``"assistant"``).
    email: Mapped[str] = mapped_column(nullable=False) #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Email address.
    created_at: Mapped[datetime] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`datetime`] : Creation timestamp (stored as naive UTC).
    last_active: Mapped[datetime] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`datetime`] : Last activity timestamp (stored as naive UTC).

class UserCreateSchema(BaseModel):
    """Pydantic input schema with the minimal information necessary to create a user."""
    username: str #: :class:`str` : Desired username.
    email: str #: :class:`str` : Email address.


class UserSchema(UserCreateSchema):
    """Pydantic schema mirroring :class:`~src.agentic_rag.persistence.user.UserORM` for I/O and validation."""
    user_id: uuid.UUID #: :class:`uuid.UUID` : Primary key of the user.
    user_role: Literal["user", "admin", "assistant"] #: :class:`typing.Literal`\[{``user``, ``admin``, ``assistant``}\] : Role assigned to the user.


class UserDB(BaseDB):
    """User DataBase CRUD (Create-Read-Update-Delete), sync and async, backed by SQLAlchemy.

    On construction, this class ensures that the :class:`~src.agentic_rag.persistence.user.UserORM` table exists
    by calling :sqlalchemy:`sqlalchemy.MetaData.create_all <core/metadata.html#sqlalchemy.schema.MetaData.create_all>` for that table only.

    .. rubric:: Notes

    Timestamps are stored as *naive* UTC datetimes using ``datetime.now(timezone.utc).replace(tzinfo=None)``
    to avoid timezone-aware values in the database.

    Parameters
    ----------
    session : :sqlalchemy:`AsyncSession <orm/extensions/asyncio.html#sqlalchemy.ext.asyncio.AsyncSession>` or :sqlalchemy:`Session </orm/session_api.html#sqlalchemy.orm.Session>`
        SQLAlchemy session used for all operations.

    """
    def __init__(self, session: Union[AsyncSession, Session]):
        super().__init__(session=session)
        ensure_database()
        engine = session.bind
        BaseORM.metadata.create_all(engine, tables=[UserORM.__table__])

    def _create_user(self, user: UserCreateSchema) -> UserORM:
        """Construct a new :class:`~src.agentic_rag.persistence.user.UserORM` (not committed).

        Parameters
        ----------
        user : :class:`~src.agentic_rag.persistence.user.UserCreateSchema`
            Input data with ``username`` and ``email``.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The newly constructed user object.
        """
        user_id = uuid.uuid4()
        user_db = UserORM(
            user_id=user_id,
            username=user.username,
            email=user.email,
            user_role="user",
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            last_active=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.session.add(user_db)
        return user_db

    def sync_create_user(self, user: UserCreateSchema) -> UserORM:
        """Create and persist a new user (synchronous).

        Parameters
        ----------
        user : :class:`~src.agentic_rag.persistence.user.UserCreateSchema`
            Input data for user creation.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The committed and refreshed user row.
        """
        user_db = self._create_user(user)
        self.session.commit()
        self.session.refresh(user_db)
        return user_db

    async def async_create_user(self, user: UserCreateSchema) -> UserORM:
        """Create and persist a new user (asynchronous).

        Parameters
        ----------
        user : :class:`~src.agentic_rag.persistence.user.UserCreateSchema`
            Input data for user creation.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The committed and refreshed user row.
        """
        user_db = self._create_user(user)
        await self.session.commit()
        await self.session.refresh(user_db)
        return user_db

    @staticmethod
    def _get_user_info_query(username: Optional[str] = None,
                             email: Optional[str] = None,
                             user_id: Optional[uuid.UUID] = None):
        """Build a ``SELECT`` for user lookup.

        Exactly one of ``username``, ``email``, or ``user_id`` must be provided.

        Parameters
        ----------
        username : :class:`str`, optional
            Filter by username.
        email : :class:`str`, optional
            Filter by email.
        user_id : :class:`uuid.UUID`, optional
            Filter by primary key.

        Returns
        -------
        sqlalchemy:`Select <core/selectable.html#sqlalchemy.sql.expression.Select>`
            A select operator over :class:`~src.agentic_rag.persistence.user.UserORM`.

        Raises
        ------
        AssertionError
            If all identifiers are missing.
        """
        assert (username or email or user_id)
        if user_id is not None:
            return select(UserORM).where(UserORM.user_id == user_id)
        if username is not None:
            return select(UserORM).where(UserORM.username == username)
        if email is not None:
            return select(UserORM).where(UserORM.email == email)
        return None

    def sync_get_user(self,
                      username: Optional[str] = None,
                      email: Optional[str] = None,
                      user_id: Optional[uuid.UUID] = None) -> Optional[UserORM]:
        """Fetch a user by one of ``username``, ``email``, or ``user_id`` (synchronous)."""
        user_info_query = self._get_user_info_query(username, email, user_id)
        result = self.session.execute(user_info_query)
        return result.scalars().first()

    async def async_get_user(self,
                             username: Optional[str] = None,
                             email: Optional[str] = None,
                             user_id: Optional[uuid.UUID] = None) -> Optional[UserORM]:
        """Fetch a user by one of ``username``, ``email``, or ``user_id`` (asynchronous)."""
        user_info_query = self._get_user_info_query(username, email, user_id)
        result = await self.session.execute(user_info_query)
        return result.scalars().first()

    def sync_update_user_activity(self, user_id: uuid.UUID) -> UserORM | None:
        """Set :attr:`~src.agentic_rag.persistence.user.UserORM.last_active` to now for the given user (synchronous).

        Parameters
        ----------
        user_id : :class:`uuid.UUID`
            Identifier of the user to touch.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM` or ``None``
            Updated user row if found, otherwise ``None``.
        """
        user_info: UserORM | None = self.sync_get_user(user_id=user_id)
        if not user_info:
            self.logger.info(f"No user info found for user_id {user_id}")
            return None
        user_info.last_active = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.commit()
        self.session.refresh(user_info)
        return user_info

    async def async_update_user_activity(self, user_id: uuid.UUID) -> UserORM | None:
        """Set :attr:`~src.agentic_rag.persistence.user.UserORM.last_active` to now for the given user (asynchronous).

        Parameters
        ----------
        user_id : :class:`uuid.UUID`
            Identifier of the user to touch.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM` or ``None``
            Updated user row if found, otherwise ``None``.
        """
        user_info: UserORM | None = await self.async_get_user(user_id=user_id)
        if not user_info:
            self.logger.info(f"No user info found for user_id {user_id}")
            return None
        user_info.last_active = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.session.commit()
        await self.session.refresh(user_info)
        return user_info


if __name__ == "__main__":
    sync = True
    if sync:
        session_generator = get_sync_db()
        current_session = next(session_generator)
        try:
            current_user_db = UserDB(current_session)
            my_user = current_user_db.sync_create_user(user=UserCreateSchema(username="sync_test2", email=""))
        finally:
            try:
                next(session_generator)
            except StopIteration:
                pass
    else:
        session_generator = get_async_db()
        current_session = asyncio.run(anext(session_generator))
        try:
            current_user_db = UserDB(current_session)
            my_user = asyncio.run(current_user_db.async_create_user(user=UserCreateSchema(username="async_test2", email="")))
        finally:
            session_generator.aclose()