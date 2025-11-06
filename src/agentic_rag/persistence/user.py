import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Literal, Union, cast
import asyncio

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy.orm import Mapped, mapped_column, Session
import jwt
from jwt import InvalidTokenError
from fastapi import HTTPException
from starlette import status

from agentic_rag.persistence.orm_base import BaseORM, BaseDB
from agentic_rag.persistence.db import get_sync_db, get_async_db, ensure_database
from agentic_rag.persistence.authentication import Authentication


class UserORM(BaseORM):
    """ORM mapping for the ``public.user`` table."""
    __tablename__ = "user"
    __table_args__ = {"schema": "public", "extend_existing": True}
    user_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4) #: :class:`~sqlalchemy.orm.Mapped`[:class:`uuid.UUID`] : Primary key; generated via :func:`uuid.uuid4`.
    username: Mapped[str] = mapped_column(nullable=False) #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] :  Unique username for the user.
    user_role: Mapped[str] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Role of the user (e.g. ``"user"``, ``"admin"``, ``"assistant"``).
    email: Mapped[str] = mapped_column(nullable=False) #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Email address.
    hashed_password: Mapped[str] = mapped_column(nullable=False)  #: :class:`~sqlalchemy.orm.Mapped`[:class:`str`] : Hashed user password for authentication.
    created_at: Mapped[datetime] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`datetime`] : Creation timestamp (stored as naive UTC).
    last_active: Mapped[datetime] = mapped_column() #: :class:`~sqlalchemy.orm.Mapped`[:class:`datetime`] : Last activity timestamp (stored as naive UTC).
    disabled: Mapped[bool] = mapped_column()  #: :class:`~sqlalchemy.orm.Mapped`[:class:`bool`] : Whether the user has currently a valid access token.


class UserCreateSchema(BaseModel):
    """Pydantic input schema with the minimal information necessary to create a user."""
    username: str #: :class:`str` : Desired username.
    email: str #: :class:`str` : Email address.
    password: str #: :class:`str` : Password.


class UserSchema(UserCreateSchema):
    """Pydantic schema mirroring :class:`~src.agentic_rag.persistence.user.UserORM` for I/O and validation."""
    user_id: uuid.UUID #: :class:`uuid.UUID` : Primary key of the user.
    user_role: Literal["user", "admin", "assistant"] #: :class:`typing.Literal`\[{``user``, ``admin``, ``assistant``}\] : Role assigned to the user.
    disabled: Optional[bool] = None  #: :class:`bool` : Whether the user has currently a valid access token.

    @classmethod
    def from_orm(cls, user_orm: UserORM) -> "UserSchema":
        return cls(
            username=user_orm.username,
            email=user_orm.email,
            password=user_orm.hashed_password,
            user_role=cast(Literal["user", "admin", "assistant"], user_orm.user_role),
            disabled=user_orm.disabled,
            user_id=user_orm.user_id,
        )


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
        admin_dsn = os.getenv("POSTGRES_ADMIN_DSN")
        if admin_dsn:
            ensure_database()
        engine = session.bind
        if isinstance(engine, AsyncEngine):
            asyncio.run(self._acreate_all(engine=engine))
        else:
            BaseORM.metadata.create_all(engine, tables=[UserORM.__table__])

    @staticmethod
    async def _acreate_all(engine: AsyncEngine):
        async with engine.begin() as conn:
            await conn.run_sync(BaseORM.metadata.create_all)

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
            disabled=False,
        )
        self.session.add(user_db)
        return user_db

    def create_user(self, user: UserCreateSchema) -> UserORM:
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

    async def acreate_user(self, user: UserCreateSchema) -> UserORM:
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

    def get_user(self,
                 username: Optional[str] = None,
                 email: Optional[str] = None,
                 user_id: Optional[uuid.UUID] = None) -> Optional[UserORM]:
        """Fetch a user by one of ``username``, ``email``, or ``user_id`` (synchronous)."""
        user_info_query = self._get_user_info_query(username, email, user_id)
        result = self.session.execute(user_info_query)
        return result.scalars().first()

    async def aget_user(self,
                        username: Optional[str] = None,
                        email: Optional[str] = None,
                        user_id: Optional[uuid.UUID] = None) -> Optional[UserORM]:
        """Fetch a user by one of ``username``, ``email``, or ``user_id`` (asynchronous)."""
        user_info_query = self._get_user_info_query(username, email, user_id)
        result = await self.session.execute(user_info_query)
        return result.scalars().first()

    def authenticate(self, username: str, password: str) -> Optional[UserORM]:
        """Authenticate a user synchronously.

        Retrieves the user from the database by username and verifies that the provided password matches the stored hash.

        Parameters
        ----------
        username : :class:`str`
            The username of the account to authenticate.
        password : :class:`str`
            The plaintext password provided by the user.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The authenticated user object if verification succeeds; otherwise, ``None``.
        """
        user_orm = self.get_user(username=username)
        if not user_orm:
            return None
        if not Authentication.verify_password(password, user_orm.hashed_password):
            return None
        return user_orm

    async def aauthenticate(self, username: str, password: str) -> Optional[UserORM]:
        """Authenticate a user asynchronously.
        Retrieves the user and verifies the password using asynchronous database calls.

        Parameters
        ----------
        username : :class:`str`
            The username of the account to authenticate.
        password : :class:`str`
            The plaintext password provided by the user.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The authenticated user object if verification succeeds; otherwise, ``None``.
        """
        user_orm = await self.aget_user(username=username)
        if not user_orm:
            return None
        if not Authentication.verify_password(password, user_orm.hashed_password):
            return None
        return user_orm

    @staticmethod
    def get_username_by_token(token: str) -> str:
        """Extract the username from a :jwt:`JWT <api.html>` token.

        Decodes the :jwt:`JWT <api.html>` token using the secret key and retrieves the
        ``sub`` (username) and ``role`` claims. Raises an :starlette:`HTTP 401 exception <exceptions>` if the token is invalid or incomplete.

        Parameters
        ----------
        token : :class:`str`
            The user's :jwt:`JWT <api.html>` token.

        Returns
        -------
        :class:`str`
            The username extracted from the token.

        Raises
        ------
        :fastapi:`HTTPException <exceptions/?h=httpe>`
            If the token is invalid or credential validation fails.
        """
        credential_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credential validation failed",
            headers={"WWW-Authenticate": "Bearer"}
        )
        try:
            user_payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
            username = user_payload.get("sub")
            user_role = user_payload.get("role")
            if username is None or user_role is None:
                raise credential_exception
        except InvalidTokenError:
            raise credential_exception
        return username

    def get_user_by_token(self, token: str) -> Optional[UserORM]:
        """Retrieve a user from a :jwt:`JWT <api.html>` token.

        Decodes the token, extracts the username, and retrieves the corresponding user object from the database. Raises an exception if the user does not exist or the token is invalid.

        Parameters
        ----------
        token : :class:`str`
            The user's :jwt:`JWT <api.html>` token.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The user object associated with the token.

        Raises
        ------
        :fastapi:`HTTPException <exceptions/?h=httpe>`
            If the token is invalid or the user does not exist.
        """
        username = self.get_username_by_token(token)
        user_orm = self.get_user(username=username)
        if user_orm is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Credential validation failed",
                                headers={"WWW-Authenticate": "Bearer"})
        return user_orm

    async def aget_user_by_token(self, token: str) -> Optional[UserORM]:
        """Retrieve a user from a :jwt:`JWT <api.html>` token asynchronously.

        Decodes the token, extracts the username, and retrieves the corresponding user object from the database. Raises an exception if the user does not exist or the token is invalid.

        Parameters
        ----------
        token : :class:`str`
            The user's :jwt:`JWT <api.html>` token.

        Returns
        -------
        :class:`~src.agentic_rag.persistence.user.UserORM`
            The user object associated with the token.

        Raises
        ------
        :fastapi:`HTTPException <exceptions/?h=httpe>`
            If the token is invalid or the user does not exist.
        """
        username = self.get_username_by_token(token)
        user_orm = await self.aget_user(username=username)
        if user_orm is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Credential validation failed",
                                headers={"WWW-Authenticate": "Bearer"})
        return user_orm

    def update_user_activity(self, user_id: uuid.UUID) -> UserORM | None:
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
        user_info: UserORM | None = self.get_user(user_id=user_id)
        if not user_info:
            self.logger.info(f"No user info found for user_id {user_id}")
            return None
        user_info.last_active = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.commit()
        self.session.refresh(user_info)
        return user_info

    async def aupdate_user_activity(self, user_id: uuid.UUID) -> UserORM | None:
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
        user_info: UserORM | None = await self.aget_user(user_id=user_id)
        if not user_info:
            self.logger.info(f"No user info found for user_id {user_id}")
            return None
        user_info.last_active = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.session.commit()
        await self.session.refresh(user_info)
        return user_info

    def delete_user(self, user_id: uuid.UUID) -> None:
        user_db = self.get_user(user_id=user_id)
        if user_db is not None:
            self.session.delete(user_db)
            self.session.commit()
        return None

    async def adelete_user(self, user_id: uuid.UUID) -> None:
        user_db = await self.aget_user(user_id=user_id)
        if user_db is not None:
            await self.session.delete(user_db)
            await self.session.commit()
        return None


if __name__ == "__main__":
    sync = True
    if sync:
        session_generator = get_sync_db()
        current_session = next(session_generator)
        try:
            current_user_db = UserDB(current_session)
            my_user = current_user_db.create_user(user=UserCreateSchema(username="sync_test2", email="", password="password"))
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
            my_user = asyncio.run(current_user_db.acreate_user(user=UserCreateSchema(username="async_test2", email="", password="password")))
        finally:
            session_generator.aclose()