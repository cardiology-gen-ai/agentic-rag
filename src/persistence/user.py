import uuid
from datetime import datetime, timezone
from typing import Optional, Literal
import asyncio

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, Session

from src.persistence.orm_base import BaseORM, BaseDB
from src.persistence.db import get_sync_db, get_async_db


class UserORM(BaseORM):
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}  # TODO: check if needed
    user_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(nullable=False)
    user_role: Mapped[str] = mapped_column()
    email: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column()
    last_active: Mapped[datetime] = mapped_column()


class UserCreateSchema(BaseModel):
    username: str
    email: str


class UserSchema(UserCreateSchema):
    user_id: uuid.UUID
    user_role: Literal["user", "admin", "assistant"]


class UserDB(BaseDB):
    def __init__(self, session: AsyncSession | Session):
        super().__init__(session=session)
        engine = session.bind
        BaseORM.metadata.create_all(engine, tables=[UserORM.__table__])

    def _create_user(self, user: UserCreateSchema) -> UserORM:
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
        user_db = self._create_user(user)
        self.session.commit()
        self.session.refresh(user_db)
        return user_db

    async def async_create_user(self, user: UserCreateSchema) -> UserORM:
        user_db = self._create_user(user)
        await self.session.commit()
        await self.session.refresh(user_db)
        return user_db

    @staticmethod
    def _get_user_info_query(username: Optional[str] = None,
                             email: Optional[str] = None,
                             user_id: Optional[uuid.UUID] = None):
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
        user_info_query = self._get_user_info_query(username, email, user_id)
        result = self.session.execute(user_info_query)
        return result.scalars().first()

    async def async_get_user(self,
                             username: Optional[str] = None,
                             email: Optional[str] = None,
                             user_id: Optional[uuid.UUID] = None) -> Optional[UserORM]:
        user_info_query = self._get_user_info_query(username, email, user_id)
        result = await self.session.execute(user_info_query)
        return result.scalars().first()

    def sync_update_user_activity(self, user_id: uuid.UUID) -> UserORM | None:
        user_info: UserORM | None = self.sync_get_user(user_id=user_id)
        if not user_info:
            self.logger.info(f"No user info found for user_id {user_id}")
            return None
        user_info.last_active = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.commit()
        self.session.refresh(user_info)
        return user_info

    async def async_update_user_activity(self, user_id: uuid.UUID) -> UserORM | None:
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
            current_session.close()
    else:
        session_generator = get_async_db()
        current_session = asyncio.run(anext(session_generator))
        try:
            current_user_db = UserDB(current_session)
            my_user = asyncio.run(current_user_db.async_create_user(user=UserCreateSchema(username="async_test2", email="")))
        finally:
            session_generator.aclose()