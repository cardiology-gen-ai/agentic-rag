import logging

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Session

from cardiology_gen_ai.utils.logger import get_logger


logger = get_logger("Public DB handling based on Postgres via sqlalchemy")


class BaseDB:
    def __init__(self, session: AsyncSession | Session):
        self.session = session
        self.logger: logging.Logger = logger
        self.sync: bool = not isinstance(self.session, AsyncSession)


class BaseORM(DeclarativeBase):
    metadata = MetaData()
