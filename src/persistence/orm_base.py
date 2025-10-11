import logging

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Session

from cardiology_gen_ai.utils.logger import get_logger


logger = get_logger("Public DB handling based on Postgres via sqlalchemy")


class BaseDB:
    """Lightweight wrapper around a SQLAlchemy session (sync or async).

    This base class stores the bound session and exposes a boolean flag indicating whether the session is synchronous.

    Parameters
    ----------
    session : :class:`~sqlalchemy.ext.asyncio.AsyncSession` or :class:`~sqlalchemy.orm.Session`
        Bound SQLAlchemy session instance.
    """
    session: AsyncSession | Session #: :class:`~sqlalchemy.ext.asyncio.AsyncSession` or :class:`~sqlalchemy.orm.Session` : The underlying SQLAlchemy session.
    logger: logging.Logger #: :class:`logging.Logger` : Logger used for diagnostics (provided externally as ``logger``).
    sync: bool #: :class:`bool` : ``True`` if the session is a synchronous :class:`~sqlalchemy.orm.Session`, ``False`` if it is an asynchronous :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
    def __init__(self, session: AsyncSession | Session):
        self.session = session
        self.logger = logger
        self.sync = not isinstance(self.session, AsyncSession)


class BaseORM(DeclarativeBase):
    """Declarative base for SQLAlchemy ORM models.

    Attributes
    ----------
    metadata : :class:`~sqlalchemy.schema.MetaData`
        Metadata collection used by the declarative base.
    """
    metadata = MetaData()
