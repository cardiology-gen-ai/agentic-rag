import os
from typing import Optional
from pathlib import Path

import psycopg
from psycopg import sql
from dotenv import load_dotenv

from cardiology_gen_ai.utils.logger import get_logger
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

POSTGRES_ADMIN_DSN = os.getenv("POSTGRES_ADMIN_DSN")
DB_NAME = "cardiology_protocols"  # TODO: maybe put in config

logger = get_logger("Database creation")

with psycopg.connect(POSTGRES_ADMIN_DSN, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cur.fetchone() is not None

        if not exists:
            # Crea in modo sicuro (quotando l'identificatore)
            stmt = sql.SQL(
                "CREATE DATABASE {} WITH ENCODING 'UTF8' TEMPLATE template1 {}"
            ).format(
                sql.Identifier(DB_NAME), sql.SQL("")
            )
            cur.execute(stmt)
            logger.info(f"Database {DB_NAME} successfully created")
        else:
            logger.info(f"Database {DB_NAME} already exists")


class DatabaseConnection:
    """Construct the connection string according to sync/async mode.

    If ``db_connection_string`` is not provided, the string is built from the
    ``DB_CONNECTION_STRING`` environment variable with the appropriate driver
    prefix for the requested connection type.

    Parameters
    ----------
    db_connection_string : str, optional
    Full application DSN. If ``None``, it is constructed dynamically.
    sync : bool, optional
    If ``True`` (default) uses the synchronous driver (:class:`psycopg`);
    if ``False`` uses the asynchronous driver (:class:`asyncpg`).

    Raises
    ------
    ValueError
    If no connection string is available.
    """
    db_connection_str: str #: str : The SQLAlchemy/psycopg connection string.
    def __init__(self, db_connection_string: Optional[str] = None, sync: Optional[bool] = True):
        self.db_connection_string = db_connection_string if db_connection_string is not None else None
        if self.db_connection_string is None:
            prefix = "postgresql+psycopg://" if sync is True else "postgresql+asyncpg://"
            self.db_connection_string = prefix + os.getenv("DB_CONNECTION_STRING")
            if self.db_connection_string is None:
                raise ValueError("No database connection string provided")
        logger.info(f"Database connection initialized")


def get_db_connection(db_connection_string: Optional[str] = None, sync: Optional[bool] = True) -> DatabaseConnection:
    """Return a :class:`~src.persistence.db.DatabaseConnection` instance.

    Parameters
    ----------
    db_connection_string : str, optional
    Full application DSN. If ``None``, it is built based on ``sync`` and the
    ``DB_CONNECTION_STRING`` environment variable.
    sync : bool, optional
    ``True`` to create a connection for a synchronous context, otherwise for an
    asynchronous context.


    Returns
    -------
    :class:`~src.persistence.db.DatabaseConnection`
    The configured connection object.
    """
    return DatabaseConnection(db_connection_string, sync)


def get_engine(db_connection_string: Optional[str] = None, sync: bool = True) -> AsyncEngine | Engine:
    """Create a :class:`SQLAlchemy engine` (sync) or async engine (async).

    Parameters
    ----------
    db_connection_string : str, optional
    Full DSN. If ``None``, it is derived from :class:`DatabaseConnection`.
    sync : bool, optional
    ``True`` for :class:`sqlalchemy.engine.Engine`, ``False`` for  :class:`sqlalchemy.ext.asyncio.AsyncEngine`.


    Returns
    -------
     :class:`sqlalchemy.ext.asyncio.AsyncEngine` or :class:`sqlalchemy.engine.Engine`
    The created engine.
    """
    db_connection = get_db_connection(db_connection_string, sync)
    if sync:
        engine: Engine = create_engine(db_connection.db_connection_string, echo=False, pool_pre_ping=True)
    else:
        engine: AsyncEngine = create_async_engine(db_connection.db_connection_string, echo=False)
    return engine


def get_sync_db(db_connection_string: Optional[str] = None):
    """Yield a synchronous ORM session and handle cleanup.

    Parameters
    ----------
    db_connection_string : str, optional
    Full DSN. If ``None``, it is derived automatically.

    Yields
    ------
    :class:`sqlalchemy.orm.Session`
    The synchronous ORM session.
    """
    engine = get_engine(db_connection_string, sync=True)
    session_maker = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_maker()
    try:
        yield session
    finally:
        if session:
            session.close()
        if engine:
            engine.dispose()


async def get_async_db(db_connection_string: Optional[str] = None):
    """Yield an asynchronous ORM session and handle cleanup.

    Parameters
    ----------
    db_connection_string : str, optional
    Full DSN. If ``None``, it is derived automatically.

    Yields
    ------
    :class:`sqlalchemy.ext.asyncio.AsyncSession`
    The asynchronous ORM session.
    """
    engine = get_engine(db_connection_string, sync=False)
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()