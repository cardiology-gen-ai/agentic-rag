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
    def __init__(self, db_connection_string: Optional[str] = None, sync: Optional[bool] = True):
        self.db_connection_string = db_connection_string if db_connection_string is not None else None
        if self.db_connection_string is None:
            self.db_connection_string = os.getenv("SYNC_DB_CONNECTION_STRING") if sync \
                else os.getenv("ASYNC_DB_CONNECTION_STRING")
        if self.db_connection_string is None:
            raise ValueError("No database connection string provided")
        logger.info(f"Database connection initialized")


def get_db_connection(db_connection_string: Optional[str] = None, sync: Optional[bool] = True) -> DatabaseConnection:
    return DatabaseConnection(db_connection_string, sync)


def get_engine(db_connection_string: Optional[str] = None, sync: bool = True) -> AsyncEngine | Engine:
    db_connection = get_db_connection(db_connection_string, sync)
    if sync:
        engine: Engine = create_engine(db_connection.db_connection_string, echo=False, pool_pre_ping=True)
    else:
        engine: AsyncEngine = create_async_engine(db_connection.db_connection_string, echo=False)
    return engine


def get_sync_db(db_connection_string: Optional[str] = None):
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
    engine = get_engine(db_connection_string, sync=False)
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()