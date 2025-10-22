Database Connection and Base Object Relational Model
====================================================

This module provides **utility classes and functions** to manage database connections
and :sqlalchemy:`SQLAlchemy <>` sessions, supporting both synchronous and asynchronous contexts.
Moreover, it provides base classes for database session management and ORM modeling
using SQLAlchemy, supporting both synchronous and asynchronous workflows.

.. autoclass:: src.agentic_rag.persistence.db.DatabaseConnection
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: src.agentic_rag.persistence.db.get_db_connection

.. autofunction:: src.agentic_rag.persistence.db.get_engine

.. autofunction:: src.agentic_rag.persistence.db.get_sync_db

.. autofunction:: src.agentic_rag.persistence.db.get_async_db

.. automodule:: src.agentic_rag.persistence.orm_base
   :members:
   :exclude-members: from_config, model_config
   :member-order: bysource