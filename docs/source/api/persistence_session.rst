Session ORM and Create Read Update Delete (CRUD) Management
===========================================================

This module provides a :sqlalchemy:`SQLAlchemy <>` -based ORM mapping for user sessions** and a both synchronous and
asynchronous CRUD interface for managing session records.

.. autoclass:: src.persistence.session.SessionORM
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:

    .. py:attribute:: session_id
      :type: uuid.UUID

      Primary key; generated via :func:`uuid.uuid4`.

    .. py:attribute:: service_user_id
      :type: uuid.UUID

      Identifier of the owning service user.

    .. py:attribute:: title
      :type: str

      Optional human-readable session title.

    .. py:attribute:: username
      :type: str

      Username associated with the session.

    .. py:attribute:: user_role
      :type: str

      Role of the user.

    .. py:attribute:: model_name
      :type: str

      Name of the LLM model deployment used in the session.

    .. py:attribute:: embedding_name
      :type: str

      Name of the embedding model deployment used in the session.

    .. py:attribute:: created_at
      :type: datetime.datetime

      Creation timestamp (stored as naive UTC).

    .. py:attribute:: updated_at
      :type: datetime.datetime

      Last update timestamp (stored as naive UTC).

    .. py:attribute:: message_count
      :type: int

      Number of messages exchanged in the session.


.. autoclass:: src.persistence.session.SessionSchema
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:


.. autoclass:: src.persistence.session.SessionDB
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:
