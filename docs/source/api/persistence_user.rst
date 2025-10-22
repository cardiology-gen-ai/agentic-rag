User Database
=============

This module provides an interface for managing user entities within the application.
It defines the :sqlalchemy:`SQLAlchemy <>` ORM mapping (:class:`~src.persistence.user.UserORM`) for the `public.user` table, alongside :pydantic:`Pydantic BaseModel <base_model>`
(:class:`~src.persistence.user.UserCreateSchema` and :class:`~src.persistence.user.UserSchema`)
for input validation and data interchange.


.. autoclass:: src.agentic_rag.persistence.user.UserORM
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:

    .. py:attribute:: user_id
        :type: uuid.UUID

        Primary key; generated via :func:`uuid.uuid4`.

    .. py:attribute:: username
        :type: str

        Unique username for the user.

    .. py:attribute:: user_role
        :type: str

        Role of the user (e.g. ``"user"``, ``"admin"``, ``"assistant"``).

    .. py:attribute:: email
        :type: str

        Email address.

    .. py:attribute:: created_at
        :type: datetime.datetime

        Creation timestamp (stored as naive UTC).

    .. py:attribute:: last_active
        :type: datetime.datetime

        Last activity timestamp (stored as naive UTC).


.. autoclass:: src.agentic_rag.persistence.user.UserCreateSchema
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:


.. autoclass:: src.agentic_rag.persistence.user.UserSchema
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:


.. autoclass:: src.agentic_rag.persistence.user.UserDB
    :members:
    :undoc-members:
    :exclude-members: from_config, model_config
    :show-inheritance:

