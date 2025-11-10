import os
from datetime import timedelta, datetime, timezone
from typing import Dict, Optional

from pwdlib import PasswordHash
import jwt

from agentic_rag.persistence.orm_base import BaseDB


pwd_context = PasswordHash.recommended()


# TODO: check if it makes sense that this inherits from BaseDB
class Authentication(BaseDB):
    """Authentication and security utility class.

    Provides static methods for password hashing, password verification,
    and :jwt:`JWT <api.html>` token. token creation. It uses bcrypt via :passlib:`Passlib <>` for secure password
    management and :jwt:`PyJWT <>` for token encoding.
    """
    @staticmethod
    def verify_password(pwd: str, hashed_pwd: str) -> bool:
        """Verify that a plaintext password matches a hashed password.

        Parameters
        ----------
        pwd : :class:`str`
            The plaintext password to verify.
        hashed_pwd : :class:`str`
            The hashed password stored in the database.

        Returns
        -------
        :class:`bool`
            ``True`` if the password matches the hash; otherwise, ``False``.
        """
        return pwd_context.verify(pwd, hashed_pwd)

    @staticmethod
    def hash_password(password: str) -> str:
        """Generate a hash for a plaintext password.

        Parameters
        ----------
        password : :class:`str`
            The plaintext password to hash.

        Returns
        -------
        :class:`str`
            The hashed password.
        """
        return pwd_context.hash(password)

    @staticmethod
    async def create_access_token(data: Dict, validity_min: Optional[timedelta]) -> str:
        """Create a JSON Web Token (JWT) for authentication.

        Encodes user data and expiration time into a signed JWT string using
        the secret key defined in the ``SECRET_KEY`` environment variable.

        Parameters
        ----------
        data : :class:`dict`
            A dictionary containing the payload to encode (e.g., ``sub``, ``role``).
        validity_min : :class:`datetime.timedelta`, optional
            Token validity duration. If not provided, defaults to 24 hours.

        Returns
        -------
        :class:`str`
            The encoded JWT access token.
        """
        encode_data = data.copy()
        if validity_min:
            expiration = datetime.now(timezone.utc).replace(tzinfo=None) + validity_min
        else:
            expiration = datetime.now(timezone.utc) + timedelta(minutes=60*24)
        encode_data.update({"exp": expiration})
        encoded_jwt = jwt.encode(encode_data, os.getenv("SECRET_KEY"), algorithm="HS256")
        return encoded_jwt
