#!/usr/bin/env python3

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langgraph.store.postgres import PostgresStore # type: ignore


@dataclass
class ChatSession:
    """Represents a chat session"""
    session_id: str
    user_id: str
    created_at: str
    updated_at: str
    title: str = ""
    message_count: int = 0


@dataclass
class UserFeedback:
    """Represents user feedback on AI responses"""
    feedback_id: str
    session_id: str
    user_id: str
    message_id: str
    is_positive: bool
    comment: Optional[str]
    timestamp: str


# Namespace constants
USERS_NS = ("users",)
SESSIONS_NS = ("sessions",)  
FEEDBACK_NS = ("feedback",)
SESSION_METADATA_NS = ("session_metadata",)


def create_user(store: PostgresStore, user_id: str, username: str, email: str = "") -> bool:
    """Create a new user record"""
    try:
        user_data = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        }
        store.put(USERS_NS, user_id, user_data)
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

def get_user(store: PostgresStore, user_id: str) -> Optional[Dict]:
    """Get user information"""
    try:
        result = store.get(USERS_NS, user_id)
        return result.value if result else None
    except Exception:
        return None

def update_user_activity(store: PostgresStore, user_id: str) -> bool:
    """Update user's last activity timestamp"""
    try:
        user_data = get_user(store, user_id)
        if user_data:
            user_data["last_active"] = datetime.now().isoformat()
            store.put(USERS_NS, user_id, user_data)
            return True
        return False
    except Exception:
        return False

def create_session(store: PostgresStore, user_id: str, session_id: str = None, title: str = "") -> str:
    """Create a new chat session"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Create session metadata
        session_data = {
            "session_id": session_id,
            "user_id": user_id, 
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "title": title or f"Chat {session_id[:8]}",
            "message_count": 0,
            "status": "active"
        }
        
        # Store session metadata
        store.put(SESSION_METADATA_NS, session_id, session_data)
        
        # Add session to user's session list
        user_sessions = get_user_sessions(store, user_id)
        user_sessions.append(session_id)
        store.put(SESSIONS_NS, user_id, {"sessions": user_sessions})
        
        return session_id
    except Exception as e:
        print(f"Error creating session: {e}")
        return None

def get_session(store: PostgresStore, session_id: str) -> Optional[ChatSession]:
    """Get session information"""
    try:
        result = store.get(SESSION_METADATA_NS, session_id)
        if result and result.value:
            data = result.value
            return ChatSession(
                session_id=data["session_id"],
                user_id=data["user_id"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                title=data["title"],
                message_count=data["message_count"]
            )
        return None
    except Exception:
        return None

def get_user_sessions(store: PostgresStore, user_id: str) -> List[str]:
    """Get list of session IDs for a user"""
    try:
        result = store.get(SESSIONS_NS, user_id)
        if result and result.value:
            return result.value.get("sessions", [])
        return []
    except Exception:
        return []

def get_user_sessions_with_metadata(store: PostgresStore, user_id: str) -> List[ChatSession]:
    """Get user sessions with full metadata"""
    session_ids = get_user_sessions(store, user_id)
    sessions = []
    
    for session_id in session_ids:
        session = get_session(store, session_id)
        if session:
            sessions.append(session)
    
    # Sort by updated_at descending (most recent first)
    sessions.sort(key=lambda x: x.updated_at, reverse=True)
    return sessions

def update_session_activity(store: PostgresStore, session_id: str, increment_messages: bool = False) -> bool:
    """Update session's last activity and optionally increment message count"""
    try:
        result = store.get(SESSION_METADATA_NS, session_id)
        if result and result.value:
            session_data = result.value
            session_data["updated_at"] = datetime.now().isoformat()
            
            if increment_messages:
                session_data["message_count"] += 1
            
            store.put(SESSION_METADATA_NS, session_id, session_data)
            return True
        return False
    except Exception:
        return False

def delete_session(store: PostgresStore, session_id: str) -> bool:
    """Delete a session and remove from user's session list"""
    try:
        # Get session to find user_id
        session = get_session(store, session_id)
        if not session:
            return False
        
        # Remove from user's sessions list
        user_sessions = get_user_sessions(store, session.user_id)
        if session_id in user_sessions:
            user_sessions.remove(session_id)
            store.put(SESSIONS_NS, session.user_id, {"sessions": user_sessions})
        
        # Delete session metadata
        store.delete(SESSION_METADATA_NS, session_id)
        
        # Delete any feedback for this session
        _delete_session_feedback(store, session_id)
        
        return True
    except Exception as e:
        print(f"Error deleting session: {e}")
        return False

def save_feedback(store: PostgresStore, session_id: str, user_id: str, message_id: str, 
                 is_positive: bool, comment: str = None) -> str:
    """Save user feedback for a specific message"""
    try:
        feedback_id = str(uuid.uuid4())
        feedback_data = {
            "feedback_id": feedback_id,
            "session_id": session_id,
            "user_id": user_id,
            "message_id": message_id,
            "is_positive": is_positive,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store feedback with composite key: session_id:feedback_id
        feedback_key = f"{session_id}:{feedback_id}"
        store.put(FEEDBACK_NS, feedback_key, feedback_data)
        
        return feedback_id
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return None

def get_session_feedback(store: PostgresStore, session_id: str) -> List[UserFeedback]:
    """Get all feedback for a session"""
    try:
        # Search for feedback with session_id prefix
        results = store.search(FEEDBACK_NS, query=f"{session_id}:")
        feedback_list = []
        
        for result in results:
            if result.value:
                data = result.value
                feedback = UserFeedback(
                    feedback_id=data["feedback_id"],
                    session_id=data["session_id"],
                    user_id=data["user_id"],
                    message_id=data["message_id"],
                    is_positive=data["is_positive"],
                    comment=data.get("comment"),
                    timestamp=data["timestamp"]
                )
                feedback_list.append(feedback)
        
        return feedback_list
    except Exception as e:
        print(f"Error getting session feedback: {e}")
        return []

def get_user_feedback(store: PostgresStore, user_id: str) -> List[UserFeedback]:
    """Get all feedback from a user"""
    try:
        results = store.search(FEEDBACK_NS)
        feedback_list = []
        
        for result in results:
            if result.value and result.value.get("user_id") == user_id:
                data = result.value
                feedback = UserFeedback(
                    feedback_id=data["feedback_id"],
                    session_id=data["session_id"],
                    user_id=data["user_id"],
                    message_id=data["message_id"],
                    is_positive=data["is_positive"],
                    comment=data.get("comment"),
                    timestamp=data["timestamp"]
                )
                feedback_list.append(feedback)
        
        return feedback_list
    except Exception as e:
        print(f"Error getting user feedback: {e}")
        return []

def _delete_session_feedback(store: PostgresStore, session_id: str) -> bool:
    """Delete all feedback for a session"""
    try:
        results = store.search(FEEDBACK_NS, query=f"{session_id}:")
        for result in results:
            store.delete(FEEDBACK_NS, result.key)
        return True
    except Exception:
        return False

def get_feedback_stats(store: PostgresStore, user_id: str = None, session_id: str = None) -> Dict[str, int]:
    """Get feedback statistics"""
    try:
        if session_id:
            feedback_list = get_session_feedback(store, session_id)
        elif user_id:
            feedback_list = get_user_feedback(store, user_id)
        else:
            # Get all feedback
            results = store.search(FEEDBACK_NS)
            feedback_list = []
            for result in results:
                if result.value:
                    data = result.value
                    feedback = UserFeedback(
                        feedback_id=data["feedback_id"],
                        session_id=data["session_id"],
                        user_id=data["user_id"],
                        message_id=data["message_id"],
                        is_positive=data["is_positive"],
                        comment=data.get("comment"),
                        timestamp=data["timestamp"]
                    )
                    feedback_list.append(feedback)
        
        positive_count = sum(1 for f in feedback_list if f.is_positive)
        negative_count = len(feedback_list) - positive_count
        
        return {
            "total": len(feedback_list),
            "positive": positive_count,
            "negative": negative_count
        }
    except Exception:
        return {"total": 0, "positive": 0, "negative": 0}