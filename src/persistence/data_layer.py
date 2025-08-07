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


class DataLayer:
    """Data access layer for chat sessions and feedback using LangGraph PostgresStore"""
    
    def __init__(self, store: PostgresStore):
        self.store = store
        
        # Namespace constants
        self.USERS_NS = ("users",)
        self.SESSIONS_NS = ("sessions",)  
        self.FEEDBACK_NS = ("feedback",)
        self.SESSION_METADATA_NS = ("session_metadata",)
    
    def create_user(self, user_id: str, username: str, email: str = "") -> bool:
        """Create a new user record"""
        try:
            user_data = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
            self.store.put(self.USERS_NS, user_id, user_data)
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user information"""
        try:
            result = self.store.get(self.USERS_NS, user_id)
            return result.value if result else None
        except Exception:
            return None
    
    def update_user_activity(self, user_id: str) -> bool:
        """Update user's last activity timestamp"""
        try:
            user_data = self.get_user(user_id)
            if user_data:
                user_data["last_active"] = datetime.now().isoformat()
                self.store.put(self.USERS_NS, user_id, user_data)
                return True
            return False
        except Exception:
            return False
    
    def create_session(self, user_id: str, session_id: str = None, title: str = "") -> str:
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
            self.store.put(self.SESSION_METADATA_NS, session_id, session_data)
            
            # Add session to user's session list
            user_sessions = self.get_user_sessions(user_id)
            user_sessions.append(session_id)
            self.store.put(self.SESSIONS_NS, user_id, {"sessions": user_sessions})
            
            return session_id
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session information"""
        try:
            result = self.store.get(self.SESSION_METADATA_NS, session_id)
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
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get list of session IDs for a user"""
        try:
            result = self.store.get(self.SESSIONS_NS, user_id)
            if result and result.value:
                return result.value.get("sessions", [])
            return []
        except Exception:
            return []
    
    def get_user_sessions_with_metadata(self, user_id: str) -> List[ChatSession]:
        """Get user sessions with full metadata"""
        session_ids = self.get_user_sessions(user_id)
        sessions = []
        
        for session_id in session_ids:
            session = self.get_session(session_id)
            if session:
                sessions.append(session)
        
        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return sessions
    
    def update_session_activity(self, session_id: str, increment_messages: bool = False) -> bool:
        """Update session's last activity and optionally increment message count"""
        try:
            result = self.store.get(self.SESSION_METADATA_NS, session_id)
            if result and result.value:
                session_data = result.value
                session_data["updated_at"] = datetime.now().isoformat()
                
                if increment_messages:
                    session_data["message_count"] += 1
                
                self.store.put(self.SESSION_METADATA_NS, session_id, session_data)
                return True
            return False
        except Exception:
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and remove from user's session list"""
        try:
            # Get session to find user_id
            session = self.get_session(session_id)
            if not session:
                return False
            
            # Remove from user's sessions list
            user_sessions = self.get_user_sessions(session.user_id)
            if session_id in user_sessions:
                user_sessions.remove(session_id)
                self.store.put(self.SESSIONS_NS, session.user_id, {"sessions": user_sessions})
            
            # Delete session metadata
            self.store.delete(self.SESSION_METADATA_NS, session_id)
            
            # Delete any feedback for this session
            self._delete_session_feedback(session_id)
            
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def save_feedback(self, session_id: str, user_id: str, message_id: str, 
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
            self.store.put(self.FEEDBACK_NS, feedback_key, feedback_data)
            
            return feedback_id
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return None
    
    def get_session_feedback(self, session_id: str) -> List[UserFeedback]:
        """Get all feedback for a session"""
        try:
            # Search for feedback with session_id prefix
            results = self.store.search(self.FEEDBACK_NS, query=f"{session_id}:")
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
    
    def get_user_feedback(self, user_id: str) -> List[UserFeedback]:
        """Get all feedback from a user"""
        try:
            results = self.store.search(self.FEEDBACK_NS)
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
    
    def _delete_session_feedback(self, session_id: str) -> bool:
        """Delete all feedback for a session"""
        try:
            results = self.store.search(self.FEEDBACK_NS, query=f"{session_id}:")
            for result in results:
                self.store.delete(self.FEEDBACK_NS, result.key)
            return True
        except Exception:
            return False
    
    def get_feedback_stats(self, user_id: str = None, session_id: str = None) -> Dict[str, int]:
        """Get feedback statistics"""
        try:
            if session_id:
                feedback_list = self.get_session_feedback(session_id)
            elif user_id:
                feedback_list = self.get_user_feedback(user_id)
            else:
                # Get all feedback
                results = self.store.search(self.FEEDBACK_NS)
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