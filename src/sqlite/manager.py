#!/usr/bin/env python3
"""
SQLite database manager.
"""

import json
import sqlite3
import uuid
from typing import Dict, Optional, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# SQLite State Manager
class StateManager:
    """Manages state persistence in SQLite database."""
    
    def __init__(self, db_path: str = "agent_states.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                message_type TEXT,  -- 'human' or 'ai'
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                page_content TEXT,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                message_id TEXT,
                value INTEGER,  -- 0 for negative, 1 for positive
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, user_id: str) -> str:
        """Create a new user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO users (id) VALUES (?)
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        return user_id
    
    def create_conversation(self, user_id: str, title: str = None) -> str:
        """Create a new conversation for a user."""
        conversation_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)
        ''', (conversation_id, user_id, title))
        
        conn.commit()
        conn.close()
        return conversation_id
    
    def add_message(self, conversation_id: str, message_type: str, content: str) -> str:
        """Add a message to a conversation."""
        message_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, message_type, content)
            VALUES (?, ?, ?, ?)
        ''', (message_id, conversation_id, message_type, content))
        
        conn.commit()
        conn.close()
        return message_id
    
    def add_document(self, conversation_id: str, page_content: str, metadata: Dict = None) -> str:
        """Add a document to a conversation."""
        document_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (id, conversation_id, page_content, metadata)
            VALUES (?, ?, ?, ?)
        ''', (document_id, conversation_id, page_content, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
        return document_id
    
    def add_feedback(self, message_id: str, value: int, comment: str = None) -> str:
        """Add feedback to a message."""
        feedback_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (id, message_id, value, comment)
            VALUES (?, ?, ?, ?)
        ''', (feedback_id, message_id, value, comment))
        
        conn.commit()
        conn.close()
        return feedback_id
    
    def get_conversation_messages(self, conversation_id: str) -> List:
        """Get all messages for a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, message_type, content, created_at
            FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at ASC
        ''', (conversation_id,))
        
        messages = []
        for msg_id, msg_type, content, created_at in cursor.fetchall():
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        
        conn.close()
        return messages
    
    def get_conversation_documents(self, conversation_id: str) -> List[Document]:
        """Get all documents for a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT page_content, metadata 
            FROM documents 
            WHERE conversation_id = ?
        ''', (conversation_id,))
        
        documents = []
        for content, metadata_str in cursor.fetchall():
            metadata = json.loads(metadata_str) if metadata_str else {}
            documents.append(Document(page_content=content, metadata=metadata))
        
        conn.close()
        return documents
    
    def get_message_feedback(self, message_id: str) -> Optional[Dict]:
        """Get feedback for a specific message."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, value, comment, created_at, updated_at
            FROM feedback 
            WHERE message_id = ?
        ''', (message_id,))
        
        row = cursor.fetchone()
        if row:
            feedback = {
                "id": row[0],
                "message_id": message_id,
                "value": row[1],
                "comment": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            }
            conn.close()
            return feedback
        
        conn.close()
        return None
    
    def get_user_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, created_at, updated_at
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY updated_at DESC
        ''', (user_id,))
        
        conversations = []
        for conv_id, title, created_at, updated_at in cursor.fetchall():
            conversations.append({
                "id": conv_id,
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at
            })
        
        conn.close()
        return conversations
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM conversations WHERE id = ? LIMIT 1', (conversation_id,))
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversations 
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (title, conversation_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def update_feedback(self, feedback_id: str, value: int, comment: str = None) -> bool:
        """Update existing feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE feedback 
            SET value = ?, comment = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (value, comment, feedback_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
