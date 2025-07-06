#!/usr/bin/env python3
"""
SQLite database manager.
"""

import json
import sqlite3
import uuid
from typing import Dict, Optional
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
        
        # Create states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                user_id TEXT,
                query_type TEXT,
                query TEXT,
                response TEXT,
                conversation_summary TEXT,
                metadata TEXT,  -- JSON string
                retrieval_attempts INTEGER,
                generation_attempts INTEGER,
                next_action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                state_id TEXT,
                message_type TEXT,  -- 'human' or 'ai'
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (state_id) REFERENCES agent_states (id)
            )
        ''')
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                state_id TEXT,
                page_content TEXT,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (state_id) REFERENCES agent_states (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_state(self, state: Dict) -> str:
        """Save state to database and return state ID."""
        state_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main state
        cursor.execute('''
            INSERT INTO agent_states (
                id, thread_id, user_id, query_type, query, response,
                conversation_summary, metadata, retrieval_attempts,
                generation_attempts, next_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state_id,
            state.get("thread_id"),
            state.get("user_id"),
            state.get("query_type"),
            state.get("query"),
            state.get("response"),
            state.get("conversation_summary"),
            json.dumps(state.get("metadata", {})),
            state.get("retrieval_attempts"),
            state.get("generation_attempts"),
            state.get("next_action")
        ))
        
        # Save messages
        messages = state.get("messages", [])
        for msg in messages:
            msg_id = str(uuid.uuid4())
            msg_type = "human" if isinstance(msg, HumanMessage) else "ai"
            cursor.execute('''
                INSERT INTO messages (id, state_id, message_type, content)
                VALUES (?, ?, ?, ?)
            ''', (msg_id, state_id, msg_type, msg.content))
        
        # Save documents
        documents = state.get("documents", [])
        for doc in documents:
            doc_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO documents (id, state_id, page_content, metadata)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, state_id, doc.page_content, json.dumps(doc.metadata)))
        
        conn.commit()
        conn.close()
        return state_id
    
    def load_state(self, state_id: str) -> Optional[Dict]:
        """Load state from database by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load main state
        cursor.execute('SELECT * FROM agent_states WHERE id = ?', (state_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        state = {
            "thread_id": row[1],
            "user_id": row[2],
            "query_type": row[3],
            "query": row[4],
            "response": row[5],
            "conversation_summary": row[6],
            "metadata": json.loads(row[7]) if row[7] else {},
            "retrieval_attempts": row[8],
            "generation_attempts": row[9],
            "next_action": row[10]
        }
        
        # Load messages
        cursor.execute('SELECT message_type, content FROM messages WHERE state_id = ? ORDER BY created_at', (state_id,))
        messages = []
        for msg_type, content in cursor.fetchall():
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        state["messages"] = messages
        
        # Load documents
        cursor.execute('SELECT page_content, metadata FROM documents WHERE state_id = ?', (state_id,))
        documents = []
        for content, metadata_str in cursor.fetchall():
            metadata = json.loads(metadata_str) if metadata_str else {}
            documents.append(Document(page_content=content, metadata=metadata))
        state["documents"] = documents
        
        conn.close()
        return state