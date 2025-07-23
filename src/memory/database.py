#!/usr/bin/env python3
"""
PostgreSQL Database Manager for the Cardiology Agent.
Handles state persistence, conversation storage, and memory management.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import asyncpg
import psycopg2
from psycopg2.extras import Json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from state import State

class Manager:
    """
    PostgreSQL database manager for agent state persistence.
    
    Handles:
    - Conversation state storage and retrieval
    - Message history persistence
    - Medical context tracking
    - Session management
    """
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://root:root@localhost:5432/postgres"
        )
        self.pool = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            await self.create_tables()
            self.logger.info("Database manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database pool closed")
    
    async def create_tables(self):
        """Create necessary tables for agent state persistence"""
        async with self.pool.acquire() as conn:
            # Enable uuid extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            
            # Conversations table - main conversation sessions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) UNIQUE NOT NULL,
                    conversation_summary TEXT,
                    medical_context JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Messages table - individual messages in conversations
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (thread_id) REFERENCES conversations(thread_id) ON DELETE CASCADE
                );
            """)
            
            # Create indexes for messages table
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
            """)
            
            # Agent states table - for storing agent internal states
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) NOT NULL,
                    state_type VARCHAR(100) NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create indexes for agent_states table
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_thread_id ON agent_states(thread_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_type ON agent_states(state_type);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_created_at ON agent_states(created_at);
            """)
            
            # Feedback table - for storing user feedback on responses
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) NOT NULL,
                    message_id VARCHAR(255),
                    is_positive BOOLEAN NOT NULL,
                    comment TEXT,
                    feedback_type VARCHAR(50) DEFAULT 'response',
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create indexes for feedback table
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_thread_id ON feedback(thread_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_is_positive ON feedback(is_positive);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
            """)
            
            # Update timestamp trigger
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
            
            await conn.execute("""
                DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
                CREATE TRIGGER update_conversations_updated_at
                    BEFORE UPDATE ON conversations
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """)
    
    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Serialize a LangChain message to JSON-compatible format"""
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
            "response_metadata": getattr(message, "response_metadata", {})
        }
    
    def _deserialize_message(self, data: Dict[str, Any]) -> BaseMessage:
        """Deserialize JSON data back to LangChain message"""
        message_type = data["type"]
        content = data["content"]
        
        if message_type == "HumanMessage":
            return HumanMessage(
                content=content,
                additional_kwargs=data.get("additional_kwargs", {}),
                response_metadata=data.get("response_metadata", {})
            )
        elif message_type == "AIMessage":
            return AIMessage(
                content=content,
                additional_kwargs=data.get("additional_kwargs", {}),
                response_metadata=data.get("response_metadata", {})
            )
        else:
            # Fallback for other message types
            return HumanMessage(content=content)
    
    async def save_conversation_state(self, state: ConversationState) -> str:
        """
        Save or update conversation state in database.
        
        Args:
            state: ConversationState object to save
            
        Returns:
            str: Conversation ID
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert or update conversation
                conv_id = await conn.fetchval("""
                    INSERT INTO conversations (thread_id, conversation_summary, medical_context, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (thread_id) 
                    DO UPDATE SET 
                        conversation_summary = EXCLUDED.conversation_summary,
                        medical_context = EXCLUDED.medical_context,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING id;
                """, 
                    state.thread_id,
                    state.conversation_summary,
                    json.dumps(state.medical_context or []),
                    json.dumps(state.metadata or {})
                )
                
                # Delete existing messages for this conversation
                await conn.execute("""
                    DELETE FROM messages WHERE thread_id = $1;
                """, state.thread_id)
                
                # Insert new messages
                for message in state.messages:
                    message_data = self._serialize_message(message)
                    await conn.execute("""
                        INSERT INTO messages (thread_id, message_type, content, metadata)
                        VALUES ($1, $2, $3, $4);
                    """,
                        state.thread_id,
                        message_data["type"],
                        message_data["content"],
                        json.dumps({
                            "additional_kwargs": message_data.get("additional_kwargs", {}),
                            "response_metadata": message_data.get("response_metadata", {})
                        })
                    )
                
                return str(conv_id)
    
    async def load_conversation_state(self, thread_id: str) -> Optional[ConversationState]:
        """
        Load conversation state from database.
        
        Args:
            thread_id: Session identifier
            
        Returns:
            ConversationState or None if not found
        """
        async with self.pool.acquire() as conn:
            # Get conversation data
            conv_row = await conn.fetchrow("""
                SELECT id, thread_id, conversation_summary, medical_context, metadata, created_at, updated_at
                FROM conversations 
                WHERE thread_id = $1;
            """, thread_id)
            
            if not conv_row:
                return None
            
            # Get messages
            message_rows = await conn.fetch("""
                SELECT message_type, content, metadata, timestamp
                FROM messages 
                WHERE thread_id = $1
                ORDER BY timestamp ASC;
            """, thread_id)
            
            # Reconstruct messages
            messages = []
            for row in message_rows:
                message_data = {
                    "type": row['message_type'],
                    "content": row['content'],
                    **json.loads(row['metadata'])
                }
                messages.append(self._deserialize_message(message_data))
            
            return ConversationState(
                thread_id=conv_row['thread_id'],
                messages=messages,
                conversation_summary=conv_row['conversation_summary'],
                medical_context=json.loads(conv_row['medical_context']) if conv_row['medical_context'] else [],
                metadata=json.loads(conv_row['metadata']) if conv_row['metadata'] else {},
                created_at=conv_row['created_at'],
                updated_at=conv_row['updated_at']
            )
    
    async def save_agent_state(self, thread_id: str, state_type: str, state_data: Dict[str, Any]) -> str:
        """
        Save agent internal state.
        
        Args:
            thread_id: Session identifier
            state_type: Type of state (e.g., 'routing', 'memory', 'rag')
            state_data: State data to save
            
        Returns:
            str: State ID
        """
        async with self.pool.acquire() as conn:
            state_id = await conn.fetchval("""
                INSERT INTO agent_states (thread_id, state_type, state_data)
                VALUES ($1, $2, $3)
                RETURNING id;
            """,
                thread_id,
                state_type,
                json.dumps(state_data)
            )
            return str(state_id)
    
    async def load_agent_state(self, thread_id: str, state_type: str) -> Optional[Dict[str, Any]]:
        """
        Load latest agent state of given type.
        
        Args:
            thread_id: Session identifier
            state_type: Type of state to load
            
        Returns:
            Dict with state data or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT state_data, created_at
                FROM agent_states
                WHERE thread_id = $1 AND state_type = $2
                ORDER BY created_at DESC
                LIMIT 1;
            """, thread_id, state_type)
            
            if row:
                return json.loads(row['state_data'])
            return None
    
    async def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT thread_id, conversation_summary, medical_context, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT $1;
            """, limit)
            
            return [
                {
                    "thread_id": row['thread_id'],
                    "summary": row['conversation_summary'],
                    "medical_context": json.loads(row['medical_context']) if row['medical_context'] else [],
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at']
                }
                for row in rows
            ]
    
    async def save_feedback(self, thread_id: str, is_positive: bool, comment: str = None, 
                           message_id: str = None, feedback_type: str = "response") -> str:
        """
        Save user feedback for a response.
        
        Args:
            thread_id: Session identifier
            is_positive: True for positive feedback, False for negative
            comment: Optional feedback comment
            message_id: Optional message identifier
            feedback_type: Type of feedback (response, conversation, etc.)
            
        Returns:
            str: Feedback ID
        """
        async with self.pool.acquire() as conn:
            feedback_id = await conn.fetchval("""
                INSERT INTO feedback (thread_id, is_positive, comment, message_id, feedback_type)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id;
            """, thread_id, is_positive, comment, message_id, feedback_type)
            return str(feedback_id)
    
    async def get_feedback_stats(self, thread_id: str = None) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            thread_id: Optional session to filter by
            
        Returns:
            Dict with feedback statistics
        """
        async with self.pool.acquire() as conn:
            if thread_id:
                total_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feedback WHERE thread_id = $1;
                """, thread_id)
                
                positive_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feedback WHERE thread_id = $1 AND is_positive = true;
                """, thread_id)
                
                negative_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feedback WHERE thread_id = $1 AND is_positive = false;
                """, thread_id)
            else:
                total_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feedback;
                """)
                
                positive_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feedback WHERE is_positive = true;
                """)
                
                negative_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feedback WHERE is_positive = false;
                """)
            
            positive_percentage = (positive_count / total_count * 100) if total_count > 0 else 0.0
            
            return {
                "total_feedback": total_count or 0,
                "positive_count": positive_count or 0,
                "negative_count": negative_count or 0,
                "positive_percentage": positive_percentage
            }
    
    async def delete_conversation(self, thread_id: str) -> bool:
        """
        Delete a conversation and all related data.
        
        Args:
            thread_id: Session identifier
            
        Returns:
            bool: True if deleted, False if not found
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Delete conversation (messages will be deleted by CASCADE)
                result = await conn.execute("""
                    DELETE FROM conversations WHERE thread_id = $1;
                """, thread_id)
                
                # Delete related agent states
                await conn.execute("""
                    DELETE FROM agent_states WHERE thread_id = $1;
                """, thread_id)
                
                return result != 'DELETE 0'
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        async with self.pool.acquire() as conn:
            yield conn


# Synchronous wrapper for backward compatibility
class SyncDatabaseManager:
    """Synchronous wrapper around the async DatabaseManager"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://root:root@localhost:5432/postgres"
        )
        self.conn = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize synchronous database connection"""
        try:
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            self.create_tables()
            self.logger.info("Sync database manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize sync database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
    
    def create_tables(self):
        """Create necessary tables (synchronous version)"""
        with self.conn.cursor() as cur:
            # Enable uuid extension
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            
            # Same table creation as async version but using psycopg2
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) UNIQUE NOT NULL,
                    conversation_summary TEXT,
                    medical_context JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (thread_id) REFERENCES conversations(thread_id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) NOT NULL,
                    state_type VARCHAR(100) NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_agent_states_thread_id ON agent_states(thread_id);
                CREATE INDEX IF NOT EXISTS idx_agent_states_type ON agent_states(state_type);
                CREATE INDEX IF NOT EXISTS idx_agent_states_created_at ON agent_states(created_at);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    thread_id VARCHAR(255) NOT NULL,
                    message_id VARCHAR(255),
                    is_positive BOOLEAN NOT NULL,
                    comment TEXT,
                    feedback_type VARCHAR(50) DEFAULT 'response',
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_feedback_thread_id ON feedback(thread_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_is_positive ON feedback(is_positive);
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
            """)
    
    def save_simple_state(self, thread_id: str, state_data: Dict[str, Any]) -> str:
        """
        Simple state saving for basic use cases.
        
        Args:
            thread_id: Session identifier
            state_data: State data to save
            
        Returns:
            str: State ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_states (thread_id, state_type, state_data)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (thread_id, 'general', Json(state_data)))
            
            return str(cur.fetchone()[0])
    
    def load_simple_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Load latest general state for session.
        
        Args:
            thread_id: Session identifier
            
        Returns:
            Dict with state data or None if not found
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT state_data
                FROM agent_states
                WHERE thread_id = %s AND state_type = 'general'
                ORDER BY created_at DESC
                LIMIT 1;
            """, (thread_id,))
            
            row = cur.fetchone()
            if row:
                return row[0]
            return None
    
    def save_feedback(self, thread_id: str, is_positive: bool, comment: str = None, 
                     message_id: str = None, feedback_type: str = "response") -> str:
        """
        Save user feedback for a response.
        
        Args:
            thread_id: Session identifier
            is_positive: True for positive feedback, False for negative
            comment: Optional feedback comment
            message_id: Optional message identifier
            feedback_type: Type of feedback (response, conversation, etc.)
            
        Returns:
            str: Feedback ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (thread_id, is_positive, comment, message_id, feedback_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (thread_id, is_positive, comment, message_id, feedback_type))
            
            return str(cur.fetchone()[0])
    
    def get_feedback_stats(self, thread_id: str = None) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            thread_id: Optional session to filter by
            
        Returns:
            Dict with feedback statistics
        """
        with self.conn.cursor() as cur:
            if thread_id:
                cur.execute("""
                    SELECT COUNT(*) FROM feedback WHERE thread_id = %s;
                """, (thread_id,))
                total_count = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*) FROM feedback WHERE thread_id = %s AND is_positive = true;
                """, (thread_id,))
                positive_count = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*) FROM feedback WHERE thread_id = %s AND is_positive = false;
                """, (thread_id,))
                negative_count = cur.fetchone()[0]
            else:
                cur.execute("""
                    SELECT COUNT(*) FROM feedback;
                """)
                total_count = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*) FROM feedback WHERE is_positive = true;
                """)
                positive_count = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*) FROM feedback WHERE is_positive = false;
                """)
                negative_count = cur.fetchone()[0]
            
            positive_percentage = (positive_count / total_count * 100) if total_count > 0 else 0.0
            
            return {
                "total_feedback": total_count or 0,
                "positive_count": positive_count or 0,
                "negative_count": negative_count or 0,
                "positive_percentage": positive_percentage
            }


