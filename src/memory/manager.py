#!/usr/bin/env python3
"""
Memory node for the Cardiology Protocols Pipeline.
Enables the agent to remember previous messages (short memory) and/or user preferences (long memory).

Takes as input:
    - current state with messages

Gives as output:
    - updated states with conversation context
"""

import re
import uuid
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, BaseMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.messages.utils import get_buffer_string
from database import SyncDatabaseManager, ConversationState
import configs
from state import State

class Memory:
    def __init__(self, max_tokens: int = configs.MEMORY_MAX_TOKENS,
                 llm_model: str = configs.LLM_MODEL,
                 temperature: float = configs.MEMORY_LLM_TEMPERATURE,
                 thread_id: str = None):
        self.max_tokens = max_tokens
        self.llm = ChatOllama(model=llm_model, temperature = temperature)
        self.thread_id = thread_id or str(uuid.uuid4())
        
        # Initialize database manager
        self.db_manager = SyncDatabaseManager()
        try:
            self.db_manager.initialize()
            if configs.DEBUG:
                print("✅ Database manager initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize database manager: {e}")
            self.db_manager = None
        
    def messages_to_text(self, messages: List) -> str:
        """Convert list of messages to readable text format"""
        text_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                text_parts.append(f"AI: {msg.content}")
        return "\n".join(text_parts)
    
    def summarize_conversation(self, messages: List[BaseMessage]) -> str:
        """Create medically-aware summary preserving key clinical information"""
        messages_text = self.messages_to_text(messages)
        
        prompt = f"""
        Create a summary of this medical conversation preserving:
        - Key medical information
        - ESC protocols discussed
        - Clinical context
        - Treatment recommendations

        Messages: {messages_text}

        Provide a concise medical summary:
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            # Fallback to simple text extraction
            return f"Discussed: {', '.join(self.extract_medical_entities(messages_text))}"

    def estimate_token_count(self, messages: List) -> int:
        """Estimate token count for messages"""
        try:
            # Use LangChain's buffer string method
            text = get_buffer_string(messages)
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4
        except:
            # Fallback method
            total_length = sum(len(str(msg.content)) for msg in messages if hasattr(msg, 'content'))
            return total_length // 4

    def update_state(self, state: State) -> State:
        """Main LangGraph node function"""
        previous_messages = state.get("previous_messages", [])
        
        if not previous_messages:
            return state

        current_tokens = self.estimate_token_count(previous_messages)
        
        # Check if memory management needed
        if current_tokens < self.max_tokens:
            return state
        
        if configs.DEBUG:
            print(f"Memory management triggered.")
        
        summary = self.summarize_conversation(previous_messages)
        
        state['conversation_summary'] = summary
        
        # Save to database if available
        if self.db_manager:
            try:
                conversation_state = ConversationState(
                    thread_id=self.thread_id,
                    messages=previous_messages,
                    conversation_summary=summary,
                    metadata=state.get("metadata", {})
                )
                self.save_to_database(conversation_state)
            except Exception as e:
                print(f"⚠️  Warning: Could not save to database: {e}")
        
        return state
    
    def save_to_database(self, conversation_state: ConversationState):
        """Save conversation state to database"""
        if self.db_manager:
            # For now, use the simple state method
            state_data = {
                "messages": [
                    {
                        "type": msg.__class__.__name__,
                        "content": msg.content
                    } for msg in conversation_state.messages
                ],
                "conversation_summary": conversation_state.conversation_summary,
                "metadata": conversation_state.metadata
            }
            self.db_manager.save_simple_state(self.thread_id, state_data)
    
    def load_from_database(self) -> Optional[Dict]:
        """Load conversation state from database"""
        if self.db_manager:
            try:
                return self.db_manager.load_simple_state(self.thread_id)
            except Exception as e:
                print(f"⚠️  Warning: Could not load from database: {e}")
        return None
    
    def restore_conversation_state(self, state: Dict) -> Dict:
        """Restore conversation from database if available"""
        if self.db_manager:
            saved_state = self.load_from_database()
            if saved_state:
                print(f"Restored conversation from database")
                # Reconstruct messages
                messages = []
                for msg_data in saved_state.get("messages", []):
                    if msg_data["type"] == "HumanMessage":
                        messages.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data["type"] == "AIMessage":
                        messages.append(AIMessage(content=msg_data["content"]))
                
                # Update state with restored data
                state.update({
                    "messages": messages,
                    "conversation_summary": saved_state.get("conversation_summary"),
                    "medical_context": saved_state.get("medical_context", []),
                    "metadata": saved_state.get("metadata", {})
                })
        
        return state
    
    def save_feedback(self, is_positive: bool, comment: str = None, message_id: str = None) -> Optional[str]:
        """
        Save feedback for the last response.
        
        Args:
            is_positive: True for positive feedback, False for negative
            comment: Optional feedback comment
            message_id: Optional message identifier
            
        Returns:
            str: Feedback ID if saved, None if database unavailable
        """
        if self.db_manager:
            try:
                return self.db_manager.save_feedback(
                    self.thread_id, 
                    is_positive, 
                    comment, 
                    message_id, 
                    "response"
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not save feedback: {e}")
        return None
    
    def close(self):
        """Close database connection"""
        if self.db_manager:
            self.db_manager.close()


