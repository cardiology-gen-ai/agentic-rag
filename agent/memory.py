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
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.messages.utils import get_buffer_string


class Memory:
    def __init__(self, max_tokens: int = 2000, llm_model: str = "llama3.2:1b"):
        self.max_tokens = max_tokens
        self.llm = ChatOllama(model=llm_model, temperature=0.1)
        
        # Medical entities to preserve in memory
        self.medical_keywords = [
            'myocardial infarction', 'heart attack', 'acute coronary syndrome', 'stemi', 'nstemi',
            'heart failure', 'atrial fibrillation', 'cardiomyopathy', 'pci', 'cabg',
            'anticoagulation', 'antiplatelet', 'beta blocker', 'ace inhibitor', 'statin',
            'echocardiography', 'cardiac catheterization', 'esc guidelines', 'protocol'
        ]

    def messages_to_text(self, messages: List) -> str:
        """Convert list of messages to readable text format"""
        text_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                text_parts.append(f"AI: {msg.content}")
        return "\n".join(text_parts)

    def extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities from text"""
        entities = []
        text_lower = text.lower()
        
        for keyword in self.medical_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        # Extract patient information patterns
        patient_patterns = [
            r'patient with (\w+)',
            r'diagnosis of (\w+)',
            r'(\w+) protocol',
            r'esc (\w+) guidelines'
        ]
        
        for pattern in patient_patterns:
            matches = re.findall(pattern, text_lower)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates

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

    def summarize_medical_conversation(self, messages: List, existing_summary: str = "") -> str:
        """Create medically-aware summary preserving key clinical information"""
        messages_text = self.messages_to_text(messages)
        
        if existing_summary:
            prompt = f"""
            Previous summary: {existing_summary}

            New messages: {messages_text}

            Update the summary preserving:
            - Medical conditions discussed
            - ESC protocols/guidelines mentioned
            - Patient information
            - Clinical decisions made
            - Treatment recommendations

            Provide a concise medical summary:
            """
        else:
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
            print(f"Error creating summary: {e}")
            # Fallback to simple text extraction
            return f"Discussed: {', '.join(self.extract_medical_entities(messages_text))}"

    def get_conversation_context(self, state: Dict) -> str:
        """Get formatted conversation context for use in other nodes"""
        context_parts = []
        
        # Add conversation summary if available
        summary = state.get("conversation_summary", "")
        if summary:
            context_parts.append(f"Previous conversation summary: {summary}")
        
        # Add recent messages
        recent_messages = state.get("messages", [])[-2:]  # Last 2 messages for context
        if recent_messages:
            context_parts.append("Recent conversation:")
            context_parts.append(self.messages_to_text(recent_messages))
        
        # Add medical context if available
        medical_context = state.get("medical_context", [])
        if medical_context:
            context_parts.append(f"Medical topics discussed: {', '.join(medical_context)}")
        
        return "\n\n".join(context_parts)

    def memory_management_node(self, state: Dict) -> Dict:
        """Main LangGraph node function"""
        messages = state.get("messages", [])
        
        if not messages:
            return state
        
        # Estimate current token count
        current_tokens = self.estimate_token_count(messages)
        
        # Check if memory management needed
        if current_tokens < self.max_tokens:
            print(f"📊 Memory: {current_tokens}/{self.max_tokens} tokens - no cleanup needed")
            return state
        
        print(f"🧠 Memory management triggered: {current_tokens}/{self.max_tokens} tokens")
        
        # Get existing summary
        existing_summary = state.get("conversation_summary", "")
        
        # Create/update summary with older messages
        messages_to_summarize = messages[:-4]  # All but last 4 messages
        if messages_to_summarize:
            new_summary = self.summarize_medical_conversation(messages_to_summarize, existing_summary)
        else:
            new_summary = existing_summary
        
        # Keep only recent messages
        messages_to_keep = messages[-4:]  # Keep last 4 messages
        
        # Extract medical entities from the new summary
        medical_entities = self.extract_medical_entities(new_summary)
        
        print(f"📝 Updated summary with {len(medical_entities)} medical entities")
        print(f"🔄 Keeping {len(messages_to_keep)} recent messages")
        
        # Update state
        updated_state = {
            **state,
            "messages": messages_to_keep,
            "conversation_summary": new_summary,
            "medical_context": medical_entities
        }
        
        return updated_state


def test_memory():
    """Test the Memory node"""
    print("🧪 Testing Memory Node")
    print("=" * 50)

    # Initialize memory with low threshold for testing
    memory = Memory(max_tokens=100)

    # Create test messages
    test_messages = [
        HumanMessage(content="What is the ESC protocol for acute myocardial infarction?"),
        AIMessage(content="The ESC guidelines recommend immediate PCI for STEMI patients within 120 minutes of symptom onset. This is a Class I recommendation with Level A evidence."),
        HumanMessage(content="What about anticoagulation therapy?"),
        AIMessage(content="ESC recommends dual antiplatelet therapy with aspirin and clopidogrel for acute coronary syndrome. Heparin should also be administered during PCI."),
        HumanMessage(content="How about heart failure management?"),
        AIMessage(content="ESC heart failure guidelines recommend ACE inhibitors and beta blockers as first-line therapy. Diuretics are used for symptom relief."),
        HumanMessage(content="What are the contraindications for beta blockers?"),
        AIMessage(content="Contraindications include severe bradycardia, high-degree heart block, and decompensated heart failure without optimal medical therapy."),
    ]

    # Test state
    test_state = {
        "messages": test_messages,
        "query_type": "document_based"
    }

    print(f"Initial state: {len(test_messages)} messages")
    print(f"Estimated tokens: {memory.estimate_token_count(test_messages)}")

    # Process through memory management
    result_state = memory.memory_management_node(test_state)

    # Display results
    print(f"\nResults:")
    print(f"Messages remaining: {len(result_state.get('messages', []))}")
    
    summary = result_state.get('conversation_summary', 'None')
    print(f"Summary: {summary[:150]}..." if len(summary) > 150 else f"Summary: {summary}")
    
    medical_context = result_state.get('medical_context', [])
    print(f"Medical entities: {medical_context}")

    # Test context retrieval
    print(f"\n📋 Testing context retrieval:")
    context = memory.get_conversation_context(result_state)
    print(f"Context preview: {context[:200]}...")

    # Test with no memory management needed
    print(f"\n🔄 Testing with short conversation:")
    short_state = {
        "messages": test_messages[:2],
        "query_type": "document_based"
    }
    
    short_result = memory.memory_management_node(short_state)
    print(f"Short conversation - messages: {len(short_result.get('messages', []))}")
    print(f"Summary created: {'Yes' if short_result.get('conversation_summary') else 'No'}")


if __name__ == "__main__":
    test_memory()
