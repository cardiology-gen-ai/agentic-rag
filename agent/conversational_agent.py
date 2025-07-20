#!/usr/bin/env python3
"""
Conversational Agent for the Cardiology Protocols Pipeline.
Handles non-medical queries with friendly, helpful responses.

This agent:
1. Responds to greetings, farewells, and general conversation
2. Provides system information and capabilities
3. Guides users towards medical query functionality
4. Maintains conversation context and personality
5. Handles edge cases and fallback scenarios
"""

import re
from typing import Dict, List, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


class ConversationalAgent:
    """
    Conversational agent for handling non-medical queries and maintaining 
    friendly interaction in the cardiology protocols pipeline.
    """
    
    def __init__(
        self,
        llm_model: str = 'llama3.2:1b',
        temperature: float = 0.7,
        system_name: str = "Cardiology Assistant"
    ):
        self.llm_model = llm_model
        self.temperature = temperature
        self.system_name = system_name
        self._setup_llm()
        self._setup_prompts()
        self._setup_response_templates()

    def _setup_llm(self):
        """Initialize the language model for conversational responses."""
        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=self.temperature,
                verbose=False
            )
            print(f"✅ Conversational Agent LLM initialized: {self.llm_model}")
        except Exception as e:
            print(f"❌ Error initializing conversational LLM: {e}")
            raise

    def _setup_prompts(self):
        """Setup prompts for different conversational scenarios."""
        
        self.general_conversation_prompt = ChatPromptTemplate.from_template(
            """You are a helpful and friendly medical assistant specializing in cardiology protocols from the European Society of Cardiology (ESC).

Your personality:
- Warm, professional, and approachable
- Knowledgeable about your capabilities
- Encouraging users to ask medical questions
- Clear about your limitations

Current conversation context:
{context}

User message: {message}

Respond naturally and helpfully. If the user seems interested in medical topics, gently guide them toward asking specific clinical questions. Keep responses concise but warm.

Response:"""
        )

        self.system_info_prompt = ChatPromptTemplate.from_template(
            """You are explaining the capabilities of a cardiology medical assistant system.

The system can help with:
- ESC (European Society of Cardiology) clinical guidelines
- Cardiology protocols and procedures
- Treatment recommendations
- Diagnostic criteria
- Risk assessment guidelines
- Medication protocols
- Clinical decision support

The system uses:
- Latest ESC guidelines and protocols
- Vector database for semantic search
- Self-RAG (Retrieval-Augmented Generation) for accurate responses
- Memory management for conversation context

User question: {message}

Provide a helpful explanation about the system's capabilities. Be specific about what medical topics you can help with.

Response:"""
        )

    def _setup_response_templates(self):
        """Setup template responses for common conversational patterns."""
        
        self.greeting_responses = [
            "Hello! I'm your cardiology assistant, here to help with ESC guidelines and cardiac protocols. What would you like to know?",
            "Hi there! I specialize in European Society of Cardiology protocols. How can I assist you today?",
            "Welcome! I'm here to help with cardiology guidelines and clinical protocols. What can I help you with?",
            "Hello! I can help you with ESC cardiology protocols and treatment guidelines. What's your question?"
        ]
        
        self.farewell_responses = [
            "Goodbye! Feel free to come back anytime with cardiology questions.",
            "Take care! I'm here whenever you need help with cardiac protocols.",
            "Farewell! Don't hesitate to ask if you have any cardiology questions later.",
            "See you later! I'm always here to help with ESC guidelines and protocols."
        ]
        
        self.gratitude_responses = [
            "You're very welcome! I'm glad I could help with your cardiology questions.",
            "Happy to help! Feel free to ask more about cardiac protocols anytime.",
            "My pleasure! I'm here whenever you need guidance on ESC guidelines.",
            "You're welcome! I enjoy helping with cardiology protocol questions."
        ]
        
        self.capability_responses = [
            """I'm a specialized cardiology assistant focused on ESC (European Society of Cardiology) protocols. I can help with:

🏥 **Clinical Guidelines**: ESC recommendations for various cardiac conditions
📋 **Treatment Protocols**: Evidence-based therapeutic approaches
🔬 **Diagnostic Criteria**: Clinical assessment and testing guidelines
💊 **Medication Protocols**: Drug therapy recommendations
⚡ **Emergency Procedures**: Acute cardiac care protocols
📊 **Risk Assessment**: Cardiovascular risk evaluation tools

Try asking me about specific conditions like myocardial infarction, heart failure, or atrial fibrillation!""",

            """I specialize in cardiology protocols from the European Society of Cardiology. Here's what I can help with:

• **Acute Coronary Syndromes**: STEMI, NSTEMI, unstable angina protocols
• **Heart Failure**: Diagnosis, treatment, and management guidelines
• **Arrhythmias**: Atrial fibrillation, VT/VF, bradycardia management
• **Interventional Cardiology**: PCI, CABG indications and procedures
• **Preventive Cardiology**: Risk factor management and screening
• **Emergency Cardiology**: Cardiac arrest, shock, acute presentations

What specific cardiology topic interests you?"""
        ]

    def detect_conversation_type(self, message: str) -> str:
        """
        Detect the type of conversational input to choose appropriate response strategy.
        
        Args:
            message: User input message
            
        Returns:
            str: Type of conversation ("greeting", "farewell", "gratitude", "capability", "general")
        """
        message_lower = message.lower().strip()
        
        # Greeting patterns
        greeting_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(greetings|salutations)\b',
            r'^(hi|hello)$'
        ]
        
        # Farewell patterns
        farewell_patterns = [
            r'\b(goodbye|bye|see you|farewell|good night)\b',
            r'\b(talk to you later|ttyl|catch you later)\b',
            r'^(bye|goodbye)$'
        ]
        
        # Gratitude patterns
        gratitude_patterns = [
            r'\b(thank you|thanks|appreciate|grateful)\b',
            r'\b(much appreciated|many thanks)\b'
        ]
        
        # Capability inquiry patterns
        capability_patterns = [
            r'\b(what can you do|what do you do|your capabilities)\b',
            r'\b(how can you help|what help|what assistance)\b',
            r'\b(what are you|who are you|about you|your purpose)\b',
            r'\b(what topics|what subjects|what areas)\b'
        ]
        
        # Check patterns in order of specificity
        for pattern in greeting_patterns:
            if re.search(pattern, message_lower):
                return "greeting"
        
        for pattern in farewell_patterns:
            if re.search(pattern, message_lower):
                return "farewell"
        
        for pattern in gratitude_patterns:
            if re.search(pattern, message_lower):
                return "gratitude"
        
        for pattern in capability_patterns:
            if re.search(pattern, message_lower):
                return "capability"
        
        return "general"

    def get_template_response(self, conversation_type: str, context: str = "") -> str:
        """
        Get a template response based on conversation type.
        
        Args:
            conversation_type: Type of conversation detected
            context: Additional context from conversation history
            
        Returns:
            str: Appropriate template response
        """
        import random
        
        if conversation_type == "greeting":
            return random.choice(self.greeting_responses)
        elif conversation_type == "farewell":
            return random.choice(self.farewell_responses)
        elif conversation_type == "gratitude":
            return random.choice(self.gratitude_responses)
        elif conversation_type == "capability":
            return random.choice(self.capability_responses)
        else:
            return ""

    def generate_contextual_response(self, message: str, context: str = "") -> str:
        """
        Generate a contextual response using the LLM for more complex conversational needs.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            str: Generated response
        """
        try:
            # Determine if this is a system information request
            system_keywords = ['system', 'how do you work', 'technology', 'database', 'esc guidelines']
            if any(keyword in message.lower() for keyword in system_keywords):
                chain = self.system_info_prompt | self.llm
                response = chain.invoke({"message": message})
            else:
                chain = self.general_conversation_prompt | self.llm
                response = chain.invoke({
                    "message": message,
                    "context": context or "No previous context"
                })
            
            return response.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating contextual response: {e}")
            return "I'm here to help with cardiology protocols and guidelines. What would you like to know?"

    def get_conversation_context(self, state: Dict) -> str:
        """
        Extract relevant conversation context from state.
        
        Args:
            state: Current conversation state
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        # Add conversation summary if available
        summary = state.get("conversation_summary", "")
        if summary:
            context_parts.append(f"Previous conversation: {summary}")
        
        # Add recent messages for context
        messages = state.get("messages", [])
        if len(messages) > 1:
            recent_context = []
            for msg in messages[-3:-1]:  # Last few messages except current
                if isinstance(msg, HumanMessage):
                    recent_context.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    recent_context.append(f"Assistant: {msg.content}")
            
            if recent_context:
                context_parts.append("Recent conversation:\n" + "\n".join(recent_context))
        
        # Add medical context if any medical topics were discussed
        medical_context = state.get("medical_context", [])
        if medical_context:
            context_parts.append(f"Medical topics discussed: {', '.join(medical_context)}")
        
        return "\n\n".join(context_parts) if context_parts else ""

    def conversational_agent_node(self, state: Dict) -> Dict:
        """
        Main LangGraph node function for conversational interactions.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict: Updated state with conversational response
        """
        print("\n💬 Conversational Agent Processing")
        print("=" * 50)
        
        # Extract the current message
        messages = state.get("messages", [])
        if not messages:
            state["response"] = "Hello! I'm here to help with cardiology protocols. How can I assist you?"
            return state
        
        # Get the last human message
        current_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                current_message = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        if not current_message:
            state["response"] = "I'm here to help with cardiology questions. What would you like to know?"
            return state
        
        print(f"📝 Processing conversational query: '{current_message[:50]}...'")
        
        # Get conversation context
        context = self.get_conversation_context(state)
        
        # Detect conversation type
        conversation_type = self.detect_conversation_type(current_message)
        print(f"🎯 Detected conversation type: {conversation_type}")
        
        # Generate response based on type
        if conversation_type in ["greeting", "farewell", "gratitude", "capability"]:
            # Use template response for common patterns
            response = self.get_template_response(conversation_type, context)
            print(f"📋 Using template response for {conversation_type}")
        else:
            # Use LLM for more complex conversational needs
            response = self.generate_contextual_response(current_message, context)
            print(f"🤖 Generated contextual response")
        
        # Update state
        state["response"] = response
        state["current_state"] = "conversational"
        
        # Add metadata
        state["metadata"] = {
            "response_type": "conversational",
            "conversation_type": conversation_type,
            "used_template": conversation_type in ["greeting", "farewell", "gratitude", "capability"]
        }
        
        print(f"✅ Conversational response ready: {response[:100]}...")
        return state

    def handle_edge_cases(self, message: str) -> Optional[str]:
        """
        Handle edge cases and special conversational scenarios.
        
        Args:
            message: User message
            
        Returns:
            Optional[str]: Response if edge case handled, None otherwise
        """
        message_lower = message.lower().strip()
        
        # Handle very short responses
        if len(message.strip()) <= 2:
            if message_lower in ['yes', 'y', 'ok', 'okay']:
                return "Great! What cardiology topic would you like to explore?"
            elif message_lower in ['no', 'n', 'nope']:
                return "No problem! I'm here if you change your mind and want to discuss cardiology protocols."
        
        # Handle confusion or unclear requests
        confusion_indicators = ['what?', 'huh?', 'unclear', 'confuse', "don't understand"]
        if any(indicator in message_lower for indicator in confusion_indicators):
            return "I apologize for any confusion. I'm here to help with ESC cardiology guidelines and protocols. Could you ask me about a specific cardiac condition or procedure?"
        
        # Handle requests for other medical specialties
        other_specialties = ['dermatology', 'psychiatry', 'orthopedic', 'neurology', 'oncology']
        if any(specialty in message_lower for specialty in other_specialties):
            return f"I specialize specifically in cardiology protocols from the ESC. For {message_lower} questions, you'd need a different specialist. However, I'm here to help with any heart-related questions!"
        
        return None

    def get_help_response(self) -> str:
        """
        Generate a comprehensive help response explaining how to use the system.
        
        Returns:
            str: Help response
        """
        return """I'm your cardiology assistant specialized in ESC (European Society of Cardiology) protocols! Here's how I can help:

🔍 **Ask specific questions about:**
• Acute coronary syndromes (STEMI, NSTEMI, unstable angina)
• Heart failure diagnosis and management
• Arrhythmia treatment protocols
• Cardiac procedures and interventions
• Risk assessment and prevention
• Emergency cardiac care

💡 **Example questions:**
• "What is the ESC protocol for acute MI?"
• "How do you manage atrial fibrillation?"
• "Indications for cardiac catheterization?"
• "Heart failure treatment guidelines?"

🎯 **Tips for better responses:**
• Be specific about the condition or procedure
• Mention if you're looking for diagnosis, treatment, or management guidelines
• Ask about specific patient populations if relevant

⚙️ **Available commands:**
• 'help' - Show this help message
• 'status' - Check system status
• 'clear' - Clear conversation history
• '/feedback <positive|negative> [comment]' - Rate my last response
• '/stats' - View feedback statistics

Just ask me anything about cardiology protocols and I'll search through the latest ESC guidelines to help you!"""


def test_conversational_agent():
    """Test function for the conversational agent."""
    print("🧪 Testing Conversational Agent")
    print("=" * 60)
    
    # Initialize agent
    agent = ConversationalAgent()
    
    # Test different conversation types
    test_cases = [
        # Greetings
        {
            "message": "Hello!",
            "expected_type": "greeting",
            "description": "Simple greeting"
        },
        {
            "message": "Good morning, how are you?",
            "expected_type": "greeting", 
            "description": "Formal greeting with question"
        },
        
        # Farewells
        {
            "message": "Goodbye!",
            "expected_type": "farewell",
            "description": "Simple farewell"
        },
        {
            "message": "Thanks for your help, see you later!",
            "expected_type": "farewell",
            "description": "Farewell with gratitude"
        },
        
        # Gratitude
        {
            "message": "Thank you so much!",
            "expected_type": "gratitude",
            "description": "Expression of thanks"
        },
        
        # Capability inquiries
        {
            "message": "What can you help me with?",
            "expected_type": "capability",
            "description": "Capability inquiry"
        },
        {
            "message": "What do you do?",
            "expected_type": "capability",
            "description": "Purpose inquiry"
        },
        
        # General conversation
        {
            "message": "I'm a medical student interested in cardiology",
            "expected_type": "general",
            "description": "General conversational statement"
        },
        
        # Edge cases
        {
            "message": "yes",
            "expected_type": "general",
            "description": "Short response"
        }
    ]
    
    print("Testing conversation type detection:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        message = test_case["message"]
        expected = test_case["expected_type"]
        description = test_case["description"]
        
        detected_type = agent.detect_conversation_type(message)
        
        status = "✅" if detected_type == expected else "❌"
        print(f"{i:2d}. {status} '{message}'")
        print(f"    Expected: {expected}, Detected: {detected_type}")
        print(f"    Description: {description}")
        print()
    
    # Test full node processing
    print("\nTesting full conversational agent node:")
    print("-" * 40)
    
    test_states = [
        {
            "messages": [HumanMessage(content="Hello, I'm new here!")],
            "description": "New user greeting"
        },
        {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(content="Hello! How can I help with cardiology?"),
                HumanMessage(content="What can you help me with?")
            ],
            "description": "Capability inquiry with context"
        },
        {
            "messages": [HumanMessage(content="Thank you for your help!")],
            "description": "Gratitude expression"
        }
    ]
    
    for i, test_state in enumerate(test_states, 1):
        print(f"Test {i}: {test_state['description']}")
        
        # Process through agent
        result_state = agent.conversational_agent_node(test_state.copy())
        
        # Display results
        response = result_state.get("response", "No response")
        metadata = result_state.get("metadata", {})
        
        print(f"Response: {response[:100]}...")
        print(f"Metadata: {metadata}")
        print()
    
    # Test edge cases
    print("Testing edge case handling:")
    print("-" * 40)
    
    edge_cases = [
        "yes",
        "no",
        "what?",
        "I have a dermatology question",
        "help"
    ]
    
    for case in edge_cases:
        edge_response = agent.handle_edge_cases(case)
        print(f"'{case}' -> {edge_response[:60] if edge_response else 'No edge case handling'}...")


if __name__ == "__main__":
    test_conversational_agent()
