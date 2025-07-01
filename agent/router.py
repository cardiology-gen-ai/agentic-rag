#!/usr/bin/env python3
"""
Router node for the RAG pipeline.
Enhanced classification for medical queries vs conversational queries.
"""

import re
from typing import Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


class Router:
    """
    Enhanced router node for better medical query classification.
    """
    def __init__(
            self,
            llm_model: str = 'llama3.1:latest',  # Use larger model
            temperature: float = 0.0  # Even lower for classification
            ):

        # Enhanced medical keywords with better coverage
        self.medical_keywords = [
            # Patient-related terms
            'patient', 'patient with', 'my patient', 'patients',
            
            # Cardiovascular anatomy and conditions
            'heart', 'cardiac', 'cardio', 'cardiovascular', 'cardiovasculare',  # Include typo
            'myocardium', 'myocardial', 'pericardium', 'endocardium', 'epicardium',
            'atrium', 'atrial', 'ventricle', 'ventricular', 'aorta', 'aortic',
            'mitral', 'tricuspid', 'pulmonary valve', 'coronary', 'artery', 'arterial',
            'vein', 'venous', 'vessel', 'vascular', 'circulation',

            # Cardiovascular conditions  
            'myocardial infarction', 'heart attack', 'acute coronary syndrome', 'acs',
            'stemi', 'nstemi', 'unstable angina', 'stable angina', 'heart failure',
            'hf', 'cardiomyopathy', 'dilated cardiomyopathy', 'hypertrophic cardiomyopathy',
            'arrhythmia', 'atrial fibrillation', 'afib', 'ventricular tachycardia',
            'ventricular fibrillation', 'bradycardia', 'tachycardia', 'cardiac arrest',
            'sudden cardiac death', 'heart block', 'pericarditis', 'endocarditis',
            'myocarditis', 'aortic stenosis', 'aortic regurgitation', 'mitral stenosis',
            'mitral regurgitation', 'tricuspid regurgitation', 'pulmonary embolism',
            'deep vein thrombosis', 'hypertension', 'hypotension', 'shock', 'cardiogenic shock',

            # Clinical terminology
            'protocol', 'management', 'treatment', 'therapy', 'diagnosis', 'diagnostic',
            'guideline', 'recommendation', 'indication', 'contraindication',
            'medication', 'drug', 'intervention', 'procedure',
            
            # Procedures and interventions
            'percutaneous coronary intervention', 'pci', 'angioplasty', 'stent', 'stenting',
            'coronary artery bypass', 'cabg', 'cardiac catheterization', 'angiography',
            'echocardiography', 'echo', 'electrocardiogram', 'ecg', 'ekg',
            'stress test', 'nuclear stress test', 'cardiac mri', 'cardiac ct',
            'ablation', 'cardioversion', 'defibrillation', 'pacemaker', 'icd',
            'valve replacement', 'valve repair',

            # Medications
            'anticoagulant', 'anticoagulation', 'antiplatelet', 'aspirin', 'clopidogrel',
            'warfarin', 'heparin', 'beta blocker', 'ace inhibitor', 'statin',
            'diuretic', 'calcium channel blocker', 'nitrate', 'nitroglycerin',
            'digoxin', 'amiodarone',

            # ESC specific
            'esc', 'european society of cardiology', 'guidelines', 'recommendations',
            'class i', 'class ii', 'class iii', 'level of evidence'
       ]

        # Enhanced conversational patterns
        self.conversational_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(thank you|thanks|appreciate)\b',
            r'\b(goodbye|bye|see you|farewell)\b',
            r'\b(how are you|what\'s up|how\'s it going)\b',
            r'\b(yes|no|ok|okay|sure)\b$',
            r'^(who|what|where|when|why|how) (are|is|can|do|did) you\b',
            r'\b(help me|can you help|what can you do)\b$',
            r'\b(what are you|who are you|tell me about yourself)\b'
        ]

        self.llm_model = llm_model
        self.temperature = temperature
        self._setup_llm()
        self._setup_prompts()

    def _setup_llm(self):
        """Setup the LLM for classification."""
        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=self.temperature,
                verbose=False
            )
            print(f"✅ Improved Router LLM initialized: {self.llm_model}")
        except Exception as e:
            print(f"❌ Error initializing router LLM: {e}")
            raise

    def _setup_prompts(self):
        """Enhanced prompt for better medical classification."""
        self.classification_prompt = ChatPromptTemplate.from_template(
            """
            You are a medical query classifier for a cardiology information system.

            Your task is to classify each query into exactly one category:

            **conversational**:
            - Social greetings, farewells, pleasantries ("Hello", "Thank you", "Goodbye")
            - Questions about the system itself ("What can you do?", "How do you work?")
            - General chitchat or acknowledgments ("Yes", "OK", "I understand")
            - Questions about who/what you are

            **document_based**:
            - ANY query involving patients, medical conditions, or clinical scenarios
            - Questions about medical protocols, guidelines, or treatments
            - Requests for ESC cardiology protocols or recommendations
            - Clinical decision-making questions
            - Questions about medications, procedures, or diagnostics
            - Medical terminology explanations
            - Risk assessment or management questions

            IMPORTANT RULES:
            1. If the query mentions "patient" or "my patient" → ALWAYS document_based
            2. If the query mentions any cardiovascular condition → ALWAYS document_based  
            3. If the query asks about treatment, management, or protocols → ALWAYS document_based
            4. If the query contains medical terminology → ALWAYS document_based
            5. When in doubt between the two, choose document_based for medical systems

            Examples:
            - "Hello, how are you?" → conversational
            - "I have a patient with cardiovascular disease, what can I do?" → document_based
            - "What is the protocol for acute MI?" → document_based
            - "Thank you for your help!" → conversational
            - "Patient with heart failure management?" → document_based
            - "What can you help me with?" → conversational
            - "ESC guidelines for atrial fibrillation?" → document_based

            Query to classify: "{query}"

            Respond with ONLY one word: "conversational" or "document_based"
            """
        )

    def classify_query(self, query: str) -> str:
        """
        Enhanced classification with multiple validation steps.
        """
        if not query or not query.strip():
            return "conversational"

        query = query.strip()

        # First, try rule-based classification for clear cases
        rule_result = self._enhanced_rule_based_classify(query)
        
        # If rule-based gives us a strong signal, use it
        if rule_result == "document_based_strong":
            print(f"🎯 Rule-based classified '{query[:50]}...' as: document_based (strong signal)")
            return "document_based"
        
        # Try LLM classification
        try:
            llm_result = self._llm_classify(query)
            if llm_result in ["conversational", "document_based"]:
                # Validate LLM result with rules
                if rule_result == "conversational_strong" and llm_result == "document_based":
                    print(f"⚠️  LLM/Rule conflict - using rule-based: conversational")
                    return "conversational"
                
                print(f"🎯 LLM classified '{query[:50]}...' as: {llm_result}")
                return llm_result
        except Exception as e:
            print(f"⚠️  LLM classification failed: {e}")

        # Fallback to enhanced rule-based classification
        final_result = "document_based" if rule_result == "document_based_strong" else \
                      "conversational" if rule_result == "conversational_strong" else \
                      self._fallback_classify(query)
        
        print(f"🎯 Fallback classified '{query[:50]}...' as: {final_result}")
        return final_result

    def _enhanced_rule_based_classify(self, query: str) -> str:
        """
        Enhanced rule-based classification with confidence levels.
        Returns: "document_based_strong", "conversational_strong", or "uncertain"
        """
        query_lower = query.lower()

        # Strong document-based signals
        strong_medical_indicators = [
            r'\bpatient\b',
            r'\bmy patient\b', 
            r'\bpatients\b',
            r'\bcardiovascular disease\b',
            r'\bheart failure\b',
            r'\bmyocardial infarction\b',
            r'\batrial fibrillation\b',
            r'\besc protocol\b',
            r'\besc guideline\b',
            r'\bmanagement\b.*\b(of|for)\b',
            r'\btreatment\b.*\b(of|for)\b',
            r'\bwhat.*do.*patient\b',
            r'\bhow.*manage\b',
            r'\bprotocol.*for\b'
        ]

        for pattern in strong_medical_indicators:
            if re.search(pattern, query_lower):
                return "document_based_strong"

        # Check for medical keywords
        medical_keyword_count = sum(1 for keyword in self.medical_keywords if keyword in query_lower)
        if medical_keyword_count >= 2:  # Multiple medical terms
            return "document_based_strong"
        elif medical_keyword_count >= 1:
            return "document_based_strong"  # Even one medical term in context

        # Strong conversational signals
        for pattern in self.conversational_patterns:
            if re.search(pattern, query_lower):
                return "conversational_strong"

        # Very short queries are usually conversational
        if len(query.split()) <= 2 and not any(keyword in query_lower for keyword in self.medical_keywords):
            return "conversational_strong"

        return "uncertain"

    def _llm_classify(self, query: str) -> str:
        """LLM classification with enhanced prompt."""
        classification_chain = self.classification_prompt | self.llm
        result = classification_chain.invoke({"query": query})
        
        response = result.content.strip().lower()
        
        if "document_based" in response or "document-based" in response:
            return "document_based"
        elif "conversational" in response:
            return "conversational"
        else:
            return "uncertain"

    def _fallback_classify(self, query: str) -> str:
        """Final fallback classification."""
        query_lower = query.lower()
        
        # If it contains any medical terminology, assume document_based
        if any(keyword in query_lower for keyword in self.medical_keywords):
            return "document_based"
        
        # If it's a question and longer than 3 words, assume document_based
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
        if any(word in query_lower for word in question_words) and len(query.split()) > 3:
            return "document_based"
        
        # Default to conversational for very unclear cases
        return "conversational"

    def router_node(self, state: Dict) -> Dict:
        """LangGraph node function for query routing."""
        messages = state.get("messages", [])
        if not messages:
            state["query_type"] = "conversational"
            return state

        # Get the last human message
        last_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                last_message = msg.content if hasattr(msg, 'content') else str(msg)
                break

        if not last_message:
            state["query_type"] = "conversational"
            return state

        query_type = self.classify_query(last_message)
        state["query_type"] = query_type

        return state

    def route_query(self, state: Dict) -> str:
        """Conditional edge function for LangGraph routing."""
        return state.get("query_type", "document_based")


def test_improved_router():
    """Test the improved router with various medical scenarios."""
    print("🧪 Testing Improved Router")
    print("=" * 60)

    router = Router()

    test_cases = [
        # Clear document-based cases
        ("I have a patient with cardiovascular disease, what can I do?", "document_based"),
        ("My patient has cardiovasculare disease", "document_based"),  # Typo test
        ("Patient with heart failure management", "document_based"),
        ("What is the ESC protocol for acute MI?", "document_based"),
        ("How do you manage atrial fibrillation?", "document_based"),
        ("Treatment for cardiac arrest", "document_based"),
        ("Anticoagulation therapy protocols", "document_based"),
        
        # Clear conversational cases  
        ("Hello!", "conversational"),
        ("Thank you for your help", "conversational"),
        ("What can you help me with?", "conversational"),
        ("Who are you?", "conversational"),
        ("Goodbye", "conversational"),
        ("Yes", "conversational"),
        
        # Edge cases
        ("Heart?", "document_based"),  # Short medical term
        ("Patient?", "document_based"),  # Short but medical context
        ("Protocol", "document_based"),   # Medical context
        ("Help", "conversational"),      # General help
    ]

    print("Classification Results:")
    print("-" * 40)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        actual = router.classify_query(query)
        status = "✅" if actual == expected else "❌"
        
        print(f"{status} '{query}' → {actual} (expected: {expected})")
        
        if actual == expected:
            correct += 1
    
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Test the specific failing case
    print(f"\n🔬 Detailed test of the failing case:")
    failing_query = "I have a patient with cardiovascular disease, what can I do?"
    
    print(f"Query: '{failing_query}'")
    result = router.classify_query(failing_query)
    print(f"Final classification: {result}")


if __name__ == "__main__":
    test_improved_router()
