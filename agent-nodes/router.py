#!/usr/bin/env python3
"""
Router node for the RAG pipeline.
It classifies if a user query necessitates the retrieval of cardiology guidelines or not.

First idea:
    1. Use llama model to classify the query.
    2. Use an algorithm that finds in the guidelines which are the "medical keywords"
       and classifies the sentences that contain them.
"""

import re
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from state import State


class Router:
    """
    Router node for the RAG pipeline.
    It classifies if a user query necessitates the retrieval of cardiology guidelines or not.
    """
    def __init__(
            self,
            llm_model: str = 'llama3.2:1b',
            temperature: float = 0.1  # this controls the creativity of the llm responses
            ):

        self.medical_keywords = [
            # Cardiovascular anatomy
            'heart', 'cardiac', 'cardio', 'myocardium', 'myocardial', 'pericardium',
            'endocardium', 'epicardium', 'atrium', 'atrial', 'ventricle', 'ventricular',
            'aorta', 'aortic', 'mitral', 'tricuspid', 'pulmonary valve', 'coronary',
            'artery', 'arterial', 'vein', 'venous', 'vessel', 'vascular', 'circulation',

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

            # Procedures and interventions
            'percutaneous coronary intervention', 'pci', 'angioplasty', 'stent', 'stenting',
            'coronary artery bypass', 'cabg', 'cardiac catheterization', 'angiography',
            'echocardiography', 'echo', 'transesophageal echo', 'tee', 'stress test',
            'electrocardiogram', 'ecg', 'ekg', 'holter monitor', 'event monitor',
            'cardiac mri', 'cardiac ct', 'nuclear stress test', 'thallium scan',
            'ablation', 'cardioversion', 'defibrillation', 'pacemaker', 'icd',
            'cardiac resynchronization therapy', 'crt', 'valve replacement',
            'valve repair', 'balloon valvuloplasty', 'atherectomy', 'thrombectomy',

            # Medications
            'anticoagulant', 'anticoagulation', 'antiplatelet', 'aspirin', 'clopidogrel',
            'warfarin', 'heparin', 'enoxaparin', 'dabigatran', 'rivaroxaban', 'apixaban',
            'beta blocker', 'metoprolol', 'atenolol', 'carvedilol', 'ace inhibitor',
            'lisinopril', 'enalapril', 'arb', 'losartan', 'valsartan', 'statin',
            'atorvastatin', 'simvastatin', 'rosuvastatin', 'diuretic', 'furosemide',
            'hydrochlorothiazide', 'calcium channel blocker', 'amlodipine', 'diltiazem',
            'nitrate', 'nitroglycerin', 'isosorbide', 'digoxin', 'amiodarone',

            # Risk factors and comorbidities
            'diabetes', 'diabetic', 'hyperlipidemia', 'dyslipidemia', 'cholesterol',
            'smoking', 'obesity', 'metabolic syndrome', 'chronic kidney disease',
            'peripheral artery disease', 'pad', 'cerebrovascular disease', 'stroke',

            # Clinical parameters
            'ejection fraction', 'left ventricular function', 'wall motion',
            'cardiac output', 'stroke volume', 'preload', 'afterload', 'contractility',
            'hemodynamics', 'blood pressure', 'heart rate', 'pulse', 'murmur',
            'gallop', 'rub', 'chest pain', 'dyspnea', 'orthopnea', 'edema',
            'syncope', 'palpitations', 'fatigue', 'exercise intolerance',

            # Diagnostic tests and values
            'troponin', 'ck', 'ck-mb', 'bnp', 'nt-probnp', 'lipid panel',
            'ldl', 'hdl', 'triglycerides', 'hemoglobin a1c', 'creatinine',
            'gfr', 'inr', 'pt', 'ptt', 'platelet count',

            # ESC specific
            'esc', 'european society of cardiology', 'guidelines', 'recommendations',
            'class i', 'class ii', 'class iii', 'level of evidence', 'grade a',
            'grade b', 'grade c'
       ]

        self.conversational_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(thank you|thanks|appreciate)\b',
            r'\b(goodbye|bye|see you|farewell)\b',
            r'\b(how are you|what\'s up|how\'s it going)\b',
            r'\b(yes|no|ok|okay|sure)\b$',
            r'^(who|what|where|when|why|how) (are|is|can|do|did) you',
            r'\b(help me|can you help|what can you do)\b'
        ]

        self.llm_model = llm_model
        self.temperature = temperature
        self._setup_llm()
        self._setup_prompts()

    def _setup_llm(self):
        """
        Function that setups the basic information for the llm.
        """
        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=self.temperature,
                verbose=False
            )
            print(f"✅ Router LLM initialized: {self.llm_model}")
        except Exception as e:
            print(f"❌ Error initializing router LLM: {e}")
            raise

    def _setup_prompts(self):
        """Function that tells the llm the global way to operate."""
        self.classification_prompt = ChatPromptTemplate.from_template(
            """
            You are a query classifier for a medical information system specializing in cardiology protocols from the European Society of Cardiology (ESC).

            Your task is to classify each query into exactly one of these categories:

            **conversational**:
            - Greetings, farewells, pleasantries
            - General questions about the system itself
            - Thanks, acknowledgments
            - Yes/no responses
            - Casual conversation

            **document_based**:
            - Medical questions requiring specific protocol information
            - Clinical guideline inquiries
            - Questions about procedures, treatments, diagnoses
            - Requests for ESC protocol details
            - Medical terminology explanations
            - Any query needing factual medical knowledge

            Examples:
            - "Hello, how are you?" → conversational
            - "What is the protocol for acute MI?" → document_based
            - "Thanks for your help!" → conversational
            - "ESC guidelines for heart failure?" → document_based
            - "What can you help me with?" → conversational
            - "Indications for cardiac catheterization?" → document_based
            - "Who created this system?" → conversational
            - "Risk factors for coronary artery disease?" → document_based

            Query to classify: "{query}"

            Respond with ONLY one word: "conversational" or "document_based"
            """
        )

    def classify_query(self, query: str) -> str:
        """
        Classify a query as conversational or document-based

        Args:
            query: User query string

        Returns:
            str: "conversational" or "document_based"
        """
        if not query or not query.strip():  # query.strip() removes whitespaces
            return "conversational"

        query = query.strip()

        # Trying LLM classification
        try:
            llm_result = self._llm_classify(query)
            if llm_result in ["conversational", "document_based"]:
                print(f"🎯 LLM classified '{query[:50]}...' as: {llm_result}")
                return llm_result
        except Exception as e:
            print(f"⚠️  LLM classification failed: {e}")

        # Else try deterministic classification
        rule_based_result = self._rule_based_classify(query)
        print(f"🎯 Rule-based classified '{query[:50]}...' as: {rule_based_result}")
        return rule_based_result

    def _llm_classify(self, query: str) -> str:
        """
        Uses an llm (llama in this case) for query classification
        """
        classification_chain = self.classification_prompt | self.llm  # the pipe (|) operator just connects LangChain components together in a pipeline

        result = classification_chain.invoke({"query": query})

        # Extract classification from response
        response = result.content.strip().lower()

        if "document_based" in response or "document-based" in response:
            return "document_based"
        elif "conversational" in response:
            return "conversational"
        else:
            # If response is unclear, default based on content
            return self._rule_based_classify(query)

    def _rule_based_classify(self, query: str) -> str:
        """
        Uses rules to classify the queries.
        """
        query_lower = query.lower()

        # Check for conversational patterns first
        for pattern in self.conversational_patterns:
            if re.search(pattern, query_lower):
                return "conversational"

        # Check for medical keywords
        for keyword in self.medical_keywords:
            if keyword in query_lower:
                return "document_based"

        # Check for question words that usually indicate information seeking
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
        has_question_word = any(word in query_lower for word in question_words)

        # If it's a question without obvious conversational markers, assume document-based
        if has_question_word and len(query.split()) > 3:
            return "document_based"

        # Very short queries are usually conversational
        if len(query.split()) <= 2:
            return "conversational"

        # Default to document_based for medical system
        return "document_based"

    def router_node(self, state: State) -> State:
        """
        LangGraph node function for query routing

        Args:
            state: Agent state containing messages

        Returns:
            Updated state with query_type classification
        """
        messages = state.get("messages", [])
        if not messages:
            state["query_type"] = "conversational"
            return state

        # Obtain the last human message
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

    def route_query(self, state: State) -> str:
        """
        Conditional edge function for LangGraph routing

        Args:
            state: Agent state

        Returns:
            str: Route name ("conversational" or "document_based")
        """
        return state.get("query_type", "document_based")

    def batch_classify(self, queries: List[str]) -> List[dict]:
        """
        Classify multiple queries at once

        Args:
            queries: List of query strings

        Returns:
            List of dictionaries with query and classification
        """
        results = []
        for query in queries:
            classification = self.classify_query(query)
            results.append({
                "query": query,
                "classification": classification
            })
        return results

    def get_classification_confidence(self, query: str) -> dict:
        """
        Get confidence scores for both classifications

        Args:
            query: Query string

        Returns:
            Dict with confidence scores for each classification
        """
        # Simple heuristic-based confidence scoring
        query_lower = query.lower()

        conversational_score = 0.5
        document_score = 0.5

        # Boost conversational score for patterns
        for pattern in self.conversational_patterns:
            if re.search(pattern, query_lower):
                conversational_score += 0.3
                break

        # Boost document score for medical keywords
        medical_matches = sum(1 for keyword in self.medical_keywords if keyword in query_lower)
        document_score += medical_matches * 0.2

        # Normalize scores
        total = conversational_score + document_score
        conversational_score /= total
        document_score /= total

        return {
            "conversational": round(conversational_score, 3),
            "document_based": round(document_score, 3)
        }


def test_router():
    """Test function for the query router"""
    print("🧪 Testing Query Router")
    print("=" * 50)

    # Initialize router
    router = Router()

    # Test queries
    test_queries = [
        # Conversational
        "Hello!",
        "How are you today?",
        "Thank you for your help",
        "What can you help me with?",
        "Who created you?",
        "Goodbye!",

        # Document-based
        "What is the protocol for acute myocardial infarction?",
        "ESC guidelines for heart failure management",
        "Indications for cardiac catheterization",
        "How do you treat atrial fibrillation?",
        "Risk factors for coronary artery disease",
        "Anticoagulation therapy protocols",

        # Edge cases
        "MI",
        "Protocol?",
        "Help with cardiac arrest",
        "Yes",
        "No thanks"
    ]

    # Test classification
    results = router.batch_classify(test_queries)

    print("Classification Results:")
    print("-" * 30)
    for result in results:
        confidence = router.get_classification_confidence(result["query"])
        print(f"Query: '{result['query']}'")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {confidence}")
        print()

    # Test state-based routing (simulate LangGraph usage)
    print("\nTesting LangGraph Integration:")
    print("-" * 30)

    test_state: State = {
        "messages": [HumanMessage(content="What is the ESC protocol for acute MI?")],
        "query_type": None,
        "response": None,
        "context": None,
        "conversation_summary": None,
        "documents": None,
        "metadata": None,
        "retrieval_attempts": None,
        "generation_attempts": None,
        "current_state": None,
        "next_action": None
    }

    updated_state = router.router_node(test_state)
    route = router.route_query(updated_state)

    print(f"Test state: {test_state['messages'][0].content}")
    print(f"Classification: {updated_state['query_type']}")
    print(f"Route: {route}")


if __name__ == "__main__":
    test_router()
