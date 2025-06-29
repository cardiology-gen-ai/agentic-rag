#!/usr/bin/env python3
"""
Self-RAG Node for the Cardiology Protocols Pipeline.
Implements retrieval-augmented generation with self-reflection and grading capabilities.

This node:
1. Retrieves relevant documents from the vector store
2. Grades the relevance of retrieved documents
3. Generates responses based on retrieved context
4. Self-evaluates the quality and groundedness of responses
5. Decides whether to retrieve more documents or regenerate responses
"""

import re
import os
import sys
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from state import State

# Add the data-etl directory to Python path to import vectorstore
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_dir = os.path.join(current_dir, '../../data-etl/src')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)

try:
    from vectorstore import load_vectorstore, similarity_search
    VECTORSTORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Vectorstore module not found: {e}")
    print(f"Looking in: {vectorstore_dir}")
    print("Please ensure the vectorstore directory is well defined.")
    VECTORSTORE_AVAILABLE = False

class SelfRAG:
    def __init__(
            self,
            vectorstore,
            generation_llm_model: str = 'llama3.2:1b',
            generation_temperature: float = 0.7,
            assessment_llm_model: str = 'llama3.1:latest',
            assessment_temperature: float = 0.0,
            max_retrieval_attempts: int = 3,
            max_generation_attempts: int = 3
            ):

        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5
            }
        )
        self.generation_llm_model = generation_llm_model
        self.generation_temperature = generation_temperature
        self.assessment_llm_model = assessment_llm_model
        self.assessment_temperature = assessment_temperature
        self.max_retrieval_attempts = max_retrieval_attempts
        self.max_generation_attempts = max_generation_attempts
        self._setup_llm()
        self._setup_prompts()

    def _setup_llm(self):
        try:
            self.assessment_llm = ChatOllama(
                    model=self.assessment_llm_model,
                    temperature=self.assessment_temperature,
                    verbose=False
                    )
            self.generation_llm = ChatOllama(
                    model=self.generation_llm_model,
                    temperature=self.generation_temperature,
                    verbose=False
                    )
            print(f"✅ Self-RAG LLMs initialized: generation={self.generation_llm_model}, assessment={self.assessment_llm_model}")
        except Exception as e:
            print(f"❌ Error initializing Self-RAG LLMs: {e}")
            raise

    def _setup_prompts(self):
        self.relevance_grader_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question about cardiology protocols.

Retrieved Document:
{document}

User Question: {query}

Does this document contain information that is relevant to answering the user's question?
Provide a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

Score:"""
        )

        self.hallucination_grader_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an answer is grounded in the provided medical documents.

    Documents:
    {documents}

    Answer: {answer}

    Check if the MAIN CLAIMS and FACTS in the answer are supported by the documents.
    - It's OK if the answer uses connecting phrases or contextual language
    - It's OK if the answer mentions the source file names
    - It's OK if the answer provides reasonable medical context
    - Focus on whether the core medical information is accurate and sourced

    What is NOT acceptable:
    - Specific medical facts not mentioned in the documents
    - Invented statistics or numbers
    - Made-up treatment protocols

    Provide a binary score 'yes' or 'no'. 'Yes' means the core content is grounded, 'no' means it contains unsupported medical claims.

    Score:"""
    )

        self.answer_grader_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether an answer adequately addresses a user's question about cardiology.

User Question: {query}
Answer: {answer}

Does the answer adequately address the user's question?
Consider whether it's complete, relevant, and helpful.
Provide a binary score 'yes' or 'no'.

Score:"""
        )

        self.generation_prompt = ChatPromptTemplate.from_template(
            """You are a medical assistant specializing in cardiology protocols from the European Society of Cardiology (ESC).
Answer the question based ONLY on the provided context. If the answer is not in the context, say so.

Context from ESC Guidelines:
{context}

Question: {query}

Instructions:
- Base your answer only on the provided context
- Be specific and cite which guideline the information comes from when possible
- If the context doesn't contain enough information, say so
- Keep the answer clear and clinically relevant

Answer:"""
        )

    def retrieve(self, query: str) -> List:
        """Retrieve relevant documents for the query."""
        try:
            docs = self.retriever.invoke(query)
            print(f"📄 Retrieved {len(docs)} documents for query: '{query[:50]}...'")
            return docs
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []

    def relevance_grader(self, query: str, document) -> bool:
        """Grade if a document is relevant to the query."""
        try:
            relevance_chain = self.relevance_grader_prompt | self.assessment_llm

            # Extract document content
            doc_content = document.page_content if hasattr(document, 'page_content') else str(document)

            result = relevance_chain.invoke({
                "query": query,
                "document": doc_content
            })

            response = result.content.strip().lower()
            is_relevant = "yes" in response

            print(f"  📊 Relevance grade: {'✓ Relevant' if is_relevant else '✗ Not relevant'}")
            return is_relevant

        except Exception as e:
            print(f"❌ Error grading relevance: {e}")
            return False

    def generate(self, query: str, relevant_docs: List) -> str:
        """Generate answer based on query and relevant documents."""
        try:
            # Combine document contents
            context = "\n\n---\n\n".join([
                f"From {doc.metadata.get('source_file', 'Unknown source')}:\n{doc.page_content}"
                for doc in relevant_docs
            ])

            generation_chain = self.generation_prompt | self.generation_llm

            result = generation_chain.invoke({
                "context": context,
                "query": query
            })

            response = result.content.strip()
            print(f"💬 Generated answer: {response[:100]}...")
            return response

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I encountered an error while generating the answer."

    def hallucination_grader(self, answer: str, documents: List) -> bool:
        """Check if the answer is grounded in the provided documents."""
        try:
            # Combine document contents
            docs_content = "\n\n---\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ])

            hallucination_chain = self.hallucination_grader_prompt | self.assessment_llm

            result = hallucination_chain.invoke({
                "documents": docs_content,
                "answer": answer
            })

            response = result.content.strip().lower()
            is_grounded = "yes" in response

            print(f"  🔍 Hallucination check: {'✓ Grounded' if is_grounded else '✗ Contains hallucinations'}")
            return is_grounded

        except Exception as e:
            print(f"❌ Error checking hallucinations: {e}")
            return False

    def answer_grader(self, query: str, answer: str) -> bool:
        """Grade if the answer adequately addresses the query."""
        try:
            answer_chain = self.answer_grader_prompt | self.assessment_llm

            result = answer_chain.invoke({
                "query": query,
                "answer": answer
            })

            response = result.content.strip().lower()
            is_adequate = "yes" in response

            print(f"  ✅ Answer grade: {'✓ Adequate' if is_adequate else '✗ Inadequate'}")
            return is_adequate

        except Exception as e:
            print(f"❌ Error grading answer: {e}")
            return False

    def reformulate_query(self, original_query: str, attempt: int) -> str:
        """Reformulate query for better retrieval."""
        reformulations = [
            f"{original_query} ESC guidelines protocol",
            f"European Society of Cardiology {original_query}",
            f"What are the clinical recommendations for {original_query}",
            f"cardiology protocol {original_query} management"
        ]

        if attempt < len(reformulations):
            return reformulations[attempt]
        return original_query

    def selfRAG_node(self, state: State) -> State:
        """Main self-RAG node for LangGraph integration."""
        print("\n🔄 Self-RAG Node Processing")
        print("=" * 50)

        # Extract query from messages
        messages = state.get("messages", [])
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                query = msg.content if hasattr(msg, 'content') else str(msg)
                break

        if not query:
            state["response"] = "No query found in the conversation."
            return state

        print(f"📝 Processing query: '{query}'")

        # Initialize tracking variables
        retrieval_attempt = 0
        generation_attempt = 0
        final_answer = None
        relevant_docs = []

        # Retrieval loop with self-assessment
        while retrieval_attempt < self.max_retrieval_attempts:
            print(f"\n🔍 Retrieval attempt {retrieval_attempt + 1}/{self.max_retrieval_attempts}")

            # Reformulate query if not first attempt
            if retrieval_attempt > 0:
                query_to_use = self.reformulate_query(query, retrieval_attempt)
                print(f"  🔄 Reformulated query: '{query_to_use}'")
            else:
                query_to_use = query

            # Retrieve documents
            retrieved_docs = self.retrieve(query_to_use)

            if not retrieved_docs:
                print("  ⚠️  No documents retrieved")
                retrieval_attempt += 1
                continue

            # Grade relevance of each document
            relevant_docs = []  # Reset for each attempt
            for i, doc in enumerate(retrieved_docs):
                print(f"\n  Grading document {i+1}/{len(retrieved_docs)}")
                if self.relevance_grader(query, doc):
                    relevant_docs.append(doc)

            print(f"\n  📊 Relevant documents: {len(relevant_docs)}/{len(retrieved_docs)}")

            # If we have relevant documents, proceed to generation
            if len(relevant_docs) >= 2:  # Need at least 2 relevant docs
                break

            retrieval_attempt += 1

        # Check if we have relevant documents
        if not relevant_docs:
            state["response"] = "I couldn't find relevant information in the cardiology protocols to answer your question."
            state["documents"] = []
            state["retrieval_attempts"] = retrieval_attempt
            state["generation_attempts"] = 0
            return state

        # Generation loop with self-assessment
        while generation_attempt < self.max_generation_attempts:
            print(f"\n🎯 Generation attempt {generation_attempt + 1}/{self.max_generation_attempts}")

            # Generate answer
            answer = self.generate(query, relevant_docs)

            # Check for hallucinations
            if not self.hallucination_grader(answer, relevant_docs):
                print("  ⚠️  Hallucination detected, regenerating...")
                generation_attempt += 1
                # Adjust temperature for next attempt
                self.generation_llm.temperature = max(0.1, self.generation_temperature - 0.2 * generation_attempt)
                continue

            # Grade answer quality
            if self.answer_grader(query, answer):
                final_answer = answer
                break
            else:
                print("  ⚠️  Answer inadequate, regenerating...")
                generation_attempt += 1
                continue

        # Reset temperature
        self.generation_llm.temperature = self.generation_temperature

        # Prepare final response
        if final_answer:
            state["response"] = final_answer
            state["documents"] = relevant_docs
            state["retrieval_attempts"] = retrieval_attempt + 1
            state["generation_attempts"] = generation_attempt + 1
            state["metadata"] = {
                "retrieval_attempts": retrieval_attempt + 1,
                "generation_attempts": generation_attempt + 1,
                "relevant_docs_count": len(relevant_docs)
            }
            print(f"\n✅ Self-RAG completed successfully")
        else:
            state["response"] = "I found relevant information but couldn't generate a satisfactory answer. Please try rephrasing your question."
            state["documents"] = relevant_docs
            state["retrieval_attempts"] = retrieval_attempt
            state["generation_attempts"] = generation_attempt

        return state


def test_selfRAG():
    """Test the Self-RAG implementation."""
    print("🧪 Testing Self-RAG Node")
    print("=" * 60)

    if not VECTORSTORE_AVAILABLE:
        print("❌ Cannot test: Vectorstore module not available")
        return

    # Load vectorstore
    try:
        print("Loading vectorstore...")
        vectorstore = load_vectorstore(
            collection_name="cardio_protocols",
            vectorstore_type="qdrant"
        )
        print("✅ Vectorstore loaded")
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return

    # Initialize Self-RAG
    self_rag = SelfRAG(vectorstore)

    # Test queries
    test_queries = [
        "What is the ESC protocol for acute myocardial infarction?",
        "How do you manage atrial fibrillation according to guidelines?",
        "What are the indications for cardiac catheterization?",
        "Anticoagulation therapy in heart failure patients"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print(f"{'='*60}")

        # Create test state
        test_state: State = {
            "messages": [HumanMessage(content=query)],
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

        # Process through Self-RAG
        result_state = self_rag.selfRAG_node(test_state)

        # Display results
        print(f"\n📋 Results:")
        response = result_state.get('response', 'No response')
        print(f"Response: {response[:200]}...")
        if result_state.get('metadata'):
            print(f"Metadata: {result_state['metadata']}")
        print(f"Documents retrieved: {len(result_state.get('documents', []))}")
        print(f"Retrieval attempts: {result_state.get('retrieval_attempts', 'N/A')}")
        print(f"Generation attempts: {result_state.get('generation_attempts', 'N/A')}")


if __name__ == "__main__":
    test_selfRAG()
