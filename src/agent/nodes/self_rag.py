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

import os
import sys
from typing import Dict, List, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Add the data-etl directory to Python path to import vectorstore
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_dir = os.path.join(current_dir, '../../../../data-etl/src')
state_manager_path = os.path.join(current_dir, '../../sqlite')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)
if os.path.exists(state_manager_path):
    sys.path.insert(0, state_manager_path)

from manager import StateManager

try:
    from vectorstore import load_vectorstore, similarity_search
    VECTORSTORE_AVAILABLE = True
except ImportError as e:
    VECTORSTORE_AVAILABLE = False

class SelfRAG:
    """Self-RAG node with proper grading and generation."""
    
    def __init__(self, vectorstore, llm_model: str = "llama3.2:1b", state_manager: StateManager = None, callback_handler = None):
        self.vectorstore = vectorstore
        self.llm_model = llm_model
        self.llm = ChatOllama(model=llm_model, temperature=0.0, verbose=False)
        self.generation_llm = ChatOllama(model=llm_model, temperature=0.7, verbose=False)
        self.state_manager = state_manager
        self.callback_handler = callback_handler
        
        if vectorstore:
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "score_threshold": 0.5}
            )
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        try:
            if not self.retriever:
                return []
            docs = self.retriever.invoke(query)
            return docs
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def grade_relevance(self, document: Document, query: str) -> bool:
        """Grade document relevance to query."""
        system_prompt = """Grade whether this document is relevant to the medical query.
        
        Document: {document}
        Query: {query}
        
        Respond with only: relevant OR not_relevant"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(
            prompt.format_messages(
                document=document.page_content[:500],
                query=query
            ),
            config=config
        )
        
        return "relevant" in result.content.lower()
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer based on query and documents."""
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        context = "\n\n".join([doc.page_content for doc in documents])
        
        system_prompt = """You are a medical expert. Answer the query based on the provided context.
        Be accurate, cite relevant guidelines, and indicate if information is insufficient.
        
        Context: {context}
        
        Query: {query}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.generation_llm.invoke(
            prompt.format_messages(
                context=context,
                query=query
            ),
            config=config
        )
        
        return result.content.strip()
    
    def check_hallucination(self, answer: str, documents: List[Document]) -> bool:
        """Check if answer is grounded in documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        system_prompt = """Check if the answer is supported by the provided context.
        
        Context: {context}
        Answer: {answer}
        
        Respond with only: grounded OR hallucination"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(
            prompt.format_messages(
                context=context,
                answer=answer
            ),
            config=config
        )
        
        return "grounded" in result.content.lower()
    
    def grade_answer(self, answer: str, query: str) -> bool:
        """Grade if answer adequately addresses the query."""
        system_prompt = """Grade if this answer adequately addresses the medical query.
        
        Query: {query}
        Answer: {answer}
        
        Respond with only: adequate OR inadequate"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(
            prompt.format_messages(
                query=query,
                answer=answer
            ),
            config=config
        )
        
        return "adequate" in result.content.lower()
    
    def update_state(self, state: Dict) -> Dict:
        """Main self-RAG node function."""
        query = state.get("query", "")
        max_retrieval_attempts = 2
        max_generation_attempts = 2
        
        retrieval_attempts = 0
        generation_attempts = 0
        final_answer = None
        relevant_docs = []
        
        # Retrieval loop
        while retrieval_attempts < max_retrieval_attempts:
            docs = self.retrieve_documents(query)
            
            if not docs:
                retrieval_attempts += 1
                continue
            
            # Grade document relevance
            for doc in docs:
                if self.grade_relevance(doc, query):
                    relevant_docs.append(doc)
            
            if len(relevant_docs) >= 1:  # Need at least 1 relevant doc
                break
            
            retrieval_attempts += 1
        
        # Generation loop
        while generation_attempts < max_generation_attempts:
            answer = self.generate_answer(query, relevant_docs)
            
            # Check for hallucinations
            if not self.check_hallucination(answer, relevant_docs):
                generation_attempts += 1
                continue
            
            # Grade answer quality
            if self.grade_answer(answer, query):
                final_answer = answer
                break
            
            generation_attempts += 1
        
        # Update state
        state["response"] = final_answer or "I couldn't generate a satisfactory answer. Please rephrase your question."
        state["documents"] = relevant_docs
        state["retrieval_attempts"] = retrieval_attempts + 1
        state["generation_attempts"] = generation_attempts + 1
        state["metadata"] = {
            "relevant_docs_count": len(relevant_docs),
            "total_docs_retrieved": len(docs) if docs else 0
        }
        
        if self.state_manager:
            self.state_manager.save_state(state)
        
        return state