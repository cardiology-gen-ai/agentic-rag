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
from typing import Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document

# Add the data-etl directory to Python path to import vectorstore
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(current_dir, '../../../../data-etl/src')
configs_path = os.path.join(current_dir, '../../')
state_manager_path = os.path.join(current_dir, '../../sqlite')

sys.path.extend([current_dir, vectorstore_path, configs_path, state_manager_path])

from manager import StateManager
import configs

class SelfRAG:
    """Self-RAG node with proper grading and generation."""
    
    def __init__(self, vectorstore, llm_model: str = configs.LLM_MODEL, state_manager: StateManager = None, callback_handler = None):
        self.vectorstore = vectorstore
        self.llm_model = llm_model
        self.llm = ChatOllama(model=llm_model, temperature=configs.LLM_TEMPERATURE, verbose=False)
        self.generation_llm = ChatOllama(model=llm_model, temperature=configs.LLM_GENERATION_TEMPERATURE, verbose=False)
        self.state_manager = state_manager
        self.callback_handler = callback_handler
        
        if vectorstore:
            # Remove score_threshold to get more results
            self.retriever = vectorstore.as_retriever(
                search_type=configs.SEARCH_TYPE,
                search_kwargs={"k": configs.RETRIEVAL_K}
            )
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        try:
            if not self.retriever:
                return []
            docs = self.retriever.invoke(query)
            if configs.DEBUG:
                print(f"\n -> Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def grade_relevance(self, document: Document, query: str) -> bool:
        """Grade document relevance to query - more lenient grading."""
        prompt_text = f"""Is this document relevant to answer the question?

Question: {query}

Document excerpt: {document.page_content[:configs.MAX_CONTEXT_LENGTH]}

Answer with just one word: relevant or not_relevant"""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(prompt_text, config=config)
        response = result.content.strip().lower()
        
        # More lenient matching
        is_relevant = "relevant" in response and "not" not in response
        if configs.DEBUG:
            print(f"\nDocument relevant: {is_relevant}")
        return is_relevant
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer based on query and documents."""
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Take more content from each document
        context = "\n\n---\n\n".join([doc.page_content[:configs.MAX_CONTEXT_LENGTH] for doc in documents[:configs.MAX_DOCS_TO_USE]])
        
        prompt_text = f"""You are a medical expert assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}

Provide a clear, helpful answer based on the context. If the context doesn't fully answer the question, say what you can answer and what information is missing."""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.generation_llm.invoke(prompt_text, config=config)
        
        return result.content.strip()
    
    def check_hallucination(self, answer: str, documents: List[Document]) -> bool:
        """Check if answer is grounded in documents."""
        if not documents:
            return False
        
        # Combine all document content
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Truncate context if too long
        if len(context) > configs.MAX_CONTEXT_LENGTH:
            context = context[:configs.MAX_CONTEXT_LENGTH]
        
        prompt_text = f"""Check if the answer is supported by the provided context.

Context: {context}

Answer: {answer}

Respond with only: grounded OR hallucination"""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(prompt_text, config=config)
        
        return "grounded" in result.content.lower()
    
    def grade_answer(self, answer: str, query: str) -> bool:
        """Grade if answer adequately addresses the query."""
        prompt_text = f"""Grade if this answer adequately addresses the medical query.

Query: {query}
Answer: {answer}

Respond with only: adequate OR inadequate"""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(prompt_text, config=config)
        
        return "adequate" in result.content.lower()
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite the query when the retrieve grade is negative."""
        prompt_text = f"""You are a medical query optimizer. Rewrite the following query to improve retrieval of relevant medical documents.

Original query: {query}

Make the query more specific, add relevant medical terminology, and focus on key clinical concepts.
Respond with only the rewritten query."""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(prompt_text, config=config)
        
        return result.content.strip()

    def re_generate(self, answer: str) -> str:
        """Re-generate the llm response when hallucinations happen."""
        prompt_text = f"""The previous answer contained information not supported by the provided context. 
Generate a more conservative and grounded response that only uses information directly supported by the medical documents.

Previous answer: {answer}

Provide a revised answer that is more cautious and clearly indicates when information is not available."""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.generation_llm.invoke(prompt_text, config=config)
        
        return result.content.strip()
    
    def update_state(self, state: Dict) -> Dict:
        """Main self-RAG node function with query rewriting and answer regeneration."""
        query = state.get("query", "")
        original_query = query
        max_retrieval_attempts = configs.MAX_RETRIEVAL_ATTEMPTS
        max_generation_attempts = configs.MAX_GENERATION_ATTEMPTS
        
        retrieval_attempts = 0
        generation_attempts = 0
        final_answer = None
        relevant_docs = []
        all_docs = []
        
        # Retrieval loop with query rewriting
        while retrieval_attempts < max_retrieval_attempts:
            if configs.DEBUG:
                print(f"\nRetrieval attempt {retrieval_attempts + 1}")
                if retrieval_attempts > 0:
                    print(f"Using rewritten query: {query}")
            
            docs = self.retrieve_documents(query)
            all_docs = docs
            
            if not docs:
                retrieval_attempts += 1
                print(f"\nNo documents retrieved on attempt {retrieval_attempts}")
                
                # Rewrite query for next attempt if we have attempts left
                if retrieval_attempts < max_retrieval_attempts:
                    query = self.rewrite_query(original_query)
                    if configs.DEBUG:
                        print(f"Rewriting query to: {query}")
                continue
            
            # Grade document relevance
            for i, doc in enumerate(docs):
                if self.grade_relevance(doc, query):
                    relevant_docs.append(doc)
            
            if configs.DEBUG:
                print(f"\n -> Found {len(relevant_docs)} relevant documents")
            
            # If we have relevant docs, break
            if len(relevant_docs) >= 1:
                break
            
            # If no relevant docs but we have docs, try query rewriting
            if len(docs) > 0 and len(relevant_docs) == 0:
                retrieval_attempts += 1
                
                # Rewrite query for next attempt if we have attempts left
                if retrieval_attempts < max_retrieval_attempts:
                    query = self.rewrite_query(original_query)
                    if configs.DEBUG:
                        print(f"No relevant docs found, rewriting query to: {query}")
                else:
                    # Last attempt - use top documents anyway
                    relevant_docs = docs[:configs.MAX_DOCS_TO_USE]
                    if configs.DEBUG:
                        print("Final attempt: using top retrieved docs despite low relevance")
                    break
            else:
                retrieval_attempts += 1
        
        # Generation loop with regeneration
        while generation_attempts < max_generation_attempts:
            if configs.DEBUG:
                print(f"\nGeneration attempt {generation_attempts + 1}")
            
            # Use relevant docs if available, otherwise use top retrieved docs
            docs_to_use = relevant_docs if relevant_docs else all_docs[:configs.MAX_DOCS_TO_USE]
            
            answer = self.generate_answer(original_query, docs_to_use)
            
            # Check for hallucinations
            if not self.check_hallucination(answer, docs_to_use):
                if configs.DEBUG:
                    print("\nHallucinations detected")
                
                generation_attempts += 1
                
                # Try regenerating if we have attempts left
                if generation_attempts < max_generation_attempts:
                    answer = self.re_generate(answer)
                    if configs.DEBUG:
                        print("Regenerating answer...")
                    
                    # Check regenerated answer
                    if self.check_hallucination(answer, docs_to_use):
                        if configs.DEBUG:
                            print("Regenerated answer is grounded")
                        # Still need to check if it's adequate
                        if self.grade_answer(answer, original_query):
                            final_answer = answer
                            break
                
                continue
            
            # Grade answer quality
            if self.grade_answer(answer, original_query):
                if configs.DEBUG:
                    print("\nAnswer adequate")
                final_answer = answer
                break
            elif generation_attempts == max_generation_attempts - 1:
                # Last attempt - accept the answer even if not perfect
                if configs.DEBUG:
                    print("\nFinal attempt: accepting answer despite quality issues")
                final_answer = answer
                break
            
            generation_attempts += 1
        
        # Fallback if no final answer
        if not final_answer:
            if configs.DEBUG:
                print("\nGenerating fallback answer")
            final_answer = self.generate_answer(
                original_query, 
                all_docs[:configs.MAX_DOCS_TO_USE] if all_docs else []
            )
        
        # Update state
        state["response"] = final_answer
        state["documents"] = relevant_docs if relevant_docs else all_docs[:configs.MAX_DOCS_TO_USE]
        state["retrieval_attempts"] = retrieval_attempts + 1
        state["generation_attempts"] = generation_attempts + 1
        state["metadata"] = {
            "relevant_docs_count": len(relevant_docs),
            "total_docs_retrieved": len(all_docs),
            "used_fallback": len(relevant_docs) == 0,
            "query_rewritten": query != original_query,
            "final_query": query
        }
        
        if self.state_manager:
            self.state_manager.save_state(state)
        
        return state