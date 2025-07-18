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

from sqlite.manager import StateManager
from agent import configs
from agent.state import State

class RAG:
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
    
    def retrieve_documents(self, state: State) -> List[Document]:
        """Retrieve relevant documents."""
        query = state.get('message') if state.get('is_query') else state.get('rewritten_query', '')
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
    
    def grade_relevance(self, document: Document, state: State) -> bool:
        """Grade document relevance to query - more lenient grading."""
        query = state.get('message') if state.get('is_query') else state.get('rewritten_query', '')
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
    
    def generate_answer(self, state: State) -> str:
        """Generate answer based on query and documents."""
        query = state.get('message') if state.get('is_query') else state.get('rewritten_query', '')
        documents = state.get('documents')
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
        state['previous_messages'].append(state.get("message"))
        state['is_query'] = False

        return result.content.strip()
    
    def check_hallucination(self, state: State, answer: str) -> bool:
        """Check if answer is grounded in documents."""
        if not state.get("documents"):
            return False
        
        # Combine all document content
        context = "\n\n".join([doc.page_content for doc in state.get("documents")])
        
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
        # answer parameter is passed to the function, no need to extract from state
        prompt_text = f"""Grade if this answer adequately addresses the medical query.

Query: {query}
Answer: {answer}

Respond with only: adequate OR inadequate"""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(prompt_text, config=config)
        
        return "adequate" in result.content.lower()
    
    def rewrite_query(self, state: State) -> str:
        """Rewrite the query when the retrieve grade is negative."""
        if state.get('rewritten_query'):
            query = state.get('rewritten_query')
        else:
            query = state.get("message")
        
        prompt_text = f"""You are a medical query optimizer. Rewrite the following query to improve retrieval of relevant medical documents.

Original query: {query}

Make the query more specific, add relevant medical terminology, and focus on key clinical concepts.
Respond with only the rewritten query."""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.llm.invoke(prompt_text, config=config)
        state['rewritten_query'] = result.content.strip()

        return result.content.strip()

    def re_generate(self, state: State) -> str:
        """Re-generate the llm response when hallucinations happen."""
        prompt_text = f"""The previous answer contained information not supported by the provided context. 
Generate a more conservative and grounded response that only uses information directly supported by the medical documents.

Previous answer: {state.get('message')}

Provide a revised answer that is more cautious and clearly indicates when information is not available."""
        
        config = {"callbacks": [self.callback_handler]} if self.callback_handler else {}
        
        result = self.generation_llm.invoke(prompt_text, config=config)
        
        return result.content.strip()
    
    def update_state(self, state: State) -> State:
        """Main self-RAG node function with query rewriting and answer regeneration."""
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
                    print(f"Using rewritten query: {state.get('rewritten_query', 'None')}")
            
            docs = self.retrieve_documents(state)
            all_docs = docs
            
            if not docs:
                retrieval_attempts += 1
                print(f"\nNo documents retrieved on attempt {retrieval_attempts}")
                
                # Rewrite query for next attempt if we have attempts left
                if retrieval_attempts < max_retrieval_attempts:
                    rewritten_query = self.rewrite_query(state)
                    if configs.DEBUG:
                        print(f"Rewriting query to: {rewritten_query}")
                continue
            
            # Grade document relevance
            for i, doc in enumerate(docs):
                if self.grade_relevance(doc, state):
                    relevant_docs.append(doc)

            state['documents'] = relevant_docs
            
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
                    query = self.rewrite_query(state)
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
            
            answer = self.generate_answer(state)
            
            # Check for hallucinations
            if not self.check_hallucination(state, answer):
                if configs.DEBUG:
                    print("\nHallucinations detected")
                
                generation_attempts += 1
                
                # Try regenerating if we have attempts left
                if generation_attempts < max_generation_attempts:
                    answer = self.re_generate(state)
                    if configs.DEBUG:
                        print("Regenerating answer...")
                    
                    # Check regenerated answer
                    if self.check_hallucination(state, answer):
                        if configs.DEBUG:
                            print("Regenerated answer is grounded")
                        # Still need to check if it's adequate
                        if self.grade_answer(answer, state.get("message")):
                            final_answer = answer
                            break
                
                continue
            
            # Grade answer quality
            if self.grade_answer(answer, state.get("message")):
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
       
        # Set the final answer in state
        if final_answer:
            state["response"] = final_answer
        else:
            state["response"] = "I couldn't find relevant information to answer your question."
            
        return state
