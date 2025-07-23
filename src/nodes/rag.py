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

sys.path.append('../')
import configs
from state import State

class RAG:
    def __init__(self,
                 vectorstore, 
                 llm_model: str = configs.LLM_MODEL,
                 generation_temperature: float = configs.LLM_GENERATION_TEMPERATURE,
                 temperature: float = configs.LLM_TEMPERATURE,
                 state_manager = None, 
                 callback_handler = None):
        self.vectorstore = vectorstore
        self.llm_model = llm_model
        self.llm = ChatOllama(model=llm_model, temperature=temperature, verbose=False)
        self.generation_llm = ChatOllama(model=llm_model, temperature=generation_temperature, verbose=False)
        self.state_manager = state_manager
        self.callback_handler = callback_handler
        
        if vectorstore:
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

    def grade_relevance(self, query: str, document: Document) -> bool:
        """Grade document relevance to query - more lenient grading."""
        prompt_text = f"""Is this document relevant to answer the question?

Question: {query}

Document excerpt: {document.page_content[:configs.MAX_CONTEXT_LENGTH]}

Answer with just one word: relevant or not_relevant"""
        
        result = self.llm.invoke(prompt_text)
        response = result.content.strip().lower()
        
        is_relevant = "relevant" in response and "not" not in response
        if configs.DEBUG:
            print(f"\nDocument relevant: {is_relevant}")
        return is_relevant

    def generate_answer(self, query: str, relevant_docs: List[Document]) -> str:
        """Generate answer based on query and documents."""
        context = "\n\n---\n\n".join([
                f"From {doc.metadata.get('source_file', 'Unknown source')}:\n{doc.page_content}"
                for doc in relevant_docs
            ])
        prompt_text = f"""You are a medical expert assistant. Use the following context to answer the question.

Question: {query}

Provide a clear, helpful answer based on the context. If the context doesn't fully answer the question, say what you can answer and what information is missing."""
        
        result = self.generation_llm.invoke(prompt_text)
        return result.content.strip()

    def check_hallucination(self, documents: List[Document], answer: str) -> bool:
        """Check if answer is grounded in documents."""
        
        context = "\n\n---\n\n".join([
                f"From {doc.metadata.get('source_file', 'Unknown source')}:\n{doc.page_content}"
                for doc in documents
            ])
        
        if len(context) > configs.MAX_CONTEXT_LENGTH:
            context = context[:configs.MAX_CONTEXT_LENGTH]
        
        prompt_text = f"""Check if the answer is supported by the provided context.

Context: {context}

Answer: {answer}

Respond with only: grounded OR hallucination"""
        
        result = self.llm.invoke(prompt_text)
        return "grounded" in result.content.lower()

    def grade_answer(self, answer: str, query: str) -> bool:
        """Grade if answer adequately addresses the query."""
        prompt_text = f"""Grade if this answer adequately addresses the medical query.

Query: {query}
Answer: {answer}

Respond with only: adequate OR inadequate"""
        
        result = self.llm.invoke(prompt_text)
        
        return "adequate" in result.content.lower()

    def rewrite_query(self, query:str) -> str:
        """Rewrite the query when the retrieve grade is negative."""
        prompt_text = f"""You are a medical query optimizer. Rewrite the following query to improve retrieval of relevant medical documents.

Original query: {query}

Make the query more specific, add relevant medical terminology, and focus on key clinical concepts.
Respond with only the rewritten query."""
        
        result = self.llm.invoke(prompt_text)
        return result.content.strip()

        """Re-generate the llm response when hallucinations happen."""
        prompt_text = f"""The previous answer contained information not supported by the provided context. 
Generate a more conservative and grounded response that only uses information directly supported by the medical documents.

Previous answer: {answer}

Provide a revised answer that is more cautious and clearly indicates when information is not available."""
        
        result = self.generation_llm.invoke(prompt_text)
        return result.content.strip()
    
    def update_state(self, state: State) -> State:
        """Main self-RAG node function with query rewriting and answer regeneration."""
       
        retrieval_attempts = 0
        generation_attempts = 0
        final_answer = None
        relevant_docs = []

        query = state.get('message')
        
        # Retrieval loop with query rewriting
        while retrieval_attempts < configs.MAX_RETRIEVAL_ATTEMPTS:
            if configs.DEBUG:
                print(f"\nRetrieval attempt {retrieval_attempts + 1}")
                if retrieval_attempts > 0:
                    print(f"Using rewritten query: {state.get('rewritten_query', 'None')}")
            
            docs = self.retrieve_documents(query)
            
            if not docs:
                retrieval_attempts += 1
                print(f"\nNo documents retrieved on attempt {retrieval_attempts}")

                if retrieval_attempts < configs.MAX_RETRIEVAL_ATTEMPTS:
                    if state.get('rewritten_query') != "":
                        rewritten_query = self.rewrite_query(state.get('rewritten_query'))
                    else: 
                        rewritten_query = self.rewrite_query(query)
                    state['rewritten_query'] = rewritten_query
                    if configs.DEBUG:
                        print(f"Rewriting query to: {rewritten_query}")
                continue
            
            # Grade document relevance
            for i, doc in enumerate(docs):
                if self.grade_relevance(query, doc):
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
                
                if retrieval_attempts < configs.MAX_RETRIEVAL_ATTEMPTS:
                    if state.get('rewritten_query') != "":
                        rewritten_query = self.rewrite_query(state.get('rewritten_query'))
                    else: 
                        rewritten_query = self.rewrite_query(query)
                    state['rewritten_query'] = rewritten_query

                    if configs.DEBUG:
                        print(f"No relevant docs found, rewriting query to: {rewritten_query}")
                else:
                    # Last attempt - use top documents anyway
                    relevant_docs = docs[:configs.MAX_DOCS_TO_USE]
                    if configs.DEBUG:
                        print("Final attempt: using top retrieved docs despite low relevance")
                    break
            else:
                retrieval_attempts += 1
        
        # Generation loop with regeneration
        while generation_attempts < configs.MAX_GENERATION_ATTEMPTS:
            if configs.DEBUG:
                print(f"\nGeneration attempt {generation_attempts + 1}")
            
            docs_to_use = relevant_docs if relevant_docs else all_docs[:configs.MAX_DOCS_TO_USE]
            
            answer = self.generate_answer(query, docs_to_use)
 
            # Check for hallucinations
            if not self.check_hallucination(docs_to_use, answer):
                if configs.DEBUG:
                    print("\nHallucinations detected")
                
                generation_attempts += 1
                
                if generation_attempts < configs.MAX_GENERATION_ATTEMPTS:
                    answer = self.re_generate(answer)
                    if configs.DEBUG:
                        print("Regenerating answer...")
                continue
            
            # Grade answer quality
            if self.grade_answer(answer, query):
                if configs.DEBUG:
                    print("\nAnswer adequate")
                final_answer = answer
                break
            elif generation_attempts == max_generation_attempts - 1:
                if configs.DEBUG:
                    print("\nFinal attempt: accepting answer despite quality issues")
                final_answer = answer
                break
            
            generation_attempts += 1
       
        # Set the final answer in state
        state['previous_messages'].append(state.get('message'))
        if final_answer:
            state["message"] = final_answer
        else:
            state["message"] = "I couldn't find relevant information to answer your question."
            
        return state
