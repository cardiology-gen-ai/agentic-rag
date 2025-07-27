#!/usr/bin/env python3 
"""
Utility functions that act as nodes in the agent.
"""

from datetime import datetime 

from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore

def conversational_agent(llm, system_prompt: str):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f"Today is {current_datetime}. \n" + system_prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}, \nChat history: {history}"),
        ]
    )
    contextualizer = prompt | llm | StrOutputParser()
    return contextualizer
