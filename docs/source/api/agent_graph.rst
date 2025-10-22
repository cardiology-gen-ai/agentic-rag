Agent Graph
===========

This module defines the core components of a Retrieval-Augmented Generation (RAG) and conversational agent built on top of the :langgraph:`LangGraph <reference/graphs>` framework.
It provides both the shared state representation exchanged between graph nodes and the agent orchestration logic
that connects language models, retrieval systems, and conversational memory into a coherent, stateful pipeline.

The :class:`~src.agent.graph.GraphState` specifies the structured state passed between nodes in the graph during each conversational turn.
It tracks key variables such as the original and contextualized user questions, retrieval results, language detection, and incremental generation attempts.

The :class:`~src.agent.graph.Agent` class implements the full conversational and retrieval pipeline as a compiled :langgraph:`CompiledStateGraph <reference/graphs/?h=compiled#langgraph.graph.state.CompiledStateGraph>`.
It integrates multiple components, including:

- LLM manager: for routing, generation, and grading tasks;
- Vector store retriever: for document search;
- Memory backend: for maintaining conversational context across turns;
- Routing and validation logic: for selecting the appropriate conversational branch (e.g., direct response vs. retrieval-augmented generation);
- Fallback mechanisms for error handling and default responses.

.. mermaid::
    :caption: Graph representingthe information flow in the agent.

      flowchart LR
        __start__(["<p>__start__</p>"]) --> language_detector("language_detector")
        contextualize_question("contextualize_question") -. &nbsp;conversational_question&nbsp; .-> conversational_agent("conversational_agent")
        contextualize_question -. &nbsp;document_based_question&nbsp; .-> document_request_detector("document_request_detector")
        document_request_detector --> retrieve("retrieve")
        generate("generate") -. &nbsp;grounded_and_addressed_question&nbsp; .-> __end__(["<p>__end__</p>"])
        generate -.-> generate_default_response("generate_default_response")
        generate -. &nbsp;grounded_but_not_addressed_question&nbsp; .-> transform_question("transform_question")
        language_detector --> contextualize_question
        retrieval_grader("retrieval_grader") -. &nbsp;at_least_one_doc_relevant&nbsp; .-> generate
        retrieval_grader -. &nbsp;generate_document_request_response&nbsp; .-> generate_document_response("generate_document_response")
        retrieval_grader -. &nbsp;all_docs_not_relevant&nbsp; .-> transform_question
        retrieve --> retrieval_grader
        transform_question -.-> generate_default_response & retrieve
        conversational_agent --> __end__
        generate_default_response --> __end__
        generate_document_response --> __end__
        generate -. &nbsp;generation_not_grounded&nbsp; .-> generate

        __start__:::first
        __end__:::last
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc

.. automodule:: src.agentic_rag.agent.graph
   :members:
   :exclude-members: from_config, model_config
   :member-order: bysource
