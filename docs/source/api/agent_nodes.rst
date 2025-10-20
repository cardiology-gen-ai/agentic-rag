Agent Nodes
===========

This module defines a collection of composable runnable factories used to construct the primitive pipelines
that power the agent's behaviour. Each factory returns a
:langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>` implementing
a small, well-scoped responsibility (for example: language detection, question contextualization, routing,
document grading, generation, and validation). The runnables are assembled from prompt templates, lightweight
parsers, and small transformation lambdas so that they can be composed into larger processing graphs (see
:class:`~src.agent.graph.Agent`).

.. rubric:: Design overview

- Single-responsibility runnables. Each function builds a runnable that performs one canonical task.

- Prompt-driven pipelines. Runnables are composed from a system/human prompt pair (via
  :langchain_core:`ChatPromptTemplate <prompts/langchain_core.prompts.chat.ChatPromptTemplate.html>`) followed by an LLM invocation.

- Structured output and validation. Where deterministic structure is required (routing, grading, grounding,
  document-request detection, language detection), the pipeline uses a JSON parser
  (:langchain_core:`JsonOutputParser <output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html>`)
  backed by Pydantic output models. Parsers are followed by light-weight model validation to
  return typed instances.

.. automodule:: src.agent.output
   :members:
   :exclude-members: _strip_think, from_config, model_config
   :member-order: bysource


.. automodule:: src.agent.nodes
   :members:
   :exclude-members: _strip_think, from_config, model_config
   :member-order: bysource
