LLM Manager
===========

This module defines the high-level functions for managing the agent's LLM backend,
supporting both :ollama:`Ollama <>` and :huggingface:`HuggingFace <>` models. The manager is responsible for initializing
the appropriate backend based on configuration and creating temperature-bound runnables for routing, generation, and grading.


.. rubric:: Notes

- :ollama:`Ollama <>` backend requires pulling the model from an Ollama server before use.
- :huggingface:`HuggingFace <>` backend supports quantization options for memory-efficient model loading.
- All runnables (router, generator, grader) are bound to temperatures specified in the configuration.


.. automodule:: src.managers.llm_manager
   :members:
   :exclude-members: from_config, model_config
   :member-order: bysource