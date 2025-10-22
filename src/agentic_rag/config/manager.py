import os
from dataclasses import field
from enum import Enum
from typing import Optional, Dict, Any, List, Literal

from pydantic import BaseModel

from cardiology_gen_ai.config.manager import ConfigManager
from cardiology_gen_ai import EmbeddingConfig, IndexingConfig


class SearchTypeNames(Enum):
    """Supported retrieval strategies for the vector store."""
    similarity = "similarity" #: Standard top-K similarity search.
    mmr = "mmr" #: Maximal Marginal Relevance (MMR) to improve diversity.
    similarity_score_threshold = "similarity_score_threshold" #: Similarity search constrained by a minimum score threshold.


class SearchConfig(BaseModel):
    """Configuration for vector store search/retrieval."""
    type: SearchTypeNames = SearchTypeNames.similarity #: :class:`~src.agentic_rag.config.manager.SearchTypeNames`, optional : Retrieval strategy to use. Defaults to :data:`~src.agentic_rag.config.manager.SearchTypeNames.similarity`.
    top_k: int #: :class:`int` : Number of results to return.
    kwargs: Dict[str, Any] = None #: :class:`dict`, optional : Backend-specific arguments passed to the retriever.
    fetch_k: Optional[int] = None #: :class:`int`, optional : Candidate pool size for certain strategies (e.g., MMR).
    score_threshold: Optional[float] = None #: :class:`float`, optional : Minimum similarity score when ``type`` is ``similarity_score_threshold``.
    fusion: Optional[bool] = None #: :class:`bool`, optional:  Enable hybrid fusion (e.g., RRF) when supported by the backend.
    metadata_filter: Optional[Dict[str, str]] = None #: :class:`dict`, optional : Structured metadata filter applied at retrieval time.

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "SearchConfig":
        search_type = SearchTypeNames(config_dict.get("type", None)) if config_dict.get("type") \
            else SearchTypeNames.similarity
        kwargs = {"k": config_dict["top_k"]}
        for k in ["fetch_k", "score_threshold"]:
            if config_dict.get(k, None) is not None:
                kwargs[k] = config_dict[k]
        other_config_dict = {k:v for k, v in config_dict.items() if k not in ["type"]}
        return cls(type=search_type, kwargs=kwargs, **other_config_dict)


class LLMConfig(BaseModel):
    """Configuration for the chat LLM backend."""
    model_name: str #: :class:`str` : Name or identifier of the underlying model deployment.
    ollama: bool = False #: :class:`bool`, optional :  If ``True``, use :ollama:`Ollama <>` backend; otherwise use :huggingface:`HuggingFace <>`. Default ``False``.
    nbits: Optional[Literal[4, 8, 16, 32]] = 8 #: :class:`typing.Literal`\[{``4``,``8``,``16``,``32``}\], optional : Precision/quantization setting (only applies to HF pipelines). Defaults to ``8``.
    generator_temperature: float #: :class:`float` :  Temperature for the generator runnable.
    router_temperature: float #: :class:`float` : Temperature for the router runnable.
    grader_temperature: float #: :class:`float` : Temperature for the grader runnable.

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        model_name = config_dict.get("deployment")
        other_config_dict = {k:v for k, v in config_dict.items() if k not in ["deployment"]}
        return cls(model_name=model_name, **other_config_dict)


class ContextConfig(BaseModel):
    """Contextual/system prompt configuration."""
    system_prompt: str = "" #: :class:`str`, optional : System prompt appended to the agent prompts. Defaults to empty string.


class ExamplesConfig(BaseModel):
    """Few-shot examples configuration for routing or prompting."""
    file: str = "" #: :class:`str`, optional : Path to the examples file (JSON or compatible).
    top_k: int = 0 #: :class:`int`, optional : Number of examples to select.
    template: str = "" #: :class:`str`, optional : Prompt template used to render examples.
    input_keys: List[str] = field(default_factory=list) #: :class:`list` of :class:`str`, optional : Input keys for example selection / formatting.


class MemoryConfig(BaseModel):
    """Conversation memory configuration."""
    length: int = 0 #: :class:`int`, optional : Maximum number of conversational turns (or turn pairs) to keep. Defaults to ``0``.


class AgentConfig(BaseModel):
    """Top-level configuration for an agent instance."""
    name: str = "" #: :class:`str`, optional : Agent display name.
    description: str = "" #: :class:`str`, optional : Short description of the agent.
    system_prompt: str = "" #: :class:`str`, optional :  Global system prompt used by the agent.
    language: str = "" #: :class:`str`, optional : Default language, used when detection is unavailable.
    allowed_languages: List[str] = field(default_factory=list) #: :class:`list` of :class:`str`, optional : Whitelist of languages the agent may use.
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig) #: :class:`cardiology_gen_ai.models.EmbeddingConfig` :  Embedding model configuration.
    search: SearchConfig = field(default_factory=SearchConfig) #: :class:`~src.agentic_rag.config.manager.SearchConfig` : Retrieval/search configuration.
    indexing: IndexingConfig = field(default_factory=IndexingConfig) #: :class:`cardiology_gen_ai.models.IndexingConfig` : Index backend configuration.
    llm: LLMConfig = field(default_factory=LLMConfig) #: :class:`~src.agentic_rag.config.manager.LLMConfig` : LLM backend configuration.
    context: ContextConfig = field(default_factory=ContextConfig) #: :class:`~src.agentic_rag.config.manager.ContextConfig` : Contextual prompt settings.
    examples: ExamplesConfig = field(default_factory=ExamplesConfig) #: :class:`~src.agentic_rag.config.manager.ExamplesConfig` : Few-shot examples settings.
    memory: MemoryConfig = field(default_factory=MemoryConfig) #: :class:`~src.agentic_rag.config.manager.MemoryConfig` : Conversation memory settings.

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        embedding_dict = config_dict["embeddings"]
        embedding_config = EmbeddingConfig.from_config(embedding_dict)
        indexing_dict = config_dict["indexing"]
        indexing_config = IndexingConfig.from_config(indexing_dict)
        search_dict = config_dict["search"]
        search_config = SearchConfig.from_config(search_dict)
        llm_dict = config_dict["llm"]
        llm_config = LLMConfig.from_config(llm_dict)
        other_config_dict = \
            {k: v for k, v in config_dict.items() if k not in ["indexing", "embeddings", "search", "llm"]}
        return cls(embeddings=embedding_config, search=search_config, indexing=indexing_config, llm=llm_config,
                   **other_config_dict)


class AgentConfigManager(ConfigManager):
    """Loader that produces a parsed :class:`~src.agentic_rag.config.manager.AgentConfig`.

    Parameters
    ----------
    config_path : :class:`str`, optional
        Path to the base configuration file. Defaults to ``os.getenv(\"CONFIG_PATH\")``.
    app_config_path : :class:`str`, optional
        Path to the per-app configuration file. Defaults to ``os.getenv(\"APP_CONFIG_PATH\")``.
    app_id : :class:`str`, optional
        Application identifier used to choose the proper section/files. Defaults to ``\"cardiology_protocols\"``.
    """
    config: AgentConfig #: :class:`AgentConfig` : The parsed agent configuration.
    def __init__(self, app_id: str, config_path: str = None, app_config_path: str = None):
        config_path = os.getenv("CONFIG_PATH") or config_path
        app_config_path = os.getenv("APP_CONFIG_PATH") or app_config_path
        super().__init__(config_path, app_config_path, app_id)
        self.config = AgentConfig.from_config(self._app_config)
