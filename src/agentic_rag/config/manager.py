import os
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from importlib.resources import files

from pydantic import BaseModel, ConfigDict

from cardiology_gen_ai.config.manager import ConfigManager
from cardiology_gen_ai import IndexingConfig, RetrievalTypeNames, IndexTypeNames


class LLMProvider(Enum):
    ollama = "ollama"
    huggingface = "huggingface"
    vllm = "vllm"
    openai = "openai"


class LLMConfig(BaseModel):
    """Configuration for the chat LLM backend."""
    model_name: str #: :class:`str` : Name or identifier of the underlying model deployment.
    provider: LLMProvider
    nbits: Optional[Literal[4, 8, 16, 32]] = 8 #: :class:`typing.Literal`\[{``4``,``8``,``16``,``32``}\], optional : Precision/quantization setting (only applies to HF pipelines). Defaults to ``8``.
    generator_temperature: Optional[float] = None #: :class:`float` :  Temperature for the generator runnable.
    router_temperature: Optional[float] = None #: :class:`float` : Temperature for the router runnable.
    grader_temperature: Optional[float] = None #: :class:`float` : Temperature for the grader runnable.
    temperature: float = 0.0

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        model_name = config_dict.get("deployment", "")
        other_config_dict = {k:v for k, v in config_dict.items() if k not in ["deployment"]}
        return cls(model_name=model_name, **other_config_dict)


class SearchTypeNames(Enum):
    """Supported retrieval strategies for the vector store."""
    similarity = "similarity" #: Standard top-K similarity search.
    mmr = "mmr" #: Maximal Marginal Relevance (MMR) to improve diversity.
    similarity_score_threshold = "similarity_score_threshold" #: Similarity search constrained by a minimum score threshold.


class FusionStrategy(Enum):
    rrf = "rrf"
    reranking = "reranking"
    cross_encoder = "cross_encoder"
    bm25 = "bm25"


class SearchConfig(BaseModel):
    """Configuration for vector store search/retrieval."""
    type: SearchTypeNames = SearchTypeNames.similarity #: :class:`~src.agentic_rag.config.manager.SearchTypeNames`, optional : Retrieval strategy to use. Defaults to :data:`~src.agentic_rag.config.manager.SearchTypeNames.similarity`.
    top_k: int #: :class:`int` : Number of results to return.
    k: int #: :class:`int` : Number of chunks to retrieve.
    kwargs: Dict[str, Any] = None #: :class:`dict`, optional : Backend-specific arguments passed to the retriever.
    fetch_k: Optional[int] = None #: :class:`int`, optional : Candidate pool size for certain strategies (e.g., MMR).
    score_threshold: Optional[float] = None #: :class:`float`, optional : Minimum similarity score when ``type`` is ``similarity_score_threshold``.
    fusion: Optional[FusionStrategy] = None #: :class:`bool`, optional:  Enable hybrid fusion (e.g., RRF) when supported by the backend.
    metadata_filter: Optional[Dict[str, str]] = None #: :class:`dict`, optional : Structured metadata filter applied at retrieval time.
    reranker: Optional[LLMConfig] = None
    cross_encoder: Optional[str] = None

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "SearchConfig":
        search_type = SearchTypeNames(config_dict.get("type", None)) if config_dict.get("type") \
            else SearchTypeNames.similarity
        kwargs = {"k": config_dict["k"]}
        for k in ["fetch_k", "score_threshold"]:
            if config_dict.get(k, None) is not None:
                kwargs[k] = config_dict[k]
        reranker_config, cross_encoder, reranker = [None for _ in range(3)]
        fusion_strategy = FusionStrategy(config_dict["fusion"]) if config_dict.get("fusion") else None
        if fusion_strategy == FusionStrategy.reranking:
            reranker = config_dict.get("reranker", {})
            reranker_config = LLMConfig.from_config(reranker) if reranker is not {} else None
        elif fusion_strategy == FusionStrategy.cross_encoder:
            cross_encoder = config_dict["cross_encoder"]
        other_config_dict = {k:v for k, v in config_dict.items() if k not in ["type", "reranker", "cross_encoder", "fusion"]}
        return cls(type=search_type, kwargs=kwargs, reranker=reranker_config, cross_encoder=cross_encoder,
                   fusion=fusion_strategy, **other_config_dict)


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


class NodePromptConfig(BaseModel):
    config: Path | str
    prompts: str

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "NodePromptConfig":
        config = Path(config_dict["config"]) if isinstance(config_dict["config"], str) else config_dict["config"]
        prompts = config_dict["prompts"]
        return cls(config=config, prompts=prompts)


class AgentConfig(BaseModel):
    """Top-level configuration for an agent instance."""
    name: str = "" #: :class:`str`, optional : Agent display name.
    description: str = "" #: :class:`str`, optional : Short description of the agent.
    system_prompt: str = "" #: :class:`str`, optional :  Global system prompt used by the agent.
    language: str = "" #: :class:`str`, optional : Default language, used when detection is unavailable.
    allowed_languages: List[str] = field(default_factory=list) #: :class:`list` of :class:`str`, optional : Whitelist of languages the agent may use.
    search: SearchConfig = field(default_factory=SearchConfig) #: :class:`~src.agentic_rag.config.manager.SearchConfig` : Retrieval/search configuration.
    indexing: IndexingConfig | List[IndexingConfig] = field(default_factory=IndexingConfig) #: :class:`cardiology_gen_ai.models.IndexingConfig` : Index backend configuration.
    llm: LLMConfig = field(default_factory=LLMConfig) #: :class:`~src.agentic_rag.config.manager.LLMConfig` : LLM backend configuration.
    context: ContextConfig = field(default_factory=ContextConfig) #: :class:`~src.agentic_rag.config.manager.ContextConfig` : Contextual prompt settings.
    examples: ExamplesConfig = field(default_factory=ExamplesConfig) #: :class:`~src.agentic_rag.config.manager.ExamplesConfig` : Few-shot examples settings.
    memory: MemoryConfig = field(default_factory=MemoryConfig) #: :class:`~src.agentic_rag.config.manager.MemoryConfig` : Conversation memory settings.
    nodes: NodePromptConfig = field(default_factory=NodePromptConfig)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        indexing_dict = config_dict["indexing"]
        indexing_config = IndexingConfig.from_config(indexing_dict) if isinstance(indexing_dict, Dict) \
            else [IndexingConfig.from_config(index) for index in indexing_dict]
        search_dict = config_dict["search"]
        search_config = SearchConfig.from_config(search_dict)
        llm_dict = config_dict["llm"]
        llm_config = LLMConfig.from_config(llm_dict)
        nodes_dict = config_dict["nodes"]
        nodes_config = NodePromptConfig.from_config(nodes_dict)
        other_config_dict = \
            {k: v for k, v in config_dict.items() if k not in ["indexing", "embeddings", "search", "llm", "nodes"]}
        return cls(search=search_config, indexing=indexing_config, llm=llm_config, nodes=nodes_config, **other_config_dict)


class AgentConfigManager(ConfigManager):
    """Loader that produces a parsed :class:`~src.agentic_rag.config.manager.AgentConfig`.

    Parameters
    ----------
    config_path : :class:`str`, optional
        Path to the base configuration file. Defaults to ``os.getenv(\"CONFIG_PATH\")``.
    app_id : :class:`str`, optional
        Application identifier used to choose the proper section/files. Defaults to ``\"cardiology_protocols\"``.
    """
    config: AgentConfig #: :class:`AgentConfig` : The parsed agent configuration.
    def __init__(self, app_id: str, config_path: str = None):
        config_path = os.getenv("CONFIG_PATH") or config_path
        super().__init__(config_path, app_id)
        self._load_indexing_config()
        self.config = AgentConfig.from_config(self._config)

    def _load_single_indexing_config(self, info_config: Dict, filename: str) -> Dict | None:
        index_folder = Path(info_config["folder"]) / filename or Path(os.getenv("INDEX_ROOT")) / filename
        loaded_config = self._load_config(config_path=index_folder)
        if loaded_config is not None:
            assert (loaded_config["indexing"]["name"] == info_config["name"])
            if loaded_config["indexing"].get("distance") and info_config.get("distance"):
                assert (loaded_config["indexing"]["distance"] == info_config["distance"])
            loaded_type = loaded_config["indexing"]["type"] if isinstance(loaded_config["indexing"]["type"], list) \
                else [loaded_config["indexing"]["type"]]
            assert (info_config["type"] in loaded_type)
            loaded_config["indexing"]["type"] = info_config["type"]
            return loaded_config
        return None

    def _load_indexing_config(self, filename="config.json"):
        info_config = self._config["indexing"]
        if isinstance(info_config, Dict):
            loaded_config = self._load_single_indexing_config(info_config, filename)
            info_folder = info_config.get("folder")
            retrieval_mode = info_config.get("retrieval_mode")
            self._config["indexing"] = loaded_config["indexing"]
            if info_folder:
                self._config["indexing"]["folder"] = info_folder
            if retrieval_mode:
                self._config["indexing"]["retrieval_mode"] = retrieval_mode
            if (retrieval_mode in [RetrievalTypeNames.dense.value, RetrievalTypeNames.hybrid.value]
                    or info_config["type"] != IndexTypeNames.bm25.value):
                self._config["indexing"]["embeddings"] = loaded_config["embeddings"]

        else:
            assert isinstance(info_config, List)
            index_configs = []
            for index_config in info_config:
                index_folder = index_config.get("folder")
                index_retrieval_mode = index_config.get("retrieval_mode")
                current_loaded_config = self._load_single_indexing_config(index_config, filename)
                if index_folder:
                    current_loaded_config["indexing"]["folder"] = index_folder
                if index_retrieval_mode:
                    current_loaded_config["indexing"]["retrieval_mode"] = index_retrieval_mode
                if (index_retrieval_mode in [RetrievalTypeNames.dense.value, RetrievalTypeNames.hybrid.value]
                        or index_config["type"] != IndexTypeNames.bm25.value):
                    current_loaded_config["indexing"]["embeddings"] = current_loaded_config["embeddings"]
                index_configs.append(current_loaded_config["indexing"])
            self._config["indexing"] = index_configs
