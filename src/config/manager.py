import os
from enum import Enum
from typing import Optional, Dict, Any

from cardiology_gen_ai import EmbeddingConfig, IndexingConfig
from pydantic import BaseModel

from cardiology_gen_ai.config.manager import ConfigManager


class SearchTypeNames(Enum):
    similarity = "similarity"
    mmr = "mmr"
    similarity_score_threshold = "similarity_score_threshold"


class SearchConfig(BaseModel):
    type: SearchTypeNames = SearchTypeNames.similarity
    top_k: int
    kwargs: Dict[str, Any] = None
    fetch_k: Optional[int] = None
    score_threshold: Optional[float] = None
    fusion: Optional[bool] = None
    metadata_filter: Optional[Dict[str, str]] = None

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


class AgentConfig(BaseModel):
    name: str = ""
    description: str = ""
    embeddings: EmbeddingConfig
    search: SearchConfig
    indexing: IndexingConfig

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        embedding_dict = config_dict["embeddings"]
        embedding_config = EmbeddingConfig.from_config(embedding_dict)
        indexing_dict = config_dict["indexing"]
        indexing_config = IndexingConfig.from_config(indexing_dict)
        search_dict = config_dict["search"]
        search_config = SearchConfig.from_config(search_dict)
        other_config_dict = \
            {k: v for k, v in config_dict.items() if k not in ["indexing", "embeddings", "search"]}
        return cls(embeddings=embedding_config, search=search_config, indexing=indexing_config, **other_config_dict)


class AgentConfigManager(ConfigManager):
    def __init__(self,
                 config_path: str = os.getenv("CONFIG_PATH"),
                 app_config_path: str = os.getenv("APP_CONFIG_PATH"),
                 app_id: str = "cardiology_protocols"):
        super().__init__(config_path, app_config_path, app_id)
        self.config = AgentConfig.from_config(self._app_config)
