from typing import List

from agentic_rag.agent.services.context_service import ContextService
from agentic_rag.agent.services.generation_service import GenerationService
from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.agent.services.retrieval_service import RetrievalService
from agentic_rag.agent.services.routing_service import RoutingService
from agentic_rag.config.manager import NodePromptConfig
from agentic_rag.managers.llm_manager import LLMManager
from agentic_rag.managers.search_manager import SearchManager


class ServiceContainer:
    def __init__(
            self, llm_manager: LLMManager, nodes_prompt_config: NodePromptConfig,
            search_manager: SearchManager, agent_prompt: str, index_description: str,
            allowed_languages: List[str],
    ):
        self.llm_service = LLMService(llm_manager=llm_manager, nodes_prompt_config=nodes_prompt_config)
        self.context_service = ContextService(self.llm_service, allowed_languages=allowed_languages)
        self.generation_service = GenerationService(
            self.llm_service, llm_manager.generator_config, agent_prompt=agent_prompt
        )
        self.routing_service = RoutingService(
            self.llm_service, llm_manager.router_config, index_description=index_description
        )
        self.retrieval_service = RetrievalService(self.llm_service, search_manager, llm_manager.grader_config)