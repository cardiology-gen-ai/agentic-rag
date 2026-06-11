"""One-question smoke test for the structured KG retrieval router."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.config.manager import (
    LLMConfig,
    LLMProvider,
    NodePromptConfig,
)
from agentic_rag.kg.router import KGStructuredRouter
from agentic_rag.managers.llm_manager import LLMManager


DEFAULT_QUESTION = (
    "What are the diagnostic criteria for hypertrophic cardiomyopathy "
    "in adults and children?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one structured-routing request without querying Neo4j."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to the environment file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Natural-language question to route.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used by the router.",
    )
    parser.add_argument(
        "--nodes-config",
        type=Path,
        default=Path("configs/graphs/rag_agent.yaml"),
        help="YAML file containing the registered LLM nodes.",
    )
    return parser.parse_args()


def load_environment(env_file: Path) -> Path:
    resolved = env_file.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Environment file not found: {resolved}"
        )

    load_dotenv(
        dotenv_path=resolved,
        override=False,
    )

    return resolved


def build_router(
    *,
    model_name: str,
    nodes_config_path: Path,
) -> KGStructuredRouter:
    llm_config = LLMConfig(
        model_name=model_name,
        provider=LLMProvider.openai,
        temperature=0.0,
        router_temperature=0.0,
        generator_temperature=None,
        grader_temperature=None,
        nbits=16,
    )

    llm_manager = LLMManager(config=llm_config)

    nodes_prompt_config = NodePromptConfig(
        config=nodes_config_path,
        prompts="agentic_rag.agent.prompts",
    )

    llm_service = LLMService(
        llm_manager=llm_manager,
        nodes_prompt_config=nodes_prompt_config,
    )

    return KGStructuredRouter(
        llm_service=llm_service,
        router_config=llm_manager.router_config,
    )


def main() -> None:
    args = parse_args()

    env_path = load_environment(args.env_file)

    router = build_router(
        model_name=args.model,
        nodes_config_path=args.nodes_config,
    )

    plan = router.route(args.question)

    print("Environment file:", env_path)
    print("Model:", args.model)
    print()
    print("Question:")
    print(args.question)
    print()
    print("Structured retrieval plan:")
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
