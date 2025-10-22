import logging
from typing import List

import torch
import ollama
from langchain_core.runnables import Runnable
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM

from agentic_rag.config.manager import LLMConfig, AgentConfigManager

from cardiology_gen_ai.utils.logger import get_logger


LOCAL_FILES = True


class LLMManager:
    """High-level orchestrator for an :ollama:`Ollama <>`- or :huggingface:`HuggingFace <>` -based chat LLM.

    .. rubric:: Notes

    - If ``config.ollama`` is ``True``, the manager calls :meth:`init_ollama` which
      pulls the specified model from an :ollama:`Ollama <>` server before constructing
      :langchain:`ChatOllama <ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html>`.
    - Otherwise, :meth:`init_huggingface` builds a :langchain:`ChatHuggingFace <huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace>` via a :transformers:`pipeline <main_classes/pipelines#transformers.Pipeline>` and may configure 4/8-bit quantization depending on
      ``config.nbits`` using :transformers:`BitsAndBytesConfig <main_classes/quantization#transformers.BitsAndBytesConfig>`.

    Parameters
    ----------
    config : :class:`~src.agentic_rag.config.manager.LLMConfig`
        Configuration object holding model name, backend selection, temperatures, quantization bits (for HF), and other options.
    """
    config: LLMConfig #: : :class:`~src.agentic_rag.config.manager.LLMConfig` : The configuration instance provided at construction time.
    logger: logging.Logger #: :class:`logging.Logger` : Logger used to emit lifecycle and diagnostic messages.
    llm: ChatOllama | ChatHuggingFace #: :langchain:`ChatOllama <ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html>` or :langchain:`ChatHuggingFace <huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace>`: The underlying chat model, selected according to ``config.ollama``.
    router: Runnable #: :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>` : Runnable bound with ``temperature=config.router_temperature``.
    generator: Runnable #: :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>` : Runnable bound with ``temperature=config.generator_temperature``.
    grader: Runnable #: :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>` : Runnable bound with ``temperature=config.grader_temperature``.
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = get_logger("LLM manager based on LangChain and either Ollama or HuggingFace")
        self.logger.info("Initializing LLM..")
        self.llm = self.init_ollama() if self.config.ollama else self.init_huggingface()
        self.logger.info(f"LLM {self.config.model_name} initialized successfully")
        self.router = self.llm.bind(options={"temperature": self.config.router_temperature})
        self.generator = self.llm.bind(options={"temperature": self.config.generator_temperature})
        self.grader = self.llm.bind(options={"temperature": self.config.grader_temperature})

    def init_ollama(self) -> ChatOllama:
        """Initialize and return an :langchain:`ChatOllama <ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html>` model.

        This method pulls the model specified by ``config.model_name`` from an :ollama:`Ollama <>` server and then constructs the chat model.

        Returns
        -------
        :langchain:`ChatOllama <ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html>`
            The initialized :ollama:`Ollama <>`-backed chat model.

        Raises
        ------
        Exception
            If the model cannot be pulled or initialized.
        """
        try:
            self.logger.info(f"Pulling model {self.config.model_name} from Ollama server..")
            ollama.pull(self.config.model_name)  # TODO: maybe move from here and pre-pull somewhere else
            self.logger.info(f"Model {self.config.model_name} pulled.")
            return ChatOllama(model=self.config.model_name, temperature=0)
        except Exception as e:
            self.logger.info(f"Model {self.config.model_name} could not be initialized: {e}")
            raise

    def init_huggingface(self) -> ChatHuggingFace:
        """Initialize and return a :langchain:`ChatHuggingFace <huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace>` model.

        Depending on ``config.nbits``, this method configures either 4/8-bit  quantization (via :transformers:`BitsAndBytesConfig <main_classes/quantization#transformers.BitsAndBytesConfig>`) or full
        precision, then builds a :transformers:`pipeline <main_classes/pipelines#transformers.Pipeline>` for text generation and wraps it into a
        :langchain:`HuggingFacePipeline <huggingface/llms/langchain_huggingface.llms.huggingface_pipeline.HuggingFacePipeline.html>`
        to construct :langchain:`ChatHuggingFace <huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace>`.

        Returns
        -------
        :langchain:`ChatHuggingFace <huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace>`
            The initialized :huggingface:`HuggingFace <>` -backed chat model.

        Raises
        ------
        Exception
            If the tokenizer/model cannot be loaded or the pipeline cannot be constructed.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True, local_files_only=LOCAL_FILES)
            if self.config.nbits in [4, 8]:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ) if self.config.nbits == 4 else BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    local_files_only=LOCAL_FILES,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    dtype=torch.bfloat16 if self.config.nbits == 16 else torch.bfloat32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    # attn_implementation="flash_attention_2"
                )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
            pip = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.01,
                return_full_text=False,
            )
            llm = HuggingFacePipeline(pipeline=pip)
            return ChatHuggingFace(llm=llm)
        except Exception as e:
            self.logger.info(f"Model {self.config.model_name} could not be initialized: {e}")
            raise

    def count_tokens(self, message_list: List[str]) -> int:
        # TODO: implement
        return 0


if __name__ == "__main__":
    agent_config = AgentConfigManager().config
    llm_manager = LLMManager(agent_config.llm)
