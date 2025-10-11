import logging
from typing import List

import torch
import ollama
from langchain_core.runnables import Runnable
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM

from src.config.manager import LLMConfig, AgentConfigManager

from cardiology_gen_ai.utils.logger import get_logger


LOCAL_FILES = True


class LLMManager:
    """High-level orchestrator for an Ollama- or HuggingFace-based chat LLM.

    The manager decides which backend to use based on ``config.ollama`` and then initializes the corresponding chat model.
    It also creates three temperature-bound runnables (:attr:`~src.managers.llm_manager.router`, :attr:`~src.managers.llm_manager.generator`, :attr:`~src.managers.llm_manager.grader`).

    .. rubric:: Notes

    - If ``config.ollama`` is ``True``, the manager calls :meth:`init_ollama` which
      pulls the specified model from an :mod:`ollama` server before constructing
      :class:`ChatOllama`.
    - Otherwise, :meth:`init_huggingface` builds a :class:`ChatHuggingFace` via a :func:`~transformers.pipeline` and may configure 4/8-bit quantization depending on
      ``config.nbits`` using :class:`~transformers.BitsAndBytesConfig`.

    Parameters
    ----------
    config : :class:`~src.config.manager.LLMConfig`
        Configuration object holding model name, backend selection, temperatures, quantization bits (for HF), and other options.
    """
    config: LLMConfig #: : :class:`~src.config.manager.LLMConfig` : The configuration instance provided at construction time.
    logger: logging.Logger #: :class:`logging.Logger` : Logger used to emit lifecycle and diagnostic messages.
    llm: ChatOllama | ChatHuggingFace #: :class:`ChatOllama` or :class:`ChatHuggingFace` : The underlying chat model, selected according to ``config.ollama``.
    router: Runnable #: :class:`~langchain_core.runnables.Runnable` : Runnable bound with ``temperature=config.router_temperature``.
    generator: Runnable #: :class:`~langchain_core.runnables.Runnable` : Runnable bound with ``temperature=config.generator_temperature``.
    grader: Runnable #: :class:`~langchain_core.runnables.Runnable` : Runnable bound with ``temperature=config.grader_temperature``.
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
        """Initialize and return an :class:`ChatOllama` model.

        This method pulls the model specified by ``config.model_name`` from an :mod:`ollama` server and then constructs the chat model.

        Returns
        -------
        :class:`ChatOllama`
            The initialized Ollama-backed chat model.

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
        """Initialize and return a :class:`ChatHuggingFace` model.

        Depending on ``config.nbits``, this method configures either 4/8-bit  quantization (via :class:`~transformers.BitsAndBytesConfig`) or full
        precision, then builds a :func:`~transformers.pipeline` for text generation and wraps it into a
        :class:`~langchain_community.llms.huggingface_pipeline.HuggingFacePipeline`
        to construct :class:`ChatHuggingFace`.

        Returns
        -------
        :class:`ChatHuggingFace`
            The initialized HuggingFace-backed chat model.

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
            pip = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
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
