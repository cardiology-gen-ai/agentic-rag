from typing import List

import torch
import ollama
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM

from src.config.manager import LLMConfig, AgentConfigManager

from cardiology_gen_ai.utils.logger import get_logger


class LLMManager:
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
        try:
            self.logger.info(f"Pulling model {self.config.model_name} from Ollama server..")
            ollama.pull(self.config.model_name)  # TODO: maybe move from here and pre-pull somewhere else
            self.logger.info(f"Model {self.config.model_name} pulled.")
            return ChatOllama(model=self.config.model_name, temperature=0)
        except Exception as e:
            self.logger.info(f"Model {self.config.model_name} could not be initialized: {e}")
            raise

    def init_huggingface(self) -> ChatHuggingFace:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
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
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.bfloat16 if self.config.nbits == 16 else torch.bfloat32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"
                )
            pip = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                # max_new_tokens=512,
                do_sample=True,
                temperature=0.01,
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
