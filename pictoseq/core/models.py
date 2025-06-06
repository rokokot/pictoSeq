# usr/bin/env python3

"""
This is where we are loading / importing models from for our other components. accordingly this contains a lot of configs and parameters which we want to keep elsewhere for easier changes down the line.

Handles things like model loading from hf api, tokenizer setup, etc


"""


import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import torch

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,T5Tokenizer,GenerationConfig)


@dataclass
class ModelConfig:
    """Configuration for different model architectures."""
    name: str
    model_id: str
    tokenizer_class: str
    description: str
    is_multilingual: bool = False


class ModelRegistry:
    
    @classmethod
    def get_all_models(cls) -> List[ModelConfig]:
        return [
            
           # standard model config entries
            ModelConfig(
                name="barthez",
                model_id="moussaKam/barthez",
                tokenizer_class="AutoTokenizer",
                description="French BART model optimized for French generation",
                is_multilingual=False
            ),
    
    
            ModelConfig(
                name="french_t5",
                model_id="plguillou/t5-base-fr-sum-cnndm", 
                tokenizer_class="T5Tokenizer",
                description="French T5 model fine-tuned for summarization",
                is_multilingual=False
            ),
            ModelConfig(
                name="mt5_base",
                model_id="google/mt5-base",
                tokenizer_class="T5Tokenizer", 
                description="Multilingual T5 base model",
                is_multilingual=True
            )
        ]
    
    @classmethod
    def get_model_by_name(cls, name: str) -> Optional[ModelConfig]:
        models = cls.get_all_models()
        return next((m for m in models if m.name == name), None)
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        return [m.name for m in cls.get_all_models()]


class ModelLoader:
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_tokenizer(self, model_config: ModelConfig) -> Tuple[object, object]:
        
        self.logger.info(f"Setting up model: {model_config.name}")
        
      #  tokenizer
        tokenizer = self._load_tokenizer(model_config)
        
  #  model
        model = self._load_model(model_config)
        
        #  tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        #  generation config with safe defaults
        self._setup_generation_config(model, tokenizer)
        
        model.to(self.device)
        
        param_count = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model loaded: {param_count:,} parameters")
        
        return model, tokenizer
    
    def _load_tokenizer(self, model_config: ModelConfig):
        
# model specific configs

        if model_config.tokenizer_class == "T5Tokenizer":
            return T5Tokenizer.from_pretrained(model_config.model_id)
        else:
            try:
                return AutoTokenizer.from_pretrained(model_config.model_id, use_fast=True)
            except Exception as e:
                self.logger.warning(f"Fast tokenizer failed: {e}")
                return AutoTokenizer.from_pretrained(model_config.model_id, use_fast=False)
    
    def _load_model(self, model_config: ModelConfig):
# load the model

        return AutoModelForSeq2SeqLM.from_pretrained(model_config.model_id)
    
    def _setup_generation_config(self, model, tokenizer):
        try:
            generation_config = GenerationConfig.from_model_config(model.config)
        except:
            generation_config = GenerationConfig()
        
        generation_config.max_length = 128
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.early_stopping = False  
        generation_config.num_beams = 1 
        generation_config.do_sample = False
        
        try:
            generation_config.validate()
            model.generation_config = generation_config
            self.logger.info(" config validated and set successfully")
        except Exception as e:
            self.logger.warning(f" config validation failed: {e}")
            # Create minimal safe config


            safe_config = GenerationConfig(
                max_length=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
                num_beams=1,
                do_sample=False
            )
            model.generation_config = safe_config
            self.logger.info("Using minimal safe generation config")


def create_model_loader(device: Optional[torch.device] = None) -> ModelLoader:
    return ModelLoader(device)


def get_available_models() -> List[str]:
    return ModelRegistry.get_model_names()


def load_model(model_name: str, device: Optional[torch.device] = None) -> Tuple[object, object]:
   
    model_config = ModelRegistry.get_model_by_name(model_name)
    if not model_config:
        available = ModelRegistry.get_model_names()
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    loader = ModelLoader(device)
    return loader.load_model_and_tokenizer(model_config)


ModelConfig = ModelConfig
ExperimentalMatrix = ModelRegistry  