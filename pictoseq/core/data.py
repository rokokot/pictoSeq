#!/usr/bin/env/ python3

"""
Data preprocessing module; handles everything from data loading, processing, splitting, and data handling.

based of off old propicto_processor.py

"""



import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from datasets import Dataset


def safe_json_dump(obj, fp, **kwargs):
    kwargs.setdefault('ensure_ascii', False)
    kwargs.setdefault('indent', 2)
    
    if hasattr(fp, 'write'):
        json.dump(obj, fp, **kwargs)
    else:
        with open(fp, 'w', encoding='utf-8', newline='') as f:
            json.dump(obj, f, **kwargs)


def safe_json_load(fp):
    if hasattr(fp, 'read'):
        return json.load(fp)
    else:
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)


def safe_text_encode(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return text


@dataclass
class DataConfig:
    name: str
    path: str
    description: str
    task_prefix_template: str


class DataRegistry:   # here we define the config for input formats, can be adjusted for custom datasets
    
    @classmethod
    def get_all_configs(cls) -> List[DataConfig]:
        
        return [
            DataConfig(
                name="keywords_to_sentence",
                path="keywords_to_sentence",
                description="ARASAAC keywords to French sentences",
                task_prefix_template="Corriger et compléter: {input}"
            ),
            DataConfig(
                name="pictos_tokens_to_sentence", 
                path="pictos_tokens_to_sentence",
                description="Pictogram tokens to French sentences",
                task_prefix_template="Transformer pictogrammes: {input}"
            ),
            DataConfig(
                name="hybrid_to_sentence",
                path="hybrid_to_sentence", 
                description="Hybrid tokens + keywords to French sentences",
                task_prefix_template="Transformer texte mixte: {input}"
            ),
            DataConfig(
                name="direct_to_sentence",
                path="direct_to_sentence",
                description="Direct pictogram text to French sentences", 
                task_prefix_template="Corriger texte: {input}"
            )
        ]
    

    
    @classmethod
    def get_config_by_name(cls, name: str) -> Optional[DataConfig]:



        configs = cls.get_all_configs()
        return next((c for c in configs if c.name == name), None)
    
    @classmethod

    def get_config_names(cls) -> List[str]:
        return [c.name for c in cls.get_all_configs()]


class DataLoader:
    
    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or Path("data/processed_propicto")
        self.logger = logging.getLogger(__name__)
    

    def load_datasets(self, data_config: DataConfig, max_train: int = None, 
       max_val: int = None, max_test: int = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        

        """
        
        Args
            data_config, max_train, max_val, max_test
            
        Out
            f (train_data, valid_data, test_data)
        """


        self.logger.info(f"Loading dataset: {data_config.name}")
        
        config_path = self.data_root / data_config.path
        
        datasets = {}
        for split in ['train', 'valid', 'test']:
            split_path = config_path / split / "data.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing {split} data: {split_path}")
            
            datasets[split] = safe_json_load(split_path)
        
        train_data = datasets['train'][:max_train] if max_train else datasets['train']
        valid_data = datasets['valid'][:max_val] if max_val else datasets['valid']
        test_data = datasets['test'][:max_test] if max_test else datasets['test']
        


        self.logger.info(f"   Loaded: {len(train_data)} train, {len(valid_data)} val, {len(test_data)} test")
        
        return train_data, valid_data, test_data
    
    def prepare_datasets(self, train_data: List[Dict], valid_data: List[Dict], test_data: List[Dict], tokenizer, data_config: DataConfig):
        
        
        def get_task_input(input_text: str) -> str:
            input_text = safe_text_encode(input_text)
            
            cleaned_input = input_text.replace('mots:', '').replace('tokens:', '').replace('hybrid:', '').replace('direct:', '').strip()
            
            return data_config.task_prefix_template.format(input=cleaned_input)
        
        def tokenize_function(examples):
            inputs = []
            targets = []
            
            for input_text, target_text in zip(examples['input_text'], examples['target_text']):
                input_text = safe_text_encode(input_text)
                target_text = safe_text_encode(target_text)
                
                formatted_input = get_task_input(input_text)
                inputs.append(formatted_input)
                targets.append(target_text)
            

            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors=None)
            labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length', return_tensors=None)
            
            model_inputs["labels"] = [[t if t != tokenizer.pad_token_id else -100 for t in label_ids] for  label_ids in labels["input_ids"]]
            
            if 'pictogram_sequence' in examples:
                model_inputs["pictogram_sequence"] = examples['pictogram_sequence']
            
            return model_inputs
        
        datasets = {}
        for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            if data:
                dataset = Dataset.from_list(data)
                
                tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[col for col in dataset.column_names if col not in ['pictogram_sequence']])
                
                tokenized_dataset._original_data = data
                
                datasets[name] = tokenized_dataset
        
        return datasets.get('train'), datasets.get('valid'), datasets.get('test')
    
class ProPictoDataProcessor:
    """
    Wrapper around the original ProPicto processor for compatibility

    """
    
    def __init__(self, source_dir: str = "data/raw/propicto_source", 
arasaac_metadata_path: str = "data/raw/metadata/arasaac_metadata.json"):

        try:
            from propicto_processor import ProPictoProcessor as OriginalProcessor
            self.processor = OriginalProcessor(source_dir, arasaac_metadata_path)
        except ImportError:
            self.processor = None
            logging.warning("ProPicto processor not available. Raw data processing disabled.")
    
    def process_raw_data(self, output_dir: str = "data/processed_propicto"):
        if not self.processor:
            raise RuntimeError("ProPicto processor not available")
        
        raw_data = self.processor.load_all_propicto_files()
        analysis = self.processor.analyze_data_quality(raw_data)
        
        filtered_data = self.processor.filter_data(
            raw_data,min_seq_length=3,
            max_seq_length=15,
            min_sentence_words=4,
            max_sentence_words=30,
            min_keyword_coverage=0.6)
        
        datasets = self.processor.create_training_datasets(filtered_data)
        split_datasets = self.processor.create_train_val_test_splits(datasets)
        self.processor.save_processed_datasets(split_datasets, output_dir)
        
        return split_datasets, analysis


def create_data_loader(data_root: Optional[Path] = None) -> DataLoader:
    return DataLoader(data_root)


def get_available_data_configs() -> List[str]:
    return DataRegistry.get_config_names()


def load_data(config_name: str, data_root: Optional[Path] = None, max_train: int = None, max_val: int = None, max_test: int = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    
    
    data_config = DataRegistry.get_config_by_name(config_name)
    if not data_config:
        available = DataRegistry.get_config_names()
        raise ValueError(f"Data config '{config_name}' not found. Available configs: {available}")
    
    loader = DataLoader(data_root)
    return loader.load_datasets(data_config, max_train, max_val, max_test)


DataConfig = DataConfig
