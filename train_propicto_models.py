#!/usr/bin/env python3
"""
CRITICAL FIXES for ProPicto Training Script
- Fixed tokenization for T5/MT5 models 
- Fixed label preparation to prevent NaN losses
- Added proper input/target prefixes
- Enhanced error handling for CUDA issues
"""

import logging
import argparse
import json
import torch
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    MT5ForConditionalGeneration,
    BartForConditionalGeneration, BartTokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, TrainerCallback
)
from datasets import Dataset
from tqdm import tqdm

@dataclass
class RobustProPictoConfig:
    """Enhanced configuration with validation and monitoring"""
    name: str
    model_architecture: str
    training_approach: str
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 8
    num_epochs: int = 2
    max_length: int = 128
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Monitoring settings
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    
    # Validation settings
    validate_data: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # Monitoring settings
    generate_samples_during_training: bool = True
    sample_generation_steps: int = 500
    num_sample_generations: int = 3
    
    def __post_init__(self):
        # Adjust parameters based on architecture
        if 'base' in self.model_architecture or 'large' in self.model_architecture:
            self.batch_size = max(2, self.batch_size // 2)
            self.learning_rate = 1e-4
            self.gradient_accumulation_steps = 2

class EnhancedDataValidator:
    """Enhanced data validator with better sanitization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_and_clean_dataset(self, data_path: str, split_name: str) -> Tuple[List[Dict], Dict]:
        """Validate and clean dataset with detailed reporting"""
        self.logger.info(f"🔍 Validating and cleaning {split_name} data: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        validation_results = {
            'total_examples': len(raw_data),
            'valid_examples': 0,
            'cleaned_examples': 0,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'cleaning_actions': defaultdict(int)
        }
        
        cleaned_data = []
        input_lengths = []
        target_lengths = []
        
        for i, example in enumerate(raw_data):
            try:
                cleaned_example = self._validate_and_clean_example(example, i, validation_results)
                if cleaned_example:
                    cleaned_data.append(cleaned_example)
                    
                    input_words = len(cleaned_example['input_text'].split())
                    target_words = len(cleaned_example['target_text'].split())
                    
                    input_lengths.append(input_words)
                    target_lengths.append(target_words)
                    
                    validation_results['valid_examples'] += 1
                    
            except Exception as e:
                validation_results['errors'].append(f"Example {i}: Processing error: {str(e)}")
        
        # Calculate statistics
        if input_lengths:
            validation_results['statistics'] = {
                'input_length_stats': {
                    'mean': np.mean(input_lengths),
                    'std': np.std(input_lengths),
                    'min': min(input_lengths),
                    'max': max(input_lengths),
                    'median': np.median(input_lengths)
                },
                'target_length_stats': {
                    'mean': np.mean(target_lengths),
                    'std': np.std(target_lengths),
                    'min': min(target_lengths),
                    'max': max(target_lengths),
                    'median': np.median(target_lengths)
                }
            }
        
        # Log detailed results
        self._log_validation_results(validation_results, split_name)
        
        return cleaned_data, validation_results
    
    def _validate_and_clean_example(self, example: Dict, index: int, validation_results: Dict) -> Optional[Dict]:
        """Validate and clean a single example"""
        
        # Check required fields
        required_fields = ['input_text', 'target_text']
        for field in required_fields:
            if field not in example:
                validation_results['errors'].append(f"Example {index}: Missing field '{field}'")
                return None
        
        # Extract and clean text
        input_text = self._clean_text(example['input_text'])
        target_text = self._clean_text(example['target_text'])
        
        # Validate after cleaning
        if not input_text or len(input_text.strip()) == 0:
            validation_results['errors'].append(f"Example {index}: Empty input after cleaning")
            return None
        
        if not target_text or len(target_text.strip()) == 0:
            validation_results['errors'].append(f"Example {index}: Empty target after cleaning")
            return None
        
        # Length validation
        input_words = len(input_text.split())
        target_words = len(target_text.split())
        
        if input_words > 100:
            validation_results['warnings'].append(f"Example {index}: Very long input ({input_words} words)")
        
        if target_words > 100:
            validation_results['warnings'].append(f"Example {index}: Very long target ({target_words} words)")
        
        if target_words < 1:
            validation_results['errors'].append(f"Example {index}: Target too short ({target_words} words)")
            return None
        
        # Create cleaned example
        cleaned_example = {
            'input_text': input_text,
            'target_text': target_text,
        }
        
        # Preserve other fields
        for key, value in example.items():
            if key not in ['input_text', 'target_text']:
                cleaned_example[key] = value
        
        validation_results['cleaned_examples'] += 1
        return cleaned_example
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common encoding issues for French
        text = text.replace("  ", "é")
        text = text.replace("   ", "é")
        text = text.replace("  ", "è")
        text = text.replace("â ", "à ")
        
        # Remove problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        
        # French-specific cleaning
        text = text.replace("'", "'")
        text = text.replace(""", '"').replace(""", '"')
        
        return text.strip()
    
    def _log_validation_results(self, results: Dict, split_name: str):
        """Log detailed validation results"""
        self.logger.info(f"✅ {split_name} validation complete:")
        self.logger.info(f"   Total examples: {results['total_examples']}")
        self.logger.info(f"   Valid examples: {results['valid_examples']}")
        self.logger.info(f"   Success rate: {results['valid_examples']/results['total_examples']*100:.1f}%")
        self.logger.info(f"   Errors: {len(results['errors'])}")
        self.logger.info(f"   Warnings: {len(results['warnings'])}")
        
        if results['statistics']:
            stats = results['statistics']
            self.logger.info(f"   Input length: {stats['input_length_stats']['mean']:.1f} ± {stats['input_length_stats']['std']:.1f} words")
            self.logger.info(f"   Target length: {stats['target_length_stats']['mean']:.1f} ± {stats['target_length_stats']['std']:.1f} words")
        
        # Show first few errors/warnings
        if results['errors']:
            self.logger.error("❌ Sample errors:")
            for error in results['errors'][:3]:
                self.logger.error(f"   {error}")
            if len(results['errors']) > 3:
                self.logger.error(f"   ... and {len(results['errors']) - 3} more errors")
        
        if results['warnings']:
            self.logger.warning("⚠️  Sample warnings:")
            for warning in results['warnings'][:3]:
                self.logger.warning(f"   {warning}")
            if len(results['warnings']) > 3:
                self.logger.warning(f"   ... and {len(results['warnings']) - 3} more warnings")
    
    def show_data_samples(self, data_path: str, num_samples: int = 3):
        """Show sample data for inspection"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"📋 Sample data from {Path(data_path).name}:")
        
        for i, example in enumerate(data[:num_samples]):
            self.logger.info(f"\nSample {i+1}:")
            self.logger.info(f"  Input:  {example.get('input_text', 'N/A')}")
            self.logger.info(f"  Target: {example.get('target_text', 'N/A')}")
            if 'id' in example:
                self.logger.info(f"  ID:     {example['id']}")

class MultilingualProPictoTrainer:
    """Enhanced trainer with FIXED tokenization and loss handling"""
    
    def __init__(self, data_dir: str = "data/processed_propicto"):
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.validator = EnhancedDataValidator()
        
        # Multilingual model mappings
        self.multilingual_models = {
            'mt5-small': 'google/mt5-small',
            'mt5-base': 'google/mt5-base',
            'mt5-large': 'google/mt5-large',
            'mbart-large': 'facebook/mbart-large-50-many-to-many-mmt',
            'mbart-large-cc25': 'facebook/mbart-large-cc25',
            't5-small': 't5-small',
            't5-base': 't5-base',
            'bart-base': 'facebook/bart-base'
        }
        
        self.logger.info(f"🚀 MultilingualProPictoTrainer initialized")
        self.logger.info(f"📱 Device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def validate_training_setup(self, config: RobustProPictoConfig) -> bool:
        """Validate complete training setup"""
        self.logger.info(f"🔧 Validating training setup for {config.name}")
        
        # Check data paths
        approach_dir = self.data_dir / config.training_approach
        train_path = approach_dir / "train" / "data.json"
        valid_path = approach_dir / "valid" / "data.json"
        
        if not train_path.exists():
            self.logger.error(f"❌ Training data not found: {train_path}")
            return False
        
        if not valid_path.exists():
            self.logger.error(f"❌ Validation data not found: {valid_path}")
            return False
        
        # Validate data
        if config.validate_data:
            train_validation = self.validator.validate_and_clean_dataset(str(train_path), "train")
            valid_validation = self.validator.validate_and_clean_dataset(str(valid_path), "valid")
            
            if train_validation[1]['errors'] or valid_validation[1]['errors']:
                self.logger.error("❌ Data validation failed")
                return False
        
        # Show sample data
        self.validator.show_data_samples(str(train_path), 2)
        
        # Check model architecture
        try:
            self._load_model_and_tokenizer(config.model_architecture)
            self.logger.info(f"✅ Model architecture {config.model_architecture} available")
        except Exception as e:
            self.logger.error(f"❌ Model architecture validation failed: {e}")
            return False
        
        return True
    
    def _load_model_and_tokenizer(self, architecture: str):
        """Load multilingual or standard models"""
        model_name = self.multilingual_models.get(architecture, architecture)
        
        self.logger.info(f"🌍 Loading model: {model_name}")
        
        if 'mt5' in model_name:
            model = MT5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            
        elif 'mbart' in model_name:
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            tokenizer.src_lang = "en_XX"
            tokenizer.tgt_lang = "fr_XX"
            
        elif 't5' in model_name:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            
        elif 'bart' in model_name:
            model = BartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name)
            
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(self.device)
        return model, tokenizer
    
    def prepare_dataset(self, data_path: str, tokenizer, config: RobustProPictoConfig) -> Dataset:
        """FIXED: Enhanced dataset preparation with proper T5/MT5 formatting"""
        
        # Validate and clean data
        cleaned_data, validation_results = self.validator.validate_and_clean_dataset(
            data_path, Path(data_path).parent.name
        )
        
        if not cleaned_data:
            raise ValueError(f"No valid examples found in {data_path}")
        
        self.logger.info(f"📊 Using {len(cleaned_data)} cleaned examples")
        
        # Limit samples for testing
        if config.max_train_samples and 'train' in str(data_path):
            cleaned_data = cleaned_data[:config.max_train_samples]
            self.logger.info(f"🧪 Limited to {len(cleaned_data)} training samples")
        elif config.max_eval_samples and ('valid' in str(data_path) or 'test' in str(data_path)):
            cleaned_data = cleaned_data[:config.max_eval_samples]
            self.logger.info(f"🧪 Limited to {len(cleaned_data)} evaluation samples")
        
        def tokenize_function(examples):
            """FIXED: Proper tokenization with prefixes for T5/MT5 models"""
            
            # Check if we're dealing with T5/MT5 models
            is_t5_family = any(name in tokenizer.name_or_path.lower() for name in ['t5', 'mt5'])
            
            if is_t5_family:
                # T5/MT5 models need task prefixes
                # Add "translate French: " prefix for input
                prefixed_inputs = ["translate French: " + text for text in examples["input_text"]]
                
                # Tokenize inputs with prefix
                model_inputs = tokenizer(
                    prefixed_inputs,
                    max_length=config.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors=None  # Return lists, not tensors
                )
                
                # Tokenize targets (no prefix needed)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        examples["target_text"],
                        max_length=config.max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors=None
                    )
                
            elif 'mbart' in tokenizer.name_or_path:
                # mBART handling
                model_inputs = tokenizer(
                    examples["input_text"],
                    max_length=config.max_length,
                    truncation=True,
                    padding='max_length'
                )
                
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        examples["target_text"],
                        max_length=config.max_length,
                        truncation=True,
                        padding='max_length'
                    )
            else:
                # BART and other models
                model_inputs = tokenizer(
                    examples["input_text"],
                    max_length=config.max_length,
                    truncation=True,
                    padding='max_length'
                )
                
                labels = tokenizer(
                    examples["target_text"],
                    max_length=config.max_length,
                    truncation=True,
                    padding='max_length'
                )
            
            # CRITICAL FIX: Properly handle labels
            model_inputs["labels"] = []
            for label_ids in labels["input_ids"]:
                # Convert to list if tensor
                if hasattr(label_ids, 'tolist'):
                    label_ids = label_ids.tolist()
                
                # Replace padding tokens with -100 (ignore index)
                label_ids = [
                    token_id if token_id != tokenizer.pad_token_id else -100 
                    for token_id in label_ids
                ]
                model_inputs["labels"].append(label_ids)
            
            return model_inputs
        
        dataset = Dataset.from_list(cleaned_data)
        
        # Show tokenization examples
        self._show_tokenization_examples(cleaned_data[:2], tokenizer, config)
        
        # Use appropriate number of processes
        num_proc = min(4, os.cpu_count() or 1)
        dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)
        
        self.logger.info(f"✅ Dataset prepared: {len(dataset)} examples")
        return dataset
    
    def _show_tokenization_examples(self, examples: List[Dict], tokenizer, config: RobustProPictoConfig):
        """Show detailed tokenization examples with proper formatting"""
        self.logger.info(f"🔍 Tokenization examples:")
        
        is_t5_family = any(name in tokenizer.name_or_path.lower() for name in ['t5', 'mt5'])
        
        for i, example in enumerate(examples):
            input_text = example['input_text']
            target_text = example['target_text']
            
            # Apply same preprocessing as in tokenize_function
            if is_t5_family:
                prefixed_input = "translate French: " + input_text
                input_encoding = tokenizer(prefixed_input, max_length=config.max_length, truncation=True, padding='max_length')
            else:
                prefixed_input = input_text
                input_encoding = tokenizer(input_text, max_length=config.max_length, truncation=True, padding='max_length')
            
            target_encoding = tokenizer(target_text, max_length=config.max_length, truncation=True, padding='max_length')
            
            self.logger.info(f"\nExample {i+1}:")
            self.logger.info(f"  Raw input:     '{input_text}'")
            self.logger.info(f"  Prefixed input: '{prefixed_input}'")
            self.logger.info(f"  Raw target:    '{target_text}'")
            self.logger.info(f"  Input tokens ({len(input_encoding['input_ids'])}): {input_encoding['input_ids'][:10]}...")
            self.logger.info(f"  Target tokens ({len(target_encoding['input_ids'])}): {target_encoding['input_ids'][:10]}...")
            
            # Show how labels will be processed
            labels_processed = [
                token_id if token_id != tokenizer.pad_token_id else -100 
                for token_id in target_encoding['input_ids']
            ]
            non_ignore_labels = [l for l in labels_processed if l != -100]
            self.logger.info(f"  Labels (non-ignore): {len(non_ignore_labels)} tokens")
    
    def train_model(self, config: RobustProPictoConfig) -> Optional[str]:
        """Enhanced training with FIXED loss handling"""
        
        # Validate setup
        if not self.validate_training_setup(config):
            return None
        
        self.logger.info(f"🌍 Starting multilingual training: {config.name}")
        self.logger.info(f"📋 Configuration:")
        self.logger.info(f"   Architecture: {config.model_architecture}")
        self.logger.info(f"   Approach: {config.training_approach}")
        self.logger.info(f"   Learning rate: {config.learning_rate}")
        self.logger.info(f"   Batch size: {config.batch_size}")
        self.logger.info(f"   Epochs: {config.num_epochs}")
        
        # Data paths
        approach_dir = self.data_dir / config.training_approach
        train_path = approach_dir / "train" / "data.json"
        valid_path = approach_dir / "valid" / "data.json"
        test_path = approach_dir / "test" / "data.json"
        
        # Load model and tokenizer
        model, tokenizer = self._load_model_and_tokenizer(config.model_architecture)
        
        # Prepare datasets with FIXED tokenization
        train_dataset = self.prepare_dataset(str(train_path), tokenizer, config)
        valid_dataset = self.prepare_dataset(str(valid_path), tokenizer, config)
        
        # Also prepare test dataset for final evaluation
        test_dataset = None
        if test_path.exists():
            test_dataset = self.prepare_dataset(str(test_path), tokenizer, config)
            self.logger.info(f"📊 Test dataset prepared: {len(test_dataset)} examples")
        
        # Output directory
        output_dir = f"models/propicto_experiments/{config.name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate training steps
        steps_per_epoch = max(1, len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps))
        total_steps = steps_per_epoch * config.num_epochs
        
        self.logger.info(f"📊 Training plan:")
        self.logger.info(f"   Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"   Total steps: {total_steps}")
        self.logger.info(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        
        # FIXED: Training arguments with proper loss handling
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=3,
            fp16=False,  # DISABLED: FP16 can cause NaN issues
            dataloader_pin_memory=False,
            report_to=[],
            logging_dir=f"{output_dir}/logs",
            run_name=config.name,
            disable_tqdm=False,
            # Additional stability settings
            max_grad_norm=1.0,  # Gradient clipping
            dataloader_drop_last=True,  # Avoid uneven batches
            remove_unused_columns=True,
            label_smoothing_factor=0.0  # No label smoothing to avoid issues
        )
        
        # FIXED: Enhanced data collator with proper handling
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt",
            label_pad_token_id=-100  # Explicit ignore index
        )
        
        # Trainer with error handling
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        try:
            # SAFETY: Clear CUDA cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Train with error handling
            self.logger.info("🚀 Starting FIXED training with proper loss handling...")
            start_time = time.time()
            
            trainer.train()
            training_time = time.time() - start_time
            
            # Final evaluation on test set
            test_results = None
            if test_dataset:
                self.logger.info("🧪 Running final test evaluation...")
                try:
                    test_results = trainer.evaluate(test_dataset)
                    self.logger.info(f"📊 Test results: {test_results}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Test evaluation failed: {e}")
            
            # Save final model
            final_output_dir = f"models/propicto_final/{config.name}"
            Path(final_output_dir).mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            
            # Test generation with SAFE parameters
            self._test_generation_safely(model, tokenizer, train_dataset, final_output_dir)
            
            self.logger.info(f"✅ Training completed successfully!")
            self.logger.info(f"⏱️  Training time: {training_time / 3600:.2f} hours")
            self.logger.info(f"📁 Model saved to: {final_output_dir}")
            
            return final_output_dir
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            
            # Additional debugging info
            if "nan" in str(e).lower() or "cuda" in str(e).lower():
                self.logger.error("🔍 Debugging suggestions:")
                self.logger.error("   1. Try with fp16=False (already set)")
                self.logger.error("   2. Reduce learning rate")
                self.logger.error("   3. Check tokenization")
                self.logger.error("   4. Verify label preparation")
            
            raise
    
    def _test_generation_safely(self, model, tokenizer, dataset, output_dir: str):
        """Test generation with safe parameters to avoid CUDA errors"""
        self.logger.info("🧪 Testing generation safely...")
        
        model.eval()
        sample_results = []
        
        try:
            # Get a few samples
            for i in range(min(3, len(dataset))):
                example = dataset[i]
                
                # Prepare input safely
                input_ids = torch.tensor([example['input_ids']]).to(model.device)
                attention_mask = torch.tensor([example['attention_mask']]).to(model.device)
                
                # Generate with SAFE parameters
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=50,  # Shorter to avoid issues
                            num_beams=1,    # Greedy search only
                            do_sample=False,  # No sampling
                            early_stopping=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Get reference
                        label_ids = [l for l in example['labels'] if l != -100]
                        reference = tokenizer.decode(label_ids, skip_special_tokens=True)
                        
                        sample_results.append({
                            'input': tokenizer.decode(example['input_ids'], skip_special_tokens=True),
                            'prediction': prediction,
                            'reference': reference
                        })
                        
                        self.logger.info(f"Sample {i+1}:")
                        self.logger.info(f"  Input: {sample_results[-1]['input']}")
                        self.logger.info(f"  Prediction: {prediction}")
                        self.logger.info(f"  Reference: {reference}")
                        
                    except Exception as gen_e:
                        self.logger.warning(f"⚠️  Generation failed for sample {i+1}: {gen_e}")
                        sample_results.append({
                            'input': 'Error reading input',
                            'prediction': f'Generation failed: {str(gen_e)}',
                            'reference': 'Error reading reference'
                        })
        
        except Exception as e:
            self.logger.warning(f"⚠️  Safe generation test failed: {e}")
        
        # Save results
        try:
            with open(f"{output_dir}/generation_test.json", 'w', encoding='utf-8') as f:
                json.dump(sample_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save generation test: {e}")
        
        model.train()

# Test configurations with SAFER parameters
def create_safe_test_configs():
    """Create safer test configurations to avoid NaN/CUDA issues"""
    
    configs = []
    
    # Ultra safe test - minimal configuration
    ultra_safe = RobustProPictoConfig(
        name="ultra_safe_test",
        model_architecture="mt5-small",
        training_approach="keywords_to_sentence",
        num_epochs=1,
        max_train_samples=10,  # Even smaller
        max_eval_samples=3,    # Even smaller
        batch_size=1,          # Smallest possible
        gradient_accumulation_steps=1,
        logging_steps=2,
        eval_steps=5,
        save_steps=10,
        learning_rate=1e-4,    # Smaller learning rate
        warmup_steps=2,        # Minimal warmup
        generate_samples_during_training=False  # Disable to avoid issues
    )
    configs.append(ultra_safe)
    
    return configs

def create_quick_test_configs():
    """Create quick test configurations"""
    
    configs = []
    
    # Quick test with safer parameters
    quick_test = RobustProPictoConfig(
        name="quick_safe_test",
        model_architecture="mt5-small",
        training_approach="keywords_to_sentence",
        num_epochs=1,
        max_train_samples=20,
        max_eval_samples=5,
        batch_size=2,
        logging_steps=3,
        eval_steps=10,
        save_steps=20,
        learning_rate=3e-5,  # Safer learning rate
        generate_samples_during_training=False
    )
    configs.append(quick_test)
    
    return configs

def main():
    parser = argparse.ArgumentParser(description='FIXED ProPicto Training - NaN and CUDA Issues Resolved')
    parser.add_argument('--config-type', choices=['ultra-safe', 'quick', 'custom'], 
                       default='ultra-safe', help='Type of test configuration')
    parser.add_argument('--model-architecture', 
                       choices=['mt5-small', 'mt5-base', 't5-small', 't5-base'],
                       help='Model architecture to use')
    parser.add_argument('--training-approach',
                       choices=['keywords_to_sentence', 'pictos_tokens_to_sentence', 
                               'hybrid_to_sentence', 'direct_to_sentence'],
                       default='keywords_to_sentence',
                       help='Training approach')
    parser.add_argument('--test-name', help='Custom test name')
    parser.add_argument('--max-train-samples', type=int, help='Limit training samples')
    parser.add_argument('--max-eval-samples', type=int, help='Limit eval samples')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without training')
    parser.add_argument('--debug-cuda', action='store_true', help='Enable CUDA debugging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('propicto_training_fixed.log')
        ]
    )
    
    print("🛠️  FIXED ProPicto Training Framework")
    print("=" * 50)
    print("✅ Fixed NaN losses and CUDA errors")
    print("✅ Proper T5/MT5 tokenization with prefixes")
    print("✅ Enhanced error handling and debugging")
    
    # Enable CUDA debugging if requested
    if args.debug_cuda:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("🔍 CUDA debugging enabled (synchronous execution)")
    
    # Check data
    if not Path("data/processed_propicto").exists():
        print("❌ No processed ProPicto data found!")
        print("Run: python propicto_processor.py")
        return
    
    # Initialize trainer
    trainer = MultilingualProPictoTrainer()
    
    # Create configuration
    if args.model_architecture or args.test_name:
        # Custom configuration
        config = RobustProPictoConfig(
            name=args.test_name or f"custom_{args.model_architecture}_{args.training_approach}",
            model_architecture=args.model_architecture or "mt5-small",
            training_approach=args.training_approach,
            max_train_samples=args.max_train_samples or 10,
            max_eval_samples=args.max_eval_samples or 3,
            batch_size=args.batch_size or 1,
            num_epochs=args.num_epochs or 1,
            learning_rate=args.learning_rate or 1e-4,
            generate_samples_during_training=False  # Disable for safety
        )
        configs = [config]
    else:
        # Predefined configurations
        if args.config_type == 'ultra-safe':
            configs = create_safe_test_configs()
        elif args.config_type == 'quick':
            configs = create_quick_test_configs()
        else:
            configs = create_safe_test_configs()
    
    print(f"\n🎯 Running {len(configs)} SAFE configurations:")
    for config in configs:
        print(f"   📋 {config.name} ({config.model_architecture}, {config.training_approach})")
        print(f"      🧪 Samples: {config.max_train_samples} train, {config.max_eval_samples} eval")
        print(f"      ⚙️  Settings: batch_size={config.batch_size}, lr={config.learning_rate}")
    
    if args.dry_run:
        print("\n🧪 DRY RUN MODE - Validating setups only")
        for config in configs:
            print(f"\n🔍 Validating: {config.name}")
            success = trainer.validate_training_setup(config)
            print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        return
    
    # Run training with enhanced error handling
    successful = 0
    failed = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n🏋️ Training {i}/{len(configs)}: {config.name}")
        print("-" * 50)
        
        try:
            # Clear CUDA cache before each training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 Cleared CUDA cache")
            
            model_path = trainer.train_model(config)
            if model_path:
                successful += 1
                print(f"✅ Success: {config.name}")
                print(f"📁 Model saved: {model_path}")
            else:
                failed.append(config.name)
                print(f"❌ Failed: {config.name}")
                
        except Exception as e:
            failed.append(config.name)
            print(f"❌ Error in {config.name}: {e}")
            
            # Enhanced error diagnosis
            error_str = str(e).lower()
            if "nan" in error_str:
                print("🔍 NaN Error Detected:")
                print("   - Check input data for invalid values")
                print("   - Verify tokenization produces valid labels")
                print("   - Consider smaller learning rate")
            elif "cuda" in error_str or "assert" in error_str:
                print("🔍 CUDA Error Detected:")
                print("   - Run with --debug-cuda for detailed info")
                print("   - Check tokenizer pad_token configuration")
                print("   - Verify model and data device placement")
            
            # Continue with next config
            continue
    
    # Summary
    print(f"\n🎉 TRAINING COMPLETE")
    print("=" * 30)
    print(f"✅ Successful: {successful}/{len(configs)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Try with --debug-cuda for detailed error info")
        print("   2. Use ultra-safe config with minimal parameters")
        print("   3. Check GPU memory and clear cache")
        print("   4. Verify data format and tokenization")
    
    if successful > 0:
        print(f"\n📁 Models saved to: models/propicto_final/")
        print("🔍 Next steps:")
        print("   1. Test the trained models")
        print("   2. Check generation_test.json for sample outputs")
        print("   3. Review training logs for any warnings")

if __name__ == "__main__":
    main()