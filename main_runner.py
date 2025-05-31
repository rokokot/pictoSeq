#!/usr/bin/env python3
"""
PictoSeq experiment runner

This script runs comprehensive experiments on sequence-to-sequence models for pictogram-to-French text generation.
It is designed to work with processed ProPicto data and includes fixes for GenerationConfig validation issues.

"""

import logging
import torch
import json
import time
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import math
import pickle
import itertools
from dataclasses import dataclass, field

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer,TrainingArguments, Trainer, DataCollatorForSeq2Seq,EarlyStoppingCallback, TrainerCallback, GenerationConfig)
from datasets import Dataset

# Check for jiwer library for WER calculation
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("Warning: jiwer not available. Using fallback WER calculation.")

def safe_json_dump(obj, fp, **kwargs):
    """Save JSON data with UTF-8 encoding."""
    kwargs.setdefault('ensure_ascii', False)
    kwargs.setdefault('indent', 2)
    
    if hasattr(fp, 'write'):
        json.dump(obj, fp, **kwargs)
    else:
        with open(fp, 'w', encoding='utf-8', newline='') as f:
            json.dump(obj, f, **kwargs)

def safe_json_load(fp):
    """Load JSON data with UTF-8 encoding."""
    if hasattr(fp, 'read'):
        return json.load(fp)
    else:
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)

def safe_text_encode(text: str) -> str:
    """Ensure text is properly encoded as UTF-8 string."""
    if not isinstance(text, str):
        return str(text)
    return text

class HPCEnvironment:
    """Manage file paths and storage for HPC cluster environments like VSC."""
    
    def __init__(self, results_base_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Check for VSC environment variables
        self.vsc_data = os.environ.get('VSC_DATA')
        self.vsc_scratch = os.environ.get('VSC_SCRATCH')
        self.slurm_job_id = os.environ.get('SLURM_JOB_ID')
        self.slurm_node = os.environ.get('SLURMD_NODENAME')
        
        # Set up results directory - prefer scratch storage for temporary results
        if results_base_path:
            self.results_base = Path(results_base_path)
        elif self.vsc_scratch:
            self.results_base = Path(self.vsc_scratch) / "pictoSeq_results"
        else:
            self.results_base = Path("pictoseq")
        
        # Set up working directory - prefer data storage for code and datasets
        if self.vsc_data:
            self.working_dir = Path(self.vsc_data) / "pictoSeq"
        else:
            self.working_dir = Path.cwd()
        
        self.logger.info(f"Working directory: {self.working_dir}")
        self.logger.info(f"Results base: {self.results_base}")
        if self.slurm_job_id:
            self.logger.info(f"SLURM Job ID: {self.slurm_job_id}")
    
    def get_results_dir(self, experiment_name: str) -> Path:
        """Create a timestamped results directory for an experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.slurm_job_id:
            dir_name = f"{experiment_name}_{timestamp}_job{self.slurm_job_id}"
        else:
            dir_name = f"{experiment_name}_{timestamp}"
        
        results_dir = self.results_base / dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def get_data_path(self) -> Path:
        """Get the path to processed ProPicto data directory."""
        return self.working_dir / "data" / "processed_propicto"
    
    def ensure_results_structure(self, results_dir: Path):
        """Create standard subdirectories for experiment results."""
        subdirs = ["logs", "visualizations", "analysis", "summaries"]
        for subdir in subdirs:
            (results_dir / subdir).mkdir(exist_ok=True)

@dataclass
class DecodingConfig:
    """Configuration for different text generation decoding strategies."""
    name: str
    strategy_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

class DecodingStrategies:
    """Handle different text generation strategies with proper configuration validation."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def get_all_strategies(cls) -> List[DecodingConfig]:
        """Return all available decoding strategies with proper parameter validation."""
        return [
            DecodingConfig(
                name="greedy",
                strategy_type="deterministic",
                params={
                    "num_beams": 1, 
                    "early_stopping": False,  # Must be False when num_beams=1
                    "do_sample": False
                },
                description="Greedy search - always pick most likely token"
            ),
            DecodingConfig(
                name="beam_search",
                strategy_type="search",
                params={
                    "num_beams": 4,
                    "length_penalty": 1.2,
                    "early_stopping": True,  # OK when num_beams > 1
                    "no_repeat_ngram_size": 2,
                    "do_sample": False
                },
                description="Beam search with length penalty and repetition control"
            ),
            DecodingConfig(
                name="nucleus_sampling",
                strategy_type="sampling",
                params={
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.8,
                    "no_repeat_ngram_size": 2,
                    "num_beams": 1,
                    "early_stopping": False  # Must be False when num_beams=1
                },
                description="Nucleus (top-p) sampling with temperature"
            )
        ]
    
    def _create_safe_generation_config(self, strategy: DecodingConfig, max_length: int = 128) -> GenerationConfig:
        """Create a GenerationConfig that passes validation checks."""
        
        # Start with minimal safe parameters - do NOT inherit from model's config
        # because it might have conflicting settings
        safe_params = {
            "max_length": max_length,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add strategy parameters with validation
        strategy_params = strategy.params.copy()
        
        # Critical fix: ensure early_stopping is compatible with num_beams
        num_beams = strategy_params.get("num_beams", 1)
        early_stopping = strategy_params.get("early_stopping", False)
        
        if num_beams == 1 and early_stopping:
            self.logger.warning(f"Fixing early_stopping=True with num_beams=1 for strategy {strategy.name}")
            strategy_params["early_stopping"] = False
        
        # Combine parameters - strategy params override safe defaults
        final_params = {**safe_params, **strategy_params}
        
        try:
            gen_config = GenerationConfig(**final_params)
            gen_config.validate()
            return gen_config
        except Exception as e:
            self.logger.warning(f"GenerationConfig validation failed for {strategy.name}: {e}")
            
            # Fallback to minimal safe configuration
            fallback_params = {
                "max_length": max_length,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_beams": num_beams,
                "do_sample": strategy_params.get("do_sample", False),
                "early_stopping": False  # Always False in fallback
            }
            
            return GenerationConfig(**fallback_params)
    
    def generate_with_strategy(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                             strategy: DecodingConfig, max_length: int = 128) -> Tuple[str, float, Dict]:
        """Generate text using the specified decoding strategy."""
        start_time = time.time()

        try:
            # Create validated generation configuration
            gen_config = self._create_safe_generation_config(strategy, max_length)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config
                )

            generation_time = time.time() - start_time
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = safe_text_encode(generated_text)

            metadata = {
                "strategy": strategy.name,
                "generation_time": generation_time,
                "output_length": len(outputs[0]),
                "success": True
            }

            return generated_text, generation_time, metadata

        except Exception as e:
            self.logger.error(f"Generation failed for strategy {strategy.name}: {e}")
            return "[GENERATION ERROR]", 0.0, {
                "strategy": strategy.name,
                "generation_time": 0.0,
                "output_length": 0,
                "success": False,
                "error": str(e)
            }
    
    def generate_all_strategies(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                              max_length: int = 128) -> Dict[str, Tuple[str, float, Dict]]:
        """Generate text using all available decoding strategies."""
        results = {}
        strategies = self.get_all_strategies()
        
        for strategy in strategies:
            results[strategy.name] = self.generate_with_strategy(input_ids, attention_mask, strategy, max_length)
        
        return results

class Evaluator:
    """Evaluate model predictions using multiple text generation metrics."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        self.has_jiwer = HAS_JIWER
        
        if not self.has_jiwer:
            self.logger.warning("jiwer not available, using fallback WER calculation")
    
    def evaluate_predictions(self, predictions: List[str], references: List[str], strategy_name: str = "unknown") -> Dict[str, float]:
        """Evaluate predictions against references using multiple metrics."""
        if not predictions or not references:
            return {}
        
        predictions = [safe_text_encode(p) for p in predictions]
        references = [safe_text_encode(r) for r in references]
        
        results = {"strategy": strategy_name}
        
        # Core NLP evaluation metrics
        results['bleu'] = self._calculate_bleu(predictions, references)
        results['rouge_l'] = self._calculate_rouge_l(predictions, references)
        
        # Word Error Rate calculation
        if self.has_jiwer:
            results['wer'] = self._calculate_wer(predictions, references)
        else:
            results['wer'] = self._calculate_wer_simple(predictions, references)
        
        # Additional quality metrics
        results.update(self._calculate_length_metrics(predictions, references))
        results.update(self._calculate_lexical_metrics(predictions, references))
        results.update(self._calculate_french_linguistic_metrics(predictions, references))
        results.update(self._calculate_quality_metrics(predictions, references))
        
        return results
    
    def _calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate using jiwer library."""
        try:
            # Filter out empty predictions and references
            valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                          if p.strip() and r.strip()]
            
            if not valid_pairs:
                return 1.0
            
            valid_predictions, valid_references = zip(*valid_pairs)
            wer_score = jiwer.wer(list(valid_references), list(valid_predictions))
            return float(wer_score)
            
        except Exception as e:
            self.logger.warning(f"WER calculation failed: {e}")
            return self._calculate_wer_simple(predictions, references)
    
    def _calculate_wer_simple(self, predictions: List[str], references: List[str]) -> float:
        """Simple Word Error Rate calculation using edit distance."""
        def edit_distance(s1_words, s2_words):
            """Calculate minimum edit distance between two word sequences."""
            m, n = len(s1_words), len(s2_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1_words[i-1] == s2_words[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
        
        total_errors = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if not ref_words:
                continue
            
            errors = edit_distance(pred_words, ref_words)
            total_errors += errors
            total_words += len(ref_words)
        
        return total_errors / total_words if total_words > 0 else 1.0
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score using n-gram overlap."""
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        total_score = 0
        valid_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                continue
            
            scores = []
            for n in range(1, 5):
                pred_ngrams = Counter(get_ngrams(pred_tokens, n))
                ref_ngrams = Counter(get_ngrams(ref_tokens, n))
                
                overlap = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())
                
                scores.append(overlap / total if total > 0 else 0.0)
            
            if all(s > 0 for s in scores):
                bleu = math.exp(sum(math.log(s) for s in scores) / len(scores))
                bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
                total_score += bp * bleu
                valid_count += 1
        
        return total_score / valid_count if valid_count > 0 else 0.0
    
    def _calculate_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """Calculate ROUGE-L F1 score using longest common subsequence."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(ref_tokens) == 0:
                continue
            
            lcs_len = lcs_length(pred_tokens, ref_tokens)
            
            precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = lcs_len / len(ref_tokens)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                rouge_scores.append(f1)
        
        return np.mean(rouge_scores) if rouge_scores else 0.0
    
    def _calculate_length_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate length-based metrics for generated text."""
        pred_lengths = [len(p.split()) for p in predictions if p.strip()]
        ref_lengths = [len(r.split()) for r in references if r.strip()]
        
        if not pred_lengths or not ref_lengths:
            return {}
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
            'length_std_pred': np.std(pred_lengths),
            'median_pred_length': np.median(pred_lengths)
        }
    
    def _calculate_lexical_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate lexical diversity and vocabulary overlap metrics."""
        metrics = {}
        
        # Calculate vocabulary overlap between predictions and references
        vocab_overlaps = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                overlap = len(pred_words & ref_words) / len(ref_words)
                vocab_overlaps.append(overlap)
        
        metrics['vocab_overlap'] = np.mean(vocab_overlaps) if vocab_overlaps else 0.0
        
        # Calculate type-token ratio (lexical diversity)
        ttr_scores = []
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                ttr = len(set(words)) / len(words)
                ttr_scores.append(ttr)
        
        metrics['type_token_ratio'] = np.mean(ttr_scores) if ttr_scores else 0.0
        
        return metrics
    
    def _calculate_french_linguistic_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate French-specific linguistic quality metrics."""
        french_articles = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de'}
        french_pronouns = {'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles'}
        
        article_ratios = []
        pronoun_ratios = []
        
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                article_count = len([w for w in words if w in french_articles])
                pronoun_count = len([w for w in words if w in french_pronouns])
                
                article_ratios.append(article_count / len(words))
                pronoun_ratios.append(pronoun_count / len(words))
        
        return {
            'french_article_ratio': np.mean(article_ratios) if article_ratios else 0.0,
            'french_pronoun_ratio': np.mean(pronoun_ratios) if pronoun_ratios else 0.0
        }
    
    def _calculate_quality_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate overall text quality assessment metrics."""
        # Generation success rate - how many predictions are valid
        valid_generations = 0
        for pred in predictions:
            if (len(pred.strip()) > 0 and 
                len(pred.split()) > 1 and 
                '<extra_id_0>' not in pred and
                len(pred.split()) < 50):
                valid_generations += 1
        
        generation_success_rate = valid_generations / len(predictions) if predictions else 0.0
        
        # Fluency score based on repetition (no excessive repetition indicates better fluency)
        fluency_scores = []
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                fluency_scores.append(unique_ratio)
        
        fluency_score = np.mean(fluency_scores) if fluency_scores else 0.0
        
        return {
            'generation_success_rate': generation_success_rate,
            'fluency_score': fluency_score
        }

@dataclass
class ModelConfig:
    """Configuration for different model architectures."""
    name: str
    model_id: str
    tokenizer_class: str
    description: str
    is_multilingual: bool = False

@dataclass
class DataConfig:
    """Configuration for different input data types."""
    name: str
    path: str
    description: str
    task_prefix_template: str

@dataclass
class ExperimentConfig:
    """Complete experimental configuration."""
    model_config: ModelConfig
    data_config: DataConfig
    experiment_id: str
    max_train_samples: int = 10000
    max_val_samples: int = 1000
    max_test_samples: int = 1000
    num_epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-5

class ExperimentalMatrix:
    """Define all experimental configurations for systematic comparison."""
    
    @classmethod
    def get_model_configs(cls) -> List[ModelConfig]:
        """Return all model configurations for evaluation."""
        return [
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
    def get_data_configs(cls) -> List[DataConfig]:
        """Return all data configurations for evaluation."""
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
    def generate_all_experiments(cls, test_run: bool = False) -> List[ExperimentConfig]:
        """Generate full experimental matrix: 3 models × 4 data configs = 12 experiments."""
        experiments = []
        
        models = cls.get_model_configs()
        data_configs = cls.get_data_configs()
        
        for model_config, data_config in itertools.product(models, data_configs):
            experiment_id = f"{model_config.name}_{data_config.name}"
            
            # Adjust parameters for test runs to validate pipeline quickly
            max_train = 1000 if test_run else 10000
            max_val = 200 if test_run else 1000
            max_test = 200 if test_run else 1000
            epochs = 2 if test_run else 5
            
            experiment = ExperimentConfig(
                model_config=model_config,
                data_config=data_config,
                experiment_id=experiment_id,
                max_train_samples=max_train,
                max_val_samples=max_val,
                max_test_samples=max_test,
                num_epochs=epochs
            )
            
            experiments.append(experiment)
        
        return experiments

class AcademicResearchCallback(TrainerCallback):
    """Academic research callback with comprehensive tracking and multi-strategy evaluation."""
    
    def __init__(self, evaluator, eval_dataset, output_dir, experiment_config, tokenizer, generation_samples=10):
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.experiment_config = experiment_config
        self.tokenizer = tokenizer
        self.generation_samples = generation_samples
        self.logger = logging.getLogger(__name__)
        
        # Track metrics and results throughout training
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.generation_history = []
        self.training_progress = []
        
        # Create subdirectories for organized storage
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "decoding_comparisons").mkdir(parents=True, exist_ok=True)
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        """Run comprehensive evaluation with all decoding strategies at regular intervals."""
        
        # Run multi-strategy evaluation every few evaluation steps to track progress
        if state.global_step % (args.eval_steps * 2) == 0:
            self.logger.info(f"Running comprehensive multi-strategy evaluation at step {state.global_step}")
            
            try:
                active_tokenizer = tokenizer if tokenizer is not None else self.tokenizer
                
                if active_tokenizer is None:
                    self.logger.warning("No tokenizer available for evaluation")
                    return
                
                # Initialize decoding strategies for this model
                decoder = DecodingStrategies(model, active_tokenizer, model.device)
                
                # Generate predictions with all strategies on a sample of validation data
                all_results = self._generate_multi_strategy_predictions(
                    model, active_tokenizer, decoder, self.eval_dataset,
                    num_samples=min(200, len(self.eval_dataset))
                )
                
                if not all_results:
                    self.logger.warning("No valid predictions generated")
                    return
                
                # Evaluate each strategy and track best performing approaches
                strategy_metrics = {}
                for strategy_name, (predictions, references, inputs, metadata) in all_results.items():
                    if predictions and references:
                        metrics = self.evaluator.evaluate_predictions(predictions, references, strategy_name)
                        strategy_metrics[strategy_name] = metrics
                        
                        # Log key results for monitoring
                        self.logger.info(f"{strategy_name.upper()} Results:")
                        for metric_name, value in metrics.items():
                            if metric_name in ['bleu', 'rouge_l', 'wer', 'vocab_overlap', 'generation_success_rate']:
                                self.logger.info(f"   {metric_name}: {value:.4f}")
                        
                        # Track metrics over time for analysis
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                full_metric_name = f"{strategy_name}_{metric_name}"
                                self.metrics_history[full_metric_name].append({
                                    'step': state.global_step,
                                    'epoch': state.epoch,
                                    'value': float(value)
                                })
                                
                                # Track best metrics (lower is better for WER, higher for others)
                                if (full_metric_name not in self.best_metrics or 
                                    (metric_name != 'wer' and value > self.best_metrics[full_metric_name]['value']) or
                                    (metric_name == 'wer' and value < self.best_metrics[full_metric_name]['value'])):
                                    self.best_metrics[full_metric_name] = {
                                        'value': float(value),
                                        'step': state.global_step,
                                        'epoch': state.epoch
                                    }
                
                # Save detailed samples for qualitative analysis
                self._save_multi_strategy_samples(all_results, state.global_step)
                
                # Save strategy comparison for later analysis
                comparison_file = self.output_dir / "decoding_comparisons" / f"comparison_step_{state.global_step}.json"
                safe_json_dump(strategy_metrics, comparison_file)
                
            except Exception as e:
                self.logger.error(f"Multi-strategy evaluation failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
    
    def _generate_multi_strategy_predictions(self, model, tokenizer, decoder, eval_dataset, num_samples=200):
        """Generate predictions with all decoding strategies and track pictogram sequences."""
        model.eval()
        
        # Sample a subset of validation data for evaluation
        sample_indices = np.random.choice(
            len(eval_dataset), 
            min(num_samples, len(eval_dataset)), 
            replace=False
        )
        sample_indices = [int(idx) for idx in sample_indices]
        
        # Initialize results storage for each strategy
        all_results = {}
        strategies = DecodingStrategies.get_all_strategies()
        
        for strategy in strategies:
            all_results[strategy.name] = ([], [], [], [])  # predictions, references, inputs, metadata
        
        successful_generations = 0
        
        with torch.no_grad():
            for idx in sample_indices:
                try:
                    example = eval_dataset[idx]
                    
                    # Prepare input tensors
                    input_ids = torch.tensor([example['input_ids']]).to(model.device)
                    attention_mask = torch.tensor([example['attention_mask']]).to(model.device)
                    
                    # Decode input and reference text
                    input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
                    input_text = safe_text_encode(input_text)
                    
                    label_ids = [l for l in example['labels'] if l != -100]
                    reference = tokenizer.decode(label_ids, skip_special_tokens=True)
                    reference = safe_text_encode(reference)
                    
                    # Extract pictogram sequence if available
                    picto_sequence = self._extract_pictogram_sequence_from_example(example, idx)
                    
                    # Generate with all strategies
                    strategy_outputs = decoder.generate_all_strategies(input_ids, attention_mask)
                    
                    # Store results for each strategy
                    for strategy_name, (prediction, gen_time, meta) in strategy_outputs.items():
                        if meta['success'] and prediction.strip() and reference.strip():
                            # Add pictogram sequence to metadata
                            enhanced_meta = {
                                **meta,
                                'pictogram_sequence': picto_sequence,
                                'sample_idx': idx,
                                'original_input': input_text
                            }
                            
                            all_results[strategy_name][0].append(prediction)  # predictions
                            all_results[strategy_name][1].append(reference)   # references
                            all_results[strategy_name][2].append(input_text)  # inputs
                            all_results[strategy_name][3].append(enhanced_meta)  # metadata
                    
                    successful_generations += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate for sample {idx}: {e}")
                    continue
        
        model.train()
        
        self.logger.info(f"Generated {successful_generations}/{len(sample_indices)} successful multi-strategy predictions")
        
        return all_results
    
    def _extract_pictogram_sequence_from_example(self, example: Dict, sample_idx: int) -> List[int]:
        """Extract pictogram sequence from the original data structure."""
        # Try to get pictogram sequence from the example if it was preserved
        if hasattr(example, 'get') and 'pictogram_sequence' in example:
            return example['pictogram_sequence']
        
        # If not available, try to extract from input text pattern
        if 'input_ids' in example:
            input_text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            return self._extract_pictogram_ids_from_text(input_text)
        
        # If we have access to the original eval_dataset
        try:
            if hasattr(self.eval_dataset, '_original_data') and self.eval_dataset._original_data:
                original_data = self.eval_dataset._original_data
                if sample_idx < len(original_data) and 'pictogram_sequence' in original_data[sample_idx]:
                    return original_data[sample_idx]['pictogram_sequence']
        except:
            pass
        
        return []
    
    def _extract_pictogram_ids_from_text(self, input_text: str) -> List[int]:
        """Extract pictogram IDs from input text based on data format."""
        picto_ids = []
        
        # Look for pictogram ID patterns in the input text
        import re
        
        # Look for the exact pattern used in the data
        pattern = r'pictogrammes:\s*([\d\s]+)'
        
        matches = re.findall(pattern, input_text, re.IGNORECASE)
        for match in matches:
            # Extract individual numbers
            numbers = re.findall(r'\d+', match)
            picto_ids.extend([int(num) for num in numbers])
        
        return picto_ids
    
    def _save_multi_strategy_samples(self, all_results, step):
        """Save detailed generation samples with pictogram sequences for all strategies."""
        comparison_samples = []
        
        # Get the minimum number of samples across all strategies
        min_samples = min(len(results[0]) for results in all_results.values() if results[0])
        sample_count = min(self.generation_samples, min_samples)
        
        for i in range(sample_count):
            sample_entry = {
                'step': step,
                'sample_id': i,
                'input': None,
                'reference': None,
                'pictogram_sequence': [],
                'strategies': {}
            }
            
            # Collect results from all strategies for this sample
            for strategy_name, (predictions, references, inputs, metadata) in all_results.items():
                if i < len(predictions):
                    # Set common fields from first strategy
                    if sample_entry['input'] is None:
                        sample_entry['input'] = safe_text_encode(inputs[i])
                        sample_entry['reference'] = safe_text_encode(references[i])
                        sample_entry['pictogram_sequence'] = metadata[i].get('pictogram_sequence', [])
                    
                    # Add strategy-specific results
                    sample_entry['strategies'][strategy_name] = {
                        'prediction': safe_text_encode(predictions[i]),
                        'generation_time': metadata[i].get('generation_time', 0),
                        'success': metadata[i].get('success', False)
                    }
            
            comparison_samples.append(sample_entry)
        
        self.generation_history.extend(comparison_samples)
        
        # Save samples for this step
        step_file = self.output_dir / "samples" / f"multi_strategy_step_{step}.json"
        safe_json_dump(comparison_samples, step_file)
    
    def save_comprehensive_results(self):
        """Save all collected results with strategy comparisons."""
        results = {
            'experiment_config': {
                'model': self.experiment_config.model_config.name,
                'data': self.experiment_config.data_config.name,
                'experiment_id': self.experiment_config.experiment_id
            },
            'metrics_history': dict(self.metrics_history),
            'best_metrics': self.best_metrics,
            'generation_history': self.generation_history,
            'training_progress': self.training_progress,
            'summary': self._create_comprehensive_summary()
        }
        
        # Save main results
        safe_json_dump(results, self.output_dir / "comprehensive_results.json")
        
        # Save as pickle for Python analysis
        with open(self.output_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create basic visualizations
        self._create_basic_plots()
        
        return results
    
    def _create_comprehensive_summary(self):
        """Create training summary with strategy comparisons."""
        summary = {
            'total_evaluations': len([k for k in self.metrics_history.keys() if 'bleu' in k]),
            'training_steps': len(self.training_progress),
            'strategies_evaluated': len(DecodingStrategies.get_all_strategies()),
            'best_metrics_by_strategy': {},
            'strategy_rankings': {}
        }
        
        # Organize best metrics by strategy
        strategies = [s.name for s in DecodingStrategies.get_all_strategies()]
        key_metrics = ['bleu', 'rouge_l', 'wer', 'generation_success_rate']
        
        for strategy in strategies:
            summary['best_metrics_by_strategy'][strategy] = {}
            for metric in key_metrics:
                full_metric_name = f"{strategy}_{metric}"
                if full_metric_name in self.best_metrics:
                    summary['best_metrics_by_strategy'][strategy][metric] = self.best_metrics[full_metric_name]['value']
        
        # Create strategy rankings for each metric
        for metric in key_metrics:
            strategy_scores = []
            for strategy in strategies:
                full_metric_name = f"{strategy}_{metric}"
                if full_metric_name in self.best_metrics:
                    score = self.best_metrics[full_metric_name]['value']
                    strategy_scores.append((strategy, score))
            
            # Sort (descending for most metrics, ascending for WER)
            reverse_sort = metric != 'wer'
            strategy_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
            summary['strategy_rankings'][metric] = [s[0] for s in strategy_scores]
        
        return summary
    
    def _create_basic_plots(self):
        """Create basic visualization plots for training progress."""
        try:
            # Strategy comparison plots
            self._plot_strategy_comparison()
            
            # Training metrics plot
            self._plot_training_metrics()
            
        except Exception as e:
            self.logger.warning(f"Could not create plots: {e}")
    
    def _plot_strategy_comparison(self):
        """Plot comparison between decoding strategies."""
        strategies = [s.name for s in DecodingStrategies.get_all_strategies()]
        key_metrics = ['bleu', 'rouge_l', 'wer', 'generation_success_rate']
        
        # Create comparison data
        comparison_data = {}
        for metric in key_metrics:
            comparison_data[metric] = []
            for strategy in strategies:
                full_metric_name = f"{strategy}_{metric}"
                if full_metric_name in self.best_metrics:
                    comparison_data[metric].append(self.best_metrics[full_metric_name]['value'])
                else:
                    comparison_data[metric].append(0)
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            values = comparison_data[metric]
            if any(v > 0 for v in values):
                bars = axes[i].bar(strategies, values, alpha=0.7)
                axes[i].set_title(f'{metric.upper()} by Decoding Strategy')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if value > 0:
                        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_metrics(self):
        """Plot training loss and learning rate."""
        if not self.training_progress:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        steps = [entry['step'] for entry in self.training_progress if 'train_loss' in entry]
        losses = [entry['train_loss'] for entry in self.training_progress if 'train_loss' in entry]
        
        if steps and losses:
            ax1.plot(steps, losses, 'b-', alpha=0.7)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # Learning rate
        lr_steps = [entry['step'] for entry in self.training_progress if 'learning_rate' in entry]
        lrs = [entry['learning_rate'] for entry in self.training_progress if 'learning_rate' in entry]
        
        if lr_steps and lrs:
            ax2.plot(lr_steps, lrs, 'g-', alpha=0.7)
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

class ComprehensiveResearchPipeline:
    """Complete academic research pipeline with systematic experiment management and VSC storage."""
    
    def __init__(self, base_experiment_name: str = "propicto_comprehensive", results_path: Optional[str] = None):
        self.base_experiment_name = base_experiment_name
        
        # Initialize HPC environment with custom results path
        self.hpc_env = HPCEnvironment(results_path)
        
        # Setup logging with proper paths
        self.logger = self._setup_master_logging()
        
        # Create master results directory
        self.master_results_dir = self.hpc_env.get_results_dir(base_experiment_name)
        self.hpc_env.ensure_results_structure(self.master_results_dir)
        
        # System info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Experiment tracking
        self.completed_experiments = []
        self.failed_experiments = []
        self.all_results = {}
        
        # Log environment setup
        self.logger.info(f"Pipeline initialized with master results directory: {self.master_results_dir}")
        
    def _setup_master_logging(self):
        """Setup master logging for the entire research pipeline."""
        # Use HPC environment for log directory
        log_dir = self.hpc_env.results_base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.hpc_env.slurm_job_id:
            log_file = log_dir / f"master_log_{timestamp}_job{self.hpc_env.slurm_job_id}.log"
        else:
            log_file = log_dir / f"master_log_{timestamp}.log"
        
        logger = logging.getLogger("master_pipeline")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - MASTER - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - MASTER - %(levelname)s - %(funcName)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_datasets(self, data_config: DataConfig, max_train: int, max_val: int, max_test: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load and limit datasets according to configuration."""
        self.logger.info(f"Loading dataset: {data_config.name}")
        
        # Use HPC environment for data path
        data_root = self.hpc_env.get_data_path()
        config_path = data_root / data_config.path
        
        # Load datasets
        datasets = {}
        for split in ['train', 'valid', 'test']:
            split_path = config_path / split / "data.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing {split} data: {split_path}")
            
            datasets[split] = safe_json_load(split_path)
        
        # Apply limits
        train_data = datasets['train'][:max_train] if max_train else datasets['train']
        valid_data = datasets['valid'][:max_val] if max_val else datasets['valid']
        test_data = datasets['test'][:max_test] if max_test else datasets['test']
        
        self.logger.info(f"   Loaded: {len(train_data)} train, {len(valid_data)} val, {len(test_data)} test")
        
        return train_data, valid_data, test_data
    
    def setup_model_and_tokenizer(self, model_config: ModelConfig):
        """Setup model and tokenizer based on configuration."""
        self.logger.info(f"Setting up model: {model_config.name}")
        
        # Load tokenizer
        if model_config.tokenizer_class == "T5Tokenizer":
            tokenizer = T5Tokenizer.from_pretrained(model_config.model_id)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, use_fast=True)
            except Exception as e:
                self.logger.warning(f"Fast tokenizer failed: {e}")
                tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, use_fast=False)
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_config.model_id)
        
        # Setup tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Setup generation config with safe defaults
        # CRITICAL FIX: Override model's default generation_config to avoid conflicts
        try:
            generation_config = GenerationConfig.from_model_config(model.config)
        except:
            generation_config = GenerationConfig()
        
        # Set safe defaults that work with all decoding strategies
        generation_config.max_length = 128
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.early_stopping = False  # CRITICAL: Set to False by default
        generation_config.num_beams = 1  # CRITICAL: Default to greedy
        generation_config.do_sample = False
        
        # Validate and set
        try:
            generation_config.validate()
            model.generation_config = generation_config
            self.logger.info("Generation config validated and set successfully")
        except Exception as e:
            self.logger.warning(f"Generation config validation failed: {e}")
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
        
        model.to(self.device)
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model loaded: {param_count:,} parameters")
        
        return model, tokenizer
    
    def prepare_datasets(self, train_data, valid_data, test_data, tokenizer, data_config):
        """Prepare datasets with task-specific formatting and preserve pictogram sequences."""
        
        def get_task_input(input_text: str) -> str:
            """Format input according to task configuration."""
            input_text = safe_text_encode(input_text)
            
            # Clean the input text
            cleaned_input = input_text.replace('mots:', '').replace('tokens:', '').replace('hybrid:', '').replace('direct:', '').strip()
            
            # Apply task-specific formatting
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
            
            model_inputs["labels"] = [
                [t if t != tokenizer.pad_token_id else -100 for t in label_ids]
                for label_ids in labels["input_ids"]
            ]
            
            # Preserve pictogram sequences if available in original data
            if 'pictogram_sequence' in examples:
                model_inputs["pictogram_sequence"] = examples['pictogram_sequence']
            
            return model_inputs
        
        # Create datasets with preserved metadata
        datasets = {}
        for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            if data:
                dataset = Dataset.from_list(data)
                
                # Store original data for pictogram sequence lookup
                tokenized_dataset = dataset.map(
                    tokenize_function, 
                    batched=True, 
                    remove_columns=[col for col in dataset.column_names if col not in ['pictogram_sequence']]
                )
                
                # Store reference to original data for pictogram sequence extraction
                tokenized_dataset._original_data = data
                
                datasets[name] = tokenized_dataset
        
        return datasets['train'], datasets['valid'], datasets.get('test')
    
    def run_single_experiment(self, experiment_config: ExperimentConfig):
        """Run a single experiment with comprehensive tracking and VSC storage optimization."""
        
        experiment_start_time = time.time()
        experiment_dir = self.master_results_dir / experiment_config.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting experiment: {experiment_config.experiment_id}")
        self.logger.info(f"   Model: {experiment_config.model_config.name}")
        self.logger.info(f"   Data: {experiment_config.data_config.name}")
        self.logger.info(f"   Will evaluate all decoding strategies")
        self.logger.info(f"   Experiment directory: {experiment_dir}")
        
        try:
            # Save experiment configuration
            config_dict = {
                'experiment_id': experiment_config.experiment_id,
                'model': {
                    'name': experiment_config.model_config.name,
                    'model_id': experiment_config.model_config.model_id,
                    'description': experiment_config.model_config.description,
                    'is_multilingual': experiment_config.model_config.is_multilingual
                },
                'data': {
                    'name': experiment_config.data_config.name,
                    'path': experiment_config.data_config.path,
                    'description': experiment_config.data_config.description,
                    'task_prefix': experiment_config.data_config.task_prefix_template
                },
                'training': {
                    'max_train_samples': experiment_config.max_train_samples,
                    'max_val_samples': experiment_config.max_val_samples,
                    'max_test_samples': experiment_config.max_test_samples,
                    'num_epochs': experiment_config.num_epochs,
                    'batch_size': experiment_config.batch_size,
                    'learning_rate': experiment_config.learning_rate
                },
                'environment': {
                    'vsc_data': self.hpc_env.vsc_data,
                    'vsc_scratch': self.hpc_env.vsc_scratch,
                    'slurm_job_id': self.hpc_env.slurm_job_id,
                    'slurm_node': self.hpc_env.slurm_node,
                    'results_base': str(self.hpc_env.results_base),
                    'working_dir': str(self.hpc_env.working_dir)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            safe_json_dump(config_dict, experiment_dir / "experiment_config.json")
            
            # Load datasets
            train_data, valid_data, test_data = self.load_datasets(
                experiment_config.data_config,
                experiment_config.max_train_samples,
                experiment_config.max_val_samples,
                experiment_config.max_test_samples
            )
            
            # Setup model and tokenizer
            model, tokenizer = self.setup_model_and_tokenizer(experiment_config.model_config)
            
            # Prepare datasets with pictogram sequence preservation
            train_dataset, valid_dataset, test_dataset = self.prepare_datasets(
                train_data, valid_data, test_data, tokenizer, experiment_config.data_config
            )
            
            # Setup evaluation
            evaluator = Evaluator(tokenizer)
            
            # Setup callback with enhanced pictogram tracking
            callback = AcademicResearchCallback(
                evaluator=evaluator,
                eval_dataset=valid_dataset,
                output_dir=experiment_dir,
                experiment_config=experiment_config,
                tokenizer=tokenizer,
                generation_samples=20
            )
            
            # Training configuration
            steps_per_epoch = len(train_dataset) // experiment_config.batch_size
            eval_steps = max(50, steps_per_epoch // 4)
            save_steps = eval_steps
            
            # Version-compatible training arguments
            try:
                TrainingArguments(output_dir="./test", eval_strategy="no")
                eval_strategy_param = "eval_strategy"
            except TypeError:
                eval_strategy_param = "evaluation_strategy"
            
            training_args_dict = {
                "output_dir": str(experiment_dir / "checkpoints"),
                "num_train_epochs": experiment_config.num_epochs,
                "per_device_train_batch_size": experiment_config.batch_size,
                "per_device_eval_batch_size": experiment_config.batch_size,
                "learning_rate": experiment_config.learning_rate,
                "weight_decay": 0.01,
                "warmup_steps": min(500, steps_per_epoch // 10),
                "logging_steps": max(10, steps_per_epoch // 10),
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                eval_strategy_param: "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "save_total_limit": 2,
                "fp16": False,
                "report_to": [],
                "run_name": experiment_config.experiment_id,
                "logging_dir": str(experiment_dir / "tensorboard")
            }
            
            training_args = TrainingArguments(**training_args_dict)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=tokenizer,
                    model=model,
                    padding=True,
                    label_pad_token_id=-100
                ),
                tokenizer=tokenizer,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=3),
                    callback
                ]
            )
            
            # Training
            self.logger.info("Starting training...")
            trainer.train()
            
            # Final evaluation with all strategies and pictogram tracking
            if test_dataset:
                self.logger.info("Running final test evaluation with all strategies...")
                
                decoder = DecodingStrategies(model, tokenizer, model.device)
                test_results = callback._generate_multi_strategy_predictions(
                    model, tokenizer, decoder, test_dataset,
                    num_samples=min(1000, len(test_dataset))
                )
                
                # Evaluate each strategy on test set
                final_test_metrics = {}
                for strategy_name, (predictions, references, inputs, metadata) in test_results.items():
                    if predictions and references:
                        metrics = evaluator.evaluate_predictions(predictions, references, strategy_name)
                        final_test_metrics[strategy_name] = metrics
                        
                        self.logger.info(f"{strategy_name.upper()} Test Results:")
                        for metric_name, value in metrics.items():
                            if metric_name in ['bleu', 'rouge_l', 'wer', 'vocab_overlap', 'generation_success_rate']:
                                self.logger.info(f"   {metric_name}: {value:.4f}")
                
                # Save test results with pictogram sequences
                detailed_test_results = []
                max_samples = min(100, len(test_results['greedy'][0]) if 'greedy' in test_results else 0)
                
                for i in range(max_samples):
                    sample_result = {
                        'sample_id': i,
                        'input': safe_text_encode(test_results['greedy'][2][i]),
                        'reference': safe_text_encode(test_results['greedy'][1][i]),
                        'pictogram_sequence': test_results['greedy'][3][i].get('pictogram_sequence', []),
                        'strategies': {}
                    }
                    
                    for strategy_name, (predictions, _, _, metadata) in test_results.items():
                        if i < len(predictions):
                            sample_result['strategies'][strategy_name] = {
                                'prediction': safe_text_encode(predictions[i]),
                                'generation_time': metadata[i].get('generation_time', 0)
                            }
                    
                    detailed_test_results.append(sample_result)
                
                safe_json_dump(detailed_test_results, experiment_dir / "final_test_results.json")
                safe_json_dump(final_test_metrics, experiment_dir / "final_test_metrics.json")
            
            # Save model for potential deployment
            final_model_dir = experiment_dir / "final_model"
            trainer.save_model(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            
            # Save comprehensive results
            training_results = callback.save_comprehensive_results()
            
            # Compile experiment results
            experiment_results = {
                'experiment_config': config_dict,
                'training_results': training_results,
                'final_test_metrics': final_test_metrics if test_dataset else {},
                'training_time_minutes': (time.time() - experiment_start_time) / 60,
                'model_path': str(final_model_dir),
                'success': True
            }
            
            safe_json_dump(experiment_results, experiment_dir / "experiment_summary.json")
            
            # Track successful experiment
            self.completed_experiments.append(experiment_config.experiment_id)
            self.all_results[experiment_config.experiment_id] = experiment_results
            
            self.logger.info(f"Experiment completed: {experiment_config.experiment_id}")
            self.logger.info(f"   Training time: {experiment_results['training_time_minutes']:.1f} minutes")
            
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {experiment_config.experiment_id} - {e}")
            self.failed_experiments.append((experiment_config.experiment_id, str(e)))
            
            # Save failure info
            failure_info = {
                'experiment_id': experiment_config.experiment_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            safe_json_dump(failure_info, experiment_dir / "failure_info.json")
            
            import traceback
            self.logger.debug(traceback.format_exc())
            raise
    
    def run_all_experiments(self, test_run: bool = False):
        """Run the complete experimental matrix with all model and data combinations."""
        
        start_time = time.time()
        experiments = ExperimentalMatrix.generate_all_experiments(test_run=test_run)
        
        self.logger.info("COMPREHENSIVE PROPICTO RESEARCH PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total experiments: {len(experiments)} (3 models × 4 data configs)")
        self.logger.info(f"Each experiment evaluates 3 decoding strategies")
        self.logger.info(f"Models: {len(ExperimentalMatrix.get_model_configs())}")
        self.logger.info(f"Data configs: {len(ExperimentalMatrix.get_data_configs())}")
        self.logger.info(f"Results storage: {self.master_results_dir}")
        
        if test_run:
            self.logger.info("TEST RUN MODE - Limited samples and epochs")
        
        # Save experimental matrix overview
        matrix_overview = {
            'total_experiments': len(experiments),
            'total_model_data_combinations': len(experiments),
            'decoding_strategies_per_experiment': 3,
            'models': [m.name for m in ExperimentalMatrix.get_model_configs()],
            'data_configs': [d.name for d in ExperimentalMatrix.get_data_configs()],
            'decoding_strategies': [s.name for s in DecodingStrategies.get_all_strategies()],
            'test_run': test_run,
            'start_time': datetime.now().isoformat(),
            'environment': {
                'vsc_data': self.hpc_env.vsc_data,
                'vsc_scratch': self.hpc_env.vsc_scratch,
                'results_base': str(self.hpc_env.results_base),
                'working_dir': str(self.hpc_env.working_dir),
                'slurm_job_id': self.hpc_env.slurm_job_id
            }
        }
        
        safe_json_dump(matrix_overview, self.master_results_dir / "experimental_matrix.json")
        
        # Run experiments
        for i, experiment in enumerate(experiments, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EXPERIMENT {i}/{len(experiments)}")
            self.logger.info(f"ID: {experiment.experiment_id}")
            self.logger.info(f"Model: {experiment.model_config.name}")
            self.logger.info(f"Data: {experiment.data_config.name}")
            self.logger.info(f"Will evaluate: greedy, beam_search, nucleus_sampling")
            self.logger.info(f"{'='*60}")
            
            try:
                self.run_single_experiment(experiment)
                
                # Log progress
                success_rate = len(self.completed_experiments) / i * 100
                self.logger.info(f"Progress: {i}/{len(experiments)} ({success_rate:.1f}% success rate)")
                
            except Exception as e:
                self.logger.error(f"Experiment {i} failed, continuing with next...")
                continue
        
        total_time = time.time() - start_time
        
        # Generate comprehensive analysis
        self._generate_comprehensive_analysis()
        
        # Final summary
        self._log_final_summary(total_time)
        
        return self.master_results_dir
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis across all experiments."""
        self.logger.info("Generating comprehensive analysis...")
        
        # Collect all results
        analysis = {
            'overview': {
                'total_experiments': len(self.completed_experiments) + len(self.failed_experiments),
                'successful_experiments': len(self.completed_experiments),
                'failed_experiments': len(self.failed_experiments),
                'success_rate': len(self.completed_experiments) / (len(self.completed_experiments) + len(self.failed_experiments)) * 100
            },
            'model_performance': {},
            'data_config_performance': {},
            'decoding_strategy_performance': {},
            'pictogram_analysis': {},
            'best_overall_results': {}
        }
        
        # Analyze by model, data config, and strategy
        model_results = defaultdict(list)
        data_results = defaultdict(list)
        strategy_results = defaultdict(list)
        
        for exp_id, results in self.all_results.items():
            if 'final_test_metrics' in results and results['final_test_metrics']:
                model_name = results['experiment_config']['model']['name']
                data_name = results['experiment_config']['data']['name']
                
                # For each decoding strategy
                for strategy, metrics in results['final_test_metrics'].items():
                    model_results[model_name].append(metrics)
                    data_results[data_name].append(metrics)
                    strategy_results[strategy].append(metrics)
        
        # Calculate averages
        key_metrics = ['bleu', 'rouge_l', 'wer', 'generation_success_rate']
        
        for model_name, metric_list in model_results.items():
            analysis['model_performance'][model_name] = {}
            for metric in key_metrics:
                values = [m.get(metric, 0) for m in metric_list if metric in m]
                if values:
                    analysis['model_performance'][model_name][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
        
        for data_name, metric_list in data_results.items():
            analysis['data_config_performance'][data_name] = {}
            for metric in key_metrics:
                values = [m.get(metric, 0) for m in metric_list if metric in m]
                if values:
                    analysis['data_config_performance'][data_name][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }
        
        for strategy_name, metric_list in strategy_results.items():
            analysis['decoding_strategy_performance'][strategy_name] = {}
            for metric in key_metrics:
                values = [m.get(metric, 0) for m in metric_list if metric in m]
                if values:
                    analysis['decoding_strategy_performance'][strategy_name][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }
        
        # Find best overall results
        best_scores = {}
        for metric in key_metrics:
            best_scores[metric] = {'value': -1 if metric != 'wer' else float('inf'), 'experiment': None}
            
            for exp_id, results in self.all_results.items():
                if 'final_test_metrics' in results:
                    for strategy, metrics in results['final_test_metrics'].items():
                        if metric in metrics:
                            value = metrics[metric]
                            if ((metric != 'wer' and value > best_scores[metric]['value']) or
                                (metric == 'wer' and value < best_scores[metric]['value'])):
                                best_scores[metric] = {
                                    'value': float(value),
                                    'experiment': exp_id,
                                    'strategy': strategy,
                                    'model': results['experiment_config']['model']['name'],
                                    'data': results['experiment_config']['data']['name']
                                }
        
        analysis['best_overall_results'] = best_scores
        
        # Save comprehensive analysis
        safe_json_dump(analysis, self.master_results_dir / "comprehensive_analysis.json")
        
        # Create basic visualizations
        self._create_master_visualizations(analysis)
        
        return analysis
    
    def _create_master_visualizations(self, analysis):
        """Create master visualizations for the entire experimental matrix."""
        
        fig_dir = self.master_results_dir / "visualizations"
        fig_dir.mkdir(exist_ok=True)
        
        # Model comparison
        self._plot_model_comparison(analysis, fig_dir)
        
        # Data configuration comparison
        self._plot_data_config_comparison(analysis, fig_dir)
        
        # Decoding strategy comparison
        self._plot_decoding_strategy_comparison(analysis, fig_dir)
        
    def _plot_model_comparison(self, analysis, fig_dir):
        """Plot model performance comparison."""
        model_perf = analysis['model_performance']
        
        if not model_perf:
            return
        
        models = list(model_perf.keys())
        metrics = ['bleu', 'rouge_l', 'wer', 'generation_success_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = []
                errors = []
                model_names = []
                
                for model in models:
                    if metric in model_perf[model]:
                        values.append(model_perf[model][metric]['mean'])
                        errors.append(model_perf[model][metric]['std'])
                        model_names.append(model)
                
                if values:
                    bars = axes[i].bar(model_names, values, yerr=errors, capsize=5, alpha=0.7)
                    axes[i].set_title(f'{metric.upper()} by Model')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_config_comparison(self, analysis, fig_dir):
        """Plot data configuration performance comparison."""
        data_perf = analysis['data_config_performance']
        
        if not data_perf:
            return
        
        configs = list(data_perf.keys())
        metrics = ['bleu', 'rouge_l', 'wer', 'generation_success_rate']
        
        # Create a grouped bar chart
        x = np.arange(len(configs))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
        
        for i, metric in enumerate(metrics):
            values = []
            for config in configs:
                if metric in data_perf[config]:
                    values.append(data_perf[config][metric]['mean'])
                else:
                    values.append(0)
            
            ax.bar(x + i*width, values, width, label=metric.upper(), color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Data Configuration')
        ax.set_ylabel('Score')
        ax.set_title('Performance by Data Configuration')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([config.replace('_', '\n') for config in configs], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "data_config_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_decoding_strategy_comparison(self, analysis, fig_dir):
        """Plot decoding strategy performance comparison."""
        strategy_perf = analysis['decoding_strategy_performance']
        
        if not strategy_perf:
            return
        
        strategies = list(strategy_perf.keys())
        metrics = ['bleu', 'rouge_l', 'wer', 'generation_success_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = []
                errors = []
                strategy_names = []
                
                for strategy in strategies:
                    if metric in strategy_perf[strategy]:
                        values.append(strategy_perf[strategy][metric]['mean'])
                        errors.append(strategy_perf[strategy][metric]['std'])
                        strategy_names.append(strategy)
                
                if values:
                    bars = axes[i].bar(strategy_names, values, yerr=errors, capsize=5, alpha=0.7)
                    axes[i].set_title(f'{metric.upper()} by Decoding Strategy')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "decoding_strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _log_final_summary(self, total_time_seconds):
        """Log comprehensive final summary."""
        
        self.logger.info("\n" + "="*100)
        self.logger.info("COMPREHENSIVE PROPICTO RESEARCH PIPELINE COMPLETED")
        self.logger.info("="*100)
        
        # Overview
        total_experiments = len(self.completed_experiments) + len(self.failed_experiments)
        success_rate = len(self.completed_experiments) / total_experiments * 100 if total_experiments > 0 else 0
        
        self.logger.info(f"EXPERIMENT OVERVIEW:")
        self.logger.info(f"   Total experiments: {total_experiments}")
        self.logger.info(f"   Successful: {len(self.completed_experiments)}")
        self.logger.info(f"   Failed: {len(self.failed_experiments)}")
        self.logger.info(f"   Success rate: {success_rate:.1f}%")
        self.logger.info(f"   Total runtime: {total_time_seconds/3600:.2f} hours")
        
        # Storage info
        self.logger.info(f"\nSTORAGE INFORMATION:")
        self.logger.info(f"   Results saved to: {self.master_results_dir}")
        self.logger.info(f"   Storage base: {self.hpc_env.results_base}")
        self.logger.info(f"   Working directory: {self.hpc_env.working_dir}")
        if self.hpc_env.slurm_job_id:
            self.logger.info(f"   SLURM Job ID: {self.hpc_env.slurm_job_id}")
        
        # Best results
        if self.all_results:
            self.logger.info(f"\nBEST RESULTS:")
            
            # Find best BLEU score
            best_bleu = 0
            best_bleu_exp = None
            best_wer = float('inf')
            best_wer_exp = None
            
            for exp_id, results in self.all_results.items():
                if 'final_test_metrics' in results:
                    for strategy, metrics in results['final_test_metrics'].items():
                        if 'bleu' in metrics and metrics['bleu'] > best_bleu:
                            best_bleu = metrics['bleu']
                            best_bleu_exp = f"{exp_id} ({strategy})"
                        
                        if 'wer' in metrics and metrics['wer'] < best_wer:
                            best_wer = metrics['wer']
                            best_wer_exp = f"{exp_id} ({strategy})"
            
            if best_bleu_exp:
                self.logger.info(f"   Best BLEU: {best_bleu:.4f} from {best_bleu_exp}")
            if best_wer_exp:
                self.logger.info(f"   Best WER: {best_wer:.4f} from {best_wer_exp}")
        
        # Failed experiments
        if self.failed_experiments:
            self.logger.info(f"\nFAILED EXPERIMENTS:")
            for exp_id, error in self.failed_experiments:
                self.logger.info(f"   {exp_id}: {error}")
        
        # Paths and deliverables
        self.logger.info(f"\nDELIVERABLES:")
        self.logger.info(f"   Master results: {self.master_results_dir}")
        self.logger.info(f"   Comprehensive analysis: {self.master_results_dir / 'comprehensive_analysis.json'}")
        self.logger.info(f"   Visualizations: {self.master_results_dir / 'visualizations'}")
        
        # Academic insights
        self.logger.info(f"\nACADEMIC INSIGHTS:")
        self.logger.info(f"   Complete experimental matrix: 3 models × 4 data configs")
        self.logger.info(f"   Comprehensive evaluation: BLEU, ROUGE-L, WER, success rate")
        self.logger.info(f"   Multi-strategy comparison per experiment")
        self.logger.info(f"   Pictogram sequence tracking for error analysis")
        
        # Final instructions
        self.logger.info(f"\nNEXT STEPS:")
        self.logger.info(f"   1. Review comprehensive analysis and visualizations")
        self.logger.info(f"   2. Analyze pictogram sequence performance")
        self.logger.info(f"   3. Prepare research publication")
        
        self.logger.info("="*100)

def main():
    """Main function with comprehensive argument parsing and VSC storage support."""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Academic Research Pipeline for ProPicto with VSC Storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run full experimental matrix (12 experiments)
  python main_runner.py --run-all
  
  # Run test mode (quick validation)
  python main_runner.py --run-all --test-run
  
  # Run with custom results path (VSC scratch)
  python main_runner.py --run-all --results-path $VSC_SCRATCH/pictoSeq_results
  
  # Run specific experiment
  python main_runner.py --model barthez --data keywords_to_sentence
  
  # Run with custom limits
  python main_runner.py --run-all --max-train 10000 --max-test 1000

Experimental Matrix:
  Models: barthez, french_t5, mt5_base
  Data Configs: keywords_to_sentence, pictos_tokens_to_sentence, hybrid_to_sentence, direct_to_sentence  
  Total: 3 × 4 = 12 experiments (each evaluates 3 decoding strategies)
  
VSC Storage:
  Results saved to $VSC_SCRATCH/pictoSeq_results/ by default
  Working directory: $VSC_DATA/pictoSeq/ (where code and data are)
  Pictogram sequences tracked throughout pipeline
        '''
    )
    
    # Main execution modes
    execution_group = parser.add_mutually_exclusive_group(required=True)
    execution_group.add_argument('--run-all', action='store_true',
                                help='Run complete experimental matrix (12 experiments)')
    execution_group.add_argument('--single-experiment', action='store_true',
                                help='Run single experiment with specified parameters')
    
    # Single experiment parameters
    parser.add_argument('--model', 
                       choices=['barthez', 'french_t5', 'mt5_base'],
                       help='Model architecture for single experiment')
    parser.add_argument('--data',
                       choices=['keywords_to_sentence', 'pictos_tokens_to_sentence', 
                               'hybrid_to_sentence', 'direct_to_sentence'],
                       help='Data configuration for single experiment')
    
    # Storage configuration
    parser.add_argument('--results-path', type=str,
                       help='Custom path for results storage (default: $VSC_SCRATCH/pictoSeq_results)')
    
    # Training parameters
    parser.add_argument('--max-train', type=int, default=10000,help='Maximum training samples (default: 10000)')
    parser.add_argument('--max-val', type=int, default=1000,help='Maximum validation samples (default: 1000)')
    parser.add_argument('--max-test', type=int, default=1000,help='Maximum test samples (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5,help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=8,help='Training batch size (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=3e-5,help='Learning rate (default: 3e-5)')
    
    # Execution options
    parser.add_argument('--test-run', action='store_true',help='Run in test mode (limited samples and epochs)')
    parser.add_argument('--experiment-name', default='propicto_comprehensive',help='Base name for experiments')
    
    args = parser.parse_args()
    
    # Validate single experiment arguments
    if args.single_experiment:
        if not all([args.model, args.data]):
            parser.error("Single experiment mode requires --model and --data")
    
    if args.test_run:
        print("TEST RUN MODE - Limited samples and epochs for validation")
        
    # Test French UTF-8 support
    test_french = "Test français: café, être, déjà, naïve, mémoire, pictogramme"
    print(f"UTF-8 test: {test_french}")
    
    # Check jiwer availability
    if HAS_JIWER:
        print("jiwer available - WER calculation enabled")
    else:
        print("jiwer not available - using fallback WER calculation")
    
    # Initialize pipeline with custom results path
    pipeline = ComprehensiveResearchPipeline(base_experiment_name=args.experiment_name,results_path=args.results_path)
    
    try:
        if args.run_all:
            # Run complete experimental matrix
            print(f"running complete experimental matrix...")
            print(f"   Total experiments: 12 (3 models × 4 configs)")
            print(f"   Each experiment evaluates 3 decoding strategies")
            print(f"   Max samples: {args.max_train} train, {args.max_val} val, {args.max_test} test")
            print(f"   Results storage: {pipeline.master_results_dir}")
            
            if args.test_run:
                print(f"   Test mode: Limited to 1000 train, 200 val/test, 2 epochs")
            
            results_dir = pipeline.run_all_experiments(test_run=args.test_run)
            
            print(f"COMPLETE EXPERIMENTAL MATRIX FINISHED!")
            print(f"Results directory: {results_dir}")
            
        else:
            # Run single experiment
            print(f"running single experiment...")
            print(f"   Model: {args.model}")
            print(f"   Data: {args.data}")
            print(f"   Will evaluate all 3 decoding strategies")
            print(f"   Results storage: {pipeline.master_results_dir}")
            
            # Create single experiment config
            model_configs = {m.name: m for m in ExperimentalMatrix.get_model_configs()}
            data_configs = {d.name: d for d in ExperimentalMatrix.get_data_configs()}
            
            experiment_config = ExperimentConfig(model_config=model_configs[args.model],data_config=data_configs[args.data],experiment_id=f"{args.model}_{args.data}",max_train_samples=args.max_train if not args.test_run else 1000,max_val_samples=args.max_val if not args.test_run else 200,max_test_samples=args.max_test if not args.test_run else 200,num_epochs=args.epochs if not args.test_run else 2,batch_size=args.batch_size,learning_rate=args.learning_rate)
            
            results = pipeline.run_single_experiment(experiment_config)
            
            print(f"SINGLE EXPERIMENT COMPLETED!")
            print(f"Results: {results['model_path']}")
            
            # Show key metrics
            if 'final_test_metrics' in results:
                print(f"Key Results:")
                for strategy, metrics in results['final_test_metrics'].items():
                    print(f"   {strategy.upper()}:")
                    for metric in ['bleu', 'rouge_l', 'wer', 'generation_success_rate']:
                        if metric in metrics:
                            print(f"     {metric}: {metrics[metric]:.3f}")
        
    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        print("check logs for detailed error information")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())