#!/usr/bin/env python3
"""
Main script for running the pictoSeq experiments. 
"""

import logging
import torch
import json
import time
import argparse
import os
import sys
import shutil
import psutil
import locale
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import math
import pickle
import yaml

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments, Trainer, DataCollatorForSeq2Seq,EarlyStoppingCallback, TrainerCallback, GenerationConfig)
from datasets import Dataset


def setup_utf8_environment():
    """
    Setup proper UTF-8 environment to prevent encoding issues
    This is the ROOT FIX for all encoding problems
    """
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except locale.Error:
            pass  # Use system default
    
    plt.rcParams['font.family'] = ['DejaVu Sans']

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
        # File-like object
        return json.load(fp)
    else:
        # Path-like object
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)

def safe_text_encode(text: str) -> str:
    """
    Ensure text is properly UTF-8 encoded
    Use this for any text that will be saved or logged
    """
    if not isinstance(text, str):
        return str(text)
    
    # Ensure proper UTF-8 encoding
    try:
        # If already proper UTF-8, this will work fine
        text.encode('utf-8').decode('utf-8')
        return text
    except UnicodeError:
        # If there are encoding issues, fix them
        return text.encode('utf-8', errors='replace').decode('utf-8')

# QUICK FIX: Suppress the generation config warning
import warnings
warnings.filterwarnings("ignore", message="Moving the following attributes in the config to the generation config")

# Call this immediately when module loads
setup_utf8_environment()

class HPCEnvironment:
    """HPC environment detection and setup"""
    
    def __init__(self):
        self.vsc_scratch = os.environ.get('VSC_SCRATCH')
        self.vsc_data = os.environ.get('VSC_DATA')
        self.job_id = os.environ.get('PBS_JOBID', 'interactive')
        self.node_name = os.environ.get('HOSTNAME', 'unknown')
        
    def get_storage_paths(self, experiment_name: str):
        """Get optimal storage paths for HPC"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.vsc_scratch:
            # Use VSC_SCRATCH for temporary files and training
            scratch_base = Path(self.vsc_scratch) / "propicto_experiments"
            experiment_dir = scratch_base / f"{experiment_name}_{timestamp}"
            
            # Use VSC_DATA for permanent storage if available
            if self.vsc_data:
                permanent_base = Path(self.vsc_data) / "propicto_results"
                permanent_dir = permanent_base / f"{experiment_name}_{timestamp}"
            else:
                permanent_dir = experiment_dir
                
        else:
            # Fallback for non-HPC environments
            base_dir = Path.cwd() / "experiments"
            experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
            permanent_dir = experiment_dir
        
        return {
            'experiment_dir': experiment_dir,
            'permanent_dir': permanent_dir,
            'scratch_dir': experiment_dir,
            'timestamp': timestamp
        }
    
    def get_system_info(self):
        """Get comprehensive system information"""
        info = {
            'hostname': self.node_name,
            'job_id': self.job_id,
            'vsc_scratch': self.vsc_scratch,
            'vsc_data': self.vsc_data,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'encoding_info': {
                'default_encoding': sys.getdefaultencoding(),
                'filesystem_encoding': sys.getfilesystemencoding(),
                'stdout_encoding': getattr(sys.stdout, 'encoding', 'unknown'),
                'locale': locale.getlocale()
            }
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'gpu_memory_gb': [torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                                for i in range(torch.cuda.device_count())]
            })
        
        return info

def setup_comprehensive_logging(log_dir: Path, experiment_name: str):
    """Setup comprehensive logging system with proper UTF-8 handling"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (detailed format) with explicit UTF-8
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Test UTF-8 logging
    test_text = "Test français: café, être, déjà, naïve"
    logger.info(f"📝 Comprehensive logging initialized: {log_file}")
    logger.info(f"🔤 UTF-8 test: {test_text}")
    
    return logger

class ResearchEvaluator:
    """Research-grade evaluator with comprehensive metrics"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate comprehensive research metrics"""
        if not predictions or not references:
            return {}
        
        # Ensure all text is properly encoded
        predictions = [safe_text_encode(p) for p in predictions]
        references = [safe_text_encode(r) for r in references]
        
        results = {}
        
        # Length statistics
        pred_lengths = [len(p.split()) for p in predictions if p.strip()]
        ref_lengths = [len(r.split()) for r in references if r.strip()]
        
        if pred_lengths and ref_lengths:
            results.update({
                'avg_pred_length': np.mean(pred_lengths),
                'avg_ref_length': np.mean(ref_lengths),
                'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
                'length_std_pred': np.std(pred_lengths),
                'length_std_ref': np.std(ref_lengths),
                'min_pred_length': min(pred_lengths),
                'max_pred_length': max(pred_lengths),
                'median_pred_length': np.median(pred_lengths)
            })
        
        # Core NLP metrics
        results['bleu'] = self._calculate_bleu(predictions, references)
        results['rouge_l'] = self._calculate_rouge_l(predictions, references)
        
        # Lexical diversity metrics
        results.update(self._calculate_lexical_metrics(predictions, references))
        
        # French linguistic metrics
        results.update(self._calculate_french_linguistic_metrics(predictions, references))
        
        # Quality assessment metrics
        results.update(self._calculate_quality_metrics(predictions, references))
        
        return results
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """BLEU score with individual n-gram precision"""
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
        """ROUGE-L F1 score"""
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
    
    def _calculate_lexical_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Lexical diversity and overlap metrics"""
        metrics = {}
        
        # Vocabulary overlap
        vocab_overlaps = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                overlap = len(pred_words & ref_words) / len(ref_words)
                vocab_overlaps.append(overlap)
        
        metrics['vocab_overlap'] = np.mean(vocab_overlaps) if vocab_overlaps else 0.0
        
        # Type-token ratio (lexical diversity)
        ttr_scores = []
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                ttr = len(set(words)) / len(words)
                ttr_scores.append(ttr)
        
        metrics['type_token_ratio'] = np.mean(ttr_scores) if ttr_scores else 0.0
        
        # Unique words per sentence
        unique_word_counts = []
        for pred in predictions:
            unique_word_counts.append(len(set(pred.lower().split())))
        
        metrics['avg_unique_words'] = np.mean(unique_word_counts) if unique_word_counts else 0.0
        
        return metrics
    
    def _calculate_french_linguistic_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """French-specific linguistic quality metrics"""
        metrics = {}
        
        # French articles
        french_articles = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'des'}
        
        # French pronouns
        french_pronouns = {'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'lui', 'leur'}
        
        # French verbs (common auxiliaries and modals)
        french_verbs = {'être', 'avoir', 'aller', 'faire', 'pouvoir', 'vouloir', 'devoir', 'savoir'}
        
        article_ratios = []
        pronoun_ratios = []
        verb_ratios = []
        
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                article_count = len([w for w in words if w in french_articles])
                pronoun_count = len([w for w in words if w in french_pronouns])
                verb_count = len([w for w in words if w in french_verbs])
                
                article_ratios.append(article_count / len(words))
                pronoun_ratios.append(pronoun_count / len(words))
                verb_ratios.append(verb_count / len(words))
        
        metrics['french_article_ratio'] = np.mean(article_ratios) if article_ratios else 0.0
        metrics['french_pronoun_ratio'] = np.mean(pronoun_ratios) if pronoun_ratios else 0.0
        metrics['french_verb_ratio'] = np.mean(verb_ratios) if verb_ratios else 0.0
        
        return metrics
    
    def _calculate_quality_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Text quality assessment metrics"""
        metrics = {}
        
        # Fluency (no excessive repetition)
        fluency_scores = []
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                fluency_scores.append(unique_ratio)
        
        metrics['fluency_score'] = np.mean(fluency_scores) if fluency_scores else 0.0
        
        # Generation completeness (non-empty, reasonable length)
        valid_generations = 0
        total_generations = len(predictions)
        
        for pred in predictions:
            if (len(pred.strip()) > 0 and 
                len(pred.split()) > 1 and 
                '<extra_id_0>' not in pred and
                len(pred.split()) < 50):  # Not too long
                valid_generations += 1
        
        metrics['generation_success_rate'] = valid_generations / total_generations if total_generations > 0 else 0.0
        
        # Average sentence complexity (simple heuristic)
        complexity_scores = []
        for pred in predictions:
            # Count conjunctions and complex punctuation
            complexity_indicators = [',', ';', 'que', 'qui', 'dont', 'où', 'et', 'mais', 'ou', 'car']
            complexity_count = sum(1 for word in pred.lower().split() if word in complexity_indicators)
            complexity_count += pred.count(',') + pred.count(';')
            
            word_count = len(pred.split())
            if word_count > 0:
                complexity_scores.append(complexity_count / word_count)
        
        metrics['complexity_score'] = np.mean(complexity_scores) if complexity_scores else 0.0
        
        return metrics

class ResearchCallback(TrainerCallback):
    """Research-grade training callback with comprehensive monitoring and proper UTF-8 handling"""
    
    def __init__(self, evaluator, eval_dataset, output_dir, config_name, tokenizer, generation_samples=10):
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.generation_samples = generation_samples
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive tracking
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.generation_history = []
        self.training_progress = []
        self.resource_usage = []
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Log training start"""
        self.logger.info("🚀 Training started - initializing comprehensive monitoring")
        self._log_system_resources()
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log epoch start"""
        self.logger.info(f"📅 Epoch {state.epoch + 1} started")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track training metrics"""
        if logs:
            progress_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.now().isoformat(),
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
            }
            self.training_progress.append(progress_entry)
            
            # Log system resources periodically
            if state.global_step % 100 == 0:
                self._log_system_resources()
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        """Comprehensive evaluation"""
        
        if state.global_step % (args.eval_steps * 2) == 0:
            self.logger.info(f"📊 Running comprehensive evaluation at step {state.global_step}")
            
            try:
                # Use stored tokenizer if parameter is None
                active_tokenizer = tokenizer if tokenizer is not None else self.tokenizer
                
                if active_tokenizer is None:
                    self.logger.warning("⚠️  No tokenizer available for evaluation")
                    return
                
                # Generate predictions
                predictions, references, inputs = self._generate_comprehensive_predictions(
                    model, active_tokenizer, self.eval_dataset, 
                    num_samples=min(200, len(self.eval_dataset))
                )
                
                if not predictions or not references:
                    self.logger.warning("⚠️  No valid predictions generated")
                    return
                
                # Calculate metrics
                metrics = self.evaluator.evaluate_predictions(predictions, references)
                
                # Log and store metrics
                self.logger.info("📈 Comprehensive Evaluation Metrics:")
                evaluation_entry = {
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'timestamp': datetime.now().isoformat(),
                    'num_samples': len(predictions)
                }
                
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.metrics_history[metric_name].append({
                            'step': state.global_step,
                            'epoch': state.epoch,
                            'value': float(value)
                        })
                        evaluation_entry[metric_name] = float(value)
                        self.logger.info(f"   {metric_name}: {value:.4f}")
                        
                        # Track best metrics
                        if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]['value']:
                            self.best_metrics[metric_name] = {
                                'value': float(value),
                                'step': state.global_step,
                                'epoch': state.epoch
                            }
                
                # Save detailed samples with proper encoding
                self._save_generation_samples(inputs, predictions, references, state.global_step)
                
                # Save metrics snapshot with proper encoding
                metrics_file = self.output_dir / "metrics" / f"eval_step_{state.global_step}.json"
                safe_json_dump(evaluation_entry, metrics_file)
                
            except Exception as e:
                self.logger.error(f"❌ Evaluation failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
    
    def _generate_comprehensive_predictions(self, model, tokenizer, eval_dataset, num_samples=200):
        """Generate predictions with input tracking and robust error handling"""
        model.eval()
        predictions = []
        references = []
        inputs = []
        
        # Validate tokenizer
        if tokenizer is None:
            self.logger.error("❌ Tokenizer is None in generation function")
            return [], [], []
        
        # Validate tokenizer has required attributes
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            self.logger.warning("⚠️  Setting missing pad_token_id")
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                self.logger.error("❌ Cannot set pad_token_id - no eos_token_id available")
                return [], [], []
        
        # Safe sampling
        sample_indices = np.random.choice(
            len(eval_dataset), 
            min(num_samples, len(eval_dataset)), 
            replace=False
        )
        sample_indices = [int(idx) for idx in sample_indices]
        
        successful_generations = 0
        
        with torch.no_grad():
            for idx in sample_indices:
                try:
                    example = eval_dataset[idx]
                    
                    # Prepare input
                    input_ids = torch.tensor([example['input_ids']]).to(model.device)
                    attention_mask = torch.tensor([example['attention_mask']]).to(model.device)
                    
                    # Generate with robust error handling
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=128,
                        num_beams=2,
                        length_penalty=1.0,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
                        do_sample=False
                    )
                    
                    output_ids = outputs[0].cpu().numpy().tolist()
                    # Decode prediction with proper UTF-8 handling
                    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    prediction = safe_text_encode(prediction)
                    
                    # Decode reference with proper UTF-8 handling
                    label_ids = [l for l in example['labels'] if l != -100]
                    reference = tokenizer.decode(label_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    reference = safe_text_encode(reference)
                    
                    # Decode input with proper UTF-8 handling
                    input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    input_text = safe_text_encode(input_text)
                    
                    # Validate outputs
                    if prediction.strip() and reference.strip() and input_text.strip():
                        predictions.append(prediction)
                        references.append(reference)
                        inputs.append(input_text)
                        successful_generations += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to generate for sample {idx}: {e}")
                    continue
        
        model.train()
        
        self.logger.info(f"✅ Generated {successful_generations}/{len(sample_indices)} successful predictions")
        
        return predictions, references, inputs
    
    def _save_generation_samples(self, inputs, predictions, references, step):
        """Save detailed generation samples with proper UTF-8 encoding"""
        samples = []
        sample_count = min(self.generation_samples, len(predictions))
        
        for i in range(sample_count):
            # Ensure all text is properly encoded
            input_text = safe_text_encode(inputs[i])
            prediction_text = safe_text_encode(predictions[i])
            reference_text = safe_text_encode(references[i])
            
            samples.append({
                'step': step,
                'sample_id': i,
                'input': input_text,
                'prediction': prediction_text,
                'reference': reference_text,
                'input_length': len(input_text.split()),
                'pred_length': len(prediction_text.split()),
                'ref_length': len(reference_text.split())
            })
        
        self.generation_history.extend(samples)
        
        # Save samples for this step with proper encoding
        step_file = self.output_dir / "samples" / f"generation_step_{step}.json"
        safe_json_dump(samples, step_file)
    
    def _log_system_resources(self):
        """Log system resource usage"""
        try:
            memory_info = psutil.virtual_memory()
            resource_entry = {
                'timestamp': datetime.now().isoformat(),
                'memory_percent': memory_info.percent,
                'memory_used_gb': memory_info.used / (1024**3),
                'cpu_percent': psutil.cpu_percent(),
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                    resource_entry[f'gpu_{i}_memory_allocated_gb'] = memory_allocated
                    resource_entry[f'gpu_{i}_memory_cached_gb'] = memory_cached
            
            self.resource_usage.append(resource_entry)
            
        except Exception as e:
            self.logger.debug(f"Could not log resources: {e}")
    
    def save_comprehensive_results(self):
        """Save all collected results with proper UTF-8 encoding"""
        results = {
            'metrics_history': dict(self.metrics_history),
            'best_metrics': self.best_metrics,
            'generation_history': self.generation_history,
            'training_progress': self.training_progress,
            'resource_usage': self.resource_usage,
            'summary': self._create_comprehensive_summary()
        }
        
        # Save main results with proper encoding
        safe_json_dump(results, self.output_dir / "comprehensive_results.json")
        
        # Save as pickle for Python analysis
        with open(self.output_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create visualizations
        self._create_comprehensive_plots()
        
        return results
    
    def _create_comprehensive_summary(self):
        """Create comprehensive training summary"""
        summary = {
            'total_evaluations': len(self.metrics_history.get('bleu', [])),
            'training_steps': len(self.training_progress),
            'best_metrics': {k: v['value'] for k, v in self.best_metrics.items()},
            'final_metrics': {},
            'training_stability': {}
        }
        
        # Final metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                values = [item['value'] for item in history]
                summary['final_metrics'][metric_name] = history[-1]['value']
                summary['training_stability'][f'{metric_name}_std'] = np.std(values)
                summary['training_stability'][f'{metric_name}_trend'] = values[-1] - values[0] if len(values) > 1 else 0
        
        return summary
    
    def _create_comprehensive_plots(self):
        """Create comprehensive visualization suite"""
        try:
            # Training metrics plot
            self._plot_training_metrics()
            
            # Evaluation metrics plot  
            self._plot_evaluation_metrics()
            
            # Resource usage plot
            self._plot_resource_usage()
            
            # Generation quality evolution
            self._plot_generation_quality()
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not create plots: {e}")
    
    def _plot_training_metrics(self):
        """Plot training loss and learning rate"""
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
    
    def _plot_evaluation_metrics(self):
        """Plot evaluation metrics over time"""
        if not self.metrics_history:
            return
            
        key_metrics = ['bleu', 'rouge_l', 'vocab_overlap', 'fluency_score']
        available_metrics = [m for m in key_metrics if m in self.metrics_history]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            if i >= 4:
                break
                
            data = self.metrics_history[metric]
            steps = [item['step'] for item in data]
            values = [item['value'] for item in data]
            
            axes[i].plot(steps, values, 'o-', alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Training Steps')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_usage(self):
        """Plot system resource usage"""
        if not self.resource_usage:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        timestamps = [entry['timestamp'] for entry in self.resource_usage]
        memory_usage = [entry.get('memory_percent', 0) for entry in self.resource_usage]
        cpu_usage = [entry.get('cpu_percent', 0) for entry in self.resource_usage]
        
        if memory_usage:
            ax1.plot(range(len(memory_usage)), memory_usage, 'r-', alpha=0.7)
            ax1.set_title('Memory Usage %')
            ax1.set_xlabel('Time Points')
            ax1.set_ylabel('Memory %')
            ax1.grid(True, alpha=0.3)
        
        if cpu_usage:
            ax2.plot(range(len(cpu_usage)), cpu_usage, 'b-', alpha=0.7)
            ax2.set_title('CPU Usage %')
            ax2.set_xlabel('Time Points')
            ax2.set_ylabel('CPU %')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "resource_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_generation_quality(self):
        """Plot generation quality evolution"""
        if not self.generation_history:
            return
            
        # Group by step
        steps = sorted(set(item['step'] for item in self.generation_history))
        avg_pred_lengths = []
        avg_ref_lengths = []
        
        for step in steps:
            step_samples = [item for item in self.generation_history if item['step'] == step]
            pred_lengths = [item['pred_length'] for item in step_samples]
            ref_lengths = [item['ref_length'] for item in step_samples]
            
            avg_pred_lengths.append(np.mean(pred_lengths) if pred_lengths else 0)
            avg_ref_lengths.append(np.mean(ref_lengths) if ref_lengths else 0)
        
        if steps and avg_pred_lengths:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, avg_pred_lengths, 'b-o', label='Generated', alpha=0.7)
            plt.plot(steps, avg_ref_lengths, 'r--', label='Reference', alpha=0.7)
            plt.title('Generation Length Evolution')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Length (words)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "generation_quality.png", dpi=300, bbox_inches='tight')
            plt.close()

class ResearchPipeline:
    """Complete research pipeline for ProPicto training with proper UTF-8 handling"""
    
    def __init__(self, experiment_name: str = "propicto_research"):
        self.hpc_env = HPCEnvironment()
        self.experiment_name = experiment_name
        
        # Setup paths
        self.paths = self.hpc_env.get_storage_paths(experiment_name)
        self.experiment_dir = self.paths['experiment_dir']
        self.permanent_dir = self.paths['permanent_dir']
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.permanent_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_comprehensive_logging(
            self.experiment_dir / "logs", 
            experiment_name
        )
        
        # Log system info
        self.system_info = self.hpc_env.get_system_info()
        self.logger.info(f"🖥️  System Info: {self.system_info}")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"📱 Using device: {self.device}")
        
    def load_datasets(self, config_name: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load complete datasets with validation and proper UTF-8 handling"""
        self.logger.info(f"📊 Loading complete dataset for {config_name}")
        
        data_root = Path("data/processed_propicto")
        config_path = data_root / config_name
        
        # Validate paths exist
        for split in ['train', 'valid', 'test']:
            split_path = config_path / split / "data.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing {split} data: {split_path}")
        
        # Load all splits with proper UTF-8 handling
        datasets = {}
        for split in ['train', 'valid', 'test']:
            split_path = config_path / split / "data.json"
            datasets[split] = safe_json_load(split_path)
            self.logger.info(f"   {split}: {len(datasets[split]):,} samples")
        
        # Data quality validation
        self._validate_dataset_quality(datasets, config_name)
        
        return datasets['train'], datasets['valid'], datasets['test']
    
    def _validate_dataset_quality(self, datasets: Dict[str, List], config_name: str):
        """Validate dataset quality and log statistics"""
        self.logger.info(f"🔍 Validating dataset quality for {config_name}")
        
        quality_report = {
            'config_name': config_name,
            'validation_timestamp': datetime.now().isoformat(),
            'splits': {}
        }
        
        for split_name, data in datasets.items():
            split_stats = {
                'total_samples': len(data),
                'valid_samples': 0,
                'empty_inputs': 0,
                'empty_targets': 0,
                'input_length_stats': {},
                'target_length_stats': {}
            }
            
            input_lengths = []
            target_lengths = []
            
            for item in data:
                input_text = safe_text_encode(item.get('input_text', '')).strip()
                target_text = safe_text_encode(item.get('target_text', '')).strip()
                
                if not input_text:
                    split_stats['empty_inputs'] += 1
                    continue
                if not target_text:
                    split_stats['empty_targets'] += 1
                    continue
                
                split_stats['valid_samples'] += 1
                input_lengths.append(len(input_text.split()))
                target_lengths.append(len(target_text.split()))
            
            if input_lengths:
                split_stats['input_length_stats'] = {
                    'mean': float(np.mean(input_lengths)),
                    'std': float(np.std(input_lengths)),
                    'min': int(min(input_lengths)),
                    'max': int(max(input_lengths)),
                    'median': float(np.median(input_lengths))
                }
                
                split_stats['target_length_stats'] = {
                    'mean': float(np.mean(target_lengths)),
                    'std': float(np.std(target_lengths)),
                    'min': int(min(target_lengths)),
                    'max': int(max(target_lengths)),
                    'median': float(np.median(target_lengths))
                }
            
            quality_report['splits'][split_name] = split_stats
            
            # Log summary
            success_rate = split_stats['valid_samples'] / split_stats['total_samples'] * 100
            self.logger.info(f"   {split_name}: {success_rate:.1f}% valid samples")
            
            if split_stats['input_length_stats']:
                mean_input = split_stats['input_length_stats']['mean']
                mean_target = split_stats['target_length_stats']['mean']
                self.logger.info(f"   {split_name}: avg {mean_input:.1f} input, {mean_target:.1f} target words")
        
        # Save quality report with proper encoding
        quality_file = self.experiment_dir / f"dataset_quality_{config_name}.json"
        safe_json_dump(quality_report, quality_file)
    
    def setup_model_and_tokenizer(self, model_choice: str):
        """Setup model and tokenizer with comprehensive logging"""
        self.logger.info(f"set up model: {model_choice}")
        
        model_configs = {
            'barthez': {
                'model_name': 'moussaKam/barthez',
                'description': 'French BART model, optimized for French generation tasks'
            },
            'french_t5': {
                'model_name': 'plguillou/t5-base-fr-sum-cnndm',
                'description': 'French T5 model, fine-tuned for French summarization'
            }
        }
        
        if model_choice not in model_configs:
            raise ValueError(f"Unknown model choice: {model_choice}")
        
        config = model_configs[model_choice]
        model_name = config['model_name']
        
        # Load model and tokenizer
        self.logger.info(f"📥 Loading {model_name}")
        
        from transformers import AutoTokenizer, T5Tokenizer

        try:
            # Try fast tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            print(f"[WARN] Fast tokenizer failed for {model_name}: {e}")
            if "t5" in model_name.lower():
                tokenizer = T5Tokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info("🔧 Set pad_token to eos_token")
        
        # FIX: Set up proper generation config to eliminate warning
        generation_config = GenerationConfig.from_model_config(model.config)
        generation_config.early_stopping = True
        generation_config.num_beams = 4
        generation_config.no_repeat_ngram_size = 3
        generation_config.length_penalty = 1.2
        generation_config.max_length = 128
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config = generation_config
        self.logger.info("✅ Generation config properly set - warnings eliminated")
        
        model.to(self.device)
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model_choice': model_choice,
            'model_name': model_name,
            'description': config['description'],
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'vocab_size': tokenizer.vocab_size,
            'model_type': type(model).__name__,
            'tokenizer_type': type(tokenizer).__name__
        }
        
        self.logger.info(f"✅ Model loaded: {param_count:,} parameters ({trainable_params:,} trainable)")
        
        # Save model info with proper encoding
        safe_json_dump(model_info, self.experiment_dir / "model_info.json")
        
        return model, tokenizer, model_info
    
    def prepare_datasets(self, train_data, valid_data, test_data, tokenizer, config_name, max_samples=None):
        """Prepare datasets with comprehensive logging and proper UTF-8 handling"""
        self.logger.info(f"🔧 Preparing datasets for {config_name}")
        
        # Apply sample limit if specified
        original_train_size = len(train_data)
        if max_samples and max_samples < len(train_data):
            train_data = train_data[:max_samples]
            self.logger.info(f"🧪 Limited training data: {len(train_data):,} / {original_train_size:,} samples")
        
        # Task prefix function
        def get_task_prefix(config_name: str, input_text: str) -> str:
            # Ensure input text is properly encoded
            input_text = safe_text_encode(input_text)
            
            prefixes = {
                'keywords_to_sentence': lambda x: f"Corriger et compléter: {x.replace('mots:', '').strip()}",
                'pictos_tokens_to_sentence': lambda x: f"Transformer pictogrammes: {x.replace('tokens:', '').strip()}",
                'hybrid_to_sentence': lambda x: f"Transformer texte mixte: {x.replace('hybrid:', '').strip()}",
                'direct_to_sentence': lambda x: f"Corriger texte: {x.replace('direct:', '').strip()}"
            }
            
            if config_name in prefixes:
                return prefixes[config_name](input_text)
            else:
                return f"Transformer: {input_text}"
        
        # Tokenization function with UTF-8 handling
        def tokenize_function(examples):
            inputs = []
            targets = []
            
            for input_text, target_text in zip(examples['input_text'], examples['target_text']):
                # Ensure proper UTF-8 encoding
                input_text = safe_text_encode(input_text)
                target_text = safe_text_encode(target_text)
                
                french_input = get_task_prefix(config_name, input_text)
                inputs.append(french_input)
                targets.append(target_text)
            
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length',return_tensors=None)
            labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length',return_tensors=None)
            
            model_inputs["labels"] = [
                [t if t != tokenizer.pad_token_id else -100 for t in label_ids]
                for label_ids in labels["input_ids"]
            ]
            
            return model_inputs
        
        # Show tokenization example
        if train_data:
            sample = train_data[0]
            example_input = get_task_prefix(config_name, sample['input_text'])
            self.logger.info(f"🔍 Task formulation example:")
            self.logger.info(f"   Original: {safe_text_encode(sample['input_text'])}")
            self.logger.info(f"   French:   {example_input}")
            self.logger.info(f"   Target:   {safe_text_encode(sample['target_text'])}")
        
        # Create datasets
        datasets = {}
        for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            if data:
                dataset = Dataset.from_list(data)
                tokenized_dataset = dataset.map(
                    tokenize_function, 
                    batched=True, 
                    remove_columns=dataset.column_names
                )
                datasets[name] = tokenized_dataset
                self.logger.info(f"✅ {name} dataset: {len(tokenized_dataset):,} samples")
        
        return datasets['train'], datasets['valid'], datasets.get('test')
    
    def run_experiment(self, config_name: str, model_choice: str = 'barthez', 
                      max_samples: Optional[int] = None, num_epochs: int = 5,
                      test_run: bool = False):
        """Run complete research experiment with proper UTF-8 handling"""
        
        # Create experiment configuration
        experiment_config = {
            'experiment_name': self.experiment_name,
            'config_name': config_name,
            'model_choice': model_choice,
            'max_samples': max_samples,
            'num_epochs': num_epochs,
            'test_run': test_run,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.paths['timestamp'],
            'system_info': self.system_info,
            'paths': {k: str(v) for k, v in self.paths.items()}
        }
        
        self.logger.info(f"🚀 Starting research experiment: {experiment_config['experiment_id']}")
        self.logger.info(f"📋 Configuration: {config_name} with {model_choice}")
        
        if test_run:
            self.logger.info("🧪 TEST RUN MODE - Limited samples and epochs")
            max_samples = min(max_samples or 1000, 1000)
            num_epochs = min(num_epochs, 2)
        
        # Save experiment config with proper encoding
        config_file = self.experiment_dir / "experiment_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(experiment_config, f, default_flow_style=False, allow_unicode=True)
        
        try:
            # Load datasets
            train_data, valid_data, test_data = self.load_datasets(config_name)
            
            # Setup model
            model, tokenizer, model_info = self.setup_model_and_tokenizer(model_choice)
            
            # Prepare datasets
            train_dataset, valid_dataset, test_dataset = self.prepare_datasets(
                train_data, valid_data, test_data, tokenizer, config_name, max_samples
            )
            
            # Setup evaluation
            evaluator = ResearchEvaluator(tokenizer)
            
            # Setup callback - PASS TOKENIZER
            callback = ResearchCallback(
                evaluator=evaluator,
                eval_dataset=valid_dataset,
                output_dir=self.experiment_dir,
                config_name=config_name,
                tokenizer=tokenizer,  # Pass tokenizer explicitly
                generation_samples=15
            )
            
            # Calculate training parameters
            batch_size = 8 if not test_run else 4
            steps_per_epoch = len(train_dataset) // batch_size
            eval_steps = max(50, steps_per_epoch // 4)
            save_steps = eval_steps
            total_steps = steps_per_epoch * num_epochs
            
            self.logger.info(f"📊 Training plan:")
            self.logger.info(f"   Steps per epoch: {steps_per_epoch}")
            self.logger.info(f"   Total steps: {total_steps}")
            self.logger.info(f"   Eval every: {eval_steps} steps")
            self.logger.info(f"   Batch size: {batch_size}")
            
            # Training arguments - VERSION COMPATIBLE
            try:
                # Test which parameter name is supported
                test_args = TrainingArguments(
                    output_dir="./test_eval_strategy",
                    eval_strategy="no"
                )
                eval_strategy_param = "eval_strategy"
                self.logger.info("✅ Using 'eval_strategy' (newer Transformers)")
            except TypeError:
                eval_strategy_param = "evaluation_strategy"
                self.logger.info("✅ Using 'evaluation_strategy' (older Transformers)")
            
            # Build compatible arguments
            training_args_dict = {
                "output_dir": str(self.experiment_dir / "checkpoints"),
                "num_train_epochs": num_epochs,
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": batch_size,
                "gradient_accumulation_steps": 1,
                "learning_rate": 3e-5,
                "weight_decay": 0.01,
                "warmup_steps": min(500, total_steps // 10),
                "logging_steps": max(10, steps_per_epoch // 10),
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                eval_strategy_param: "steps",  # Use detected parameter name
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "save_total_limit": 3,
                "fp16": False,
                "dataloader_pin_memory": False,
                "report_to": [],
                "remove_unused_columns": True,
                "run_name": f"{config_name}_{model_choice}_{experiment_config['experiment_id']}",
                "logging_dir": str(self.experiment_dir / "tensorboard"),
                "save_safetensors": True
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
            self.logger.info("🏋️ Starting training...")
            start_time = time.time()
            
            trainer.train()
            
            training_time = time.time() - start_time
            self.logger.info(f"⏱️  Training completed in {training_time/3600:.2f} hours")
            
            # Final test evaluation
            final_results = {'experiment_config': experiment_config}
            
            if test_dataset:
                self.logger.info("🧪 Running final test evaluation...")
                test_predictions, test_references, test_inputs = callback._generate_comprehensive_predictions(
                    model, tokenizer, test_dataset, num_samples=min(1000, len(test_dataset))
                )
                
                if test_predictions and test_references:
                    test_metrics = evaluator.evaluate_predictions(test_predictions, test_references)
                    final_results['test_metrics'] = test_metrics
                    
                    self.logger.info("🧪 Test Results:")
                    for metric_name, value in test_metrics.items():
                        if metric_name in ['bleu', 'rouge_l', 'vocab_overlap', 'generation_success_rate']:
                            self.logger.info(f"   {metric_name}: {value:.4f}")
                    
                    # Save test predictions with proper UTF-8 encoding
                    test_predictions_data = []
                    sample_size = min(100, len(test_predictions))
                    for i in range(sample_size):
                        test_predictions_data.append({
                            'sample_id': i,
                            'input': safe_text_encode(test_inputs[i]),
                            'prediction': safe_text_encode(test_predictions[i]),
                            'reference': safe_text_encode(test_references[i])
                        })
                    
                    safe_json_dump(test_predictions_data, self.experiment_dir / "test_predictions.json")
            
            # Save final model to permanent storage
            final_model_dir = self.permanent_dir / "final_model"
            trainer.save_model(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            
            # Save comprehensive results
            training_results = callback.save_comprehensive_results()
            
            # Compile final results
            final_results.update({
                'model_info': model_info,
                'training_time_hours': training_time / 3600,
                'training_results': training_results,
                'dataset_sizes': {
                    'train': len(train_dataset),
                    'valid': len(valid_dataset),
                    'test': len(test_dataset) if test_dataset else 0
                },
                'model_path': str(final_model_dir),
                'experiment_path': str(self.experiment_dir),
                'permanent_path': str(self.permanent_dir)
            })
            
            # Save final results with proper encoding
            safe_json_dump(final_results, self.permanent_dir / "final_results.json")
            
            # Copy important results to permanent storage
            self._archive_results()
            
            # Log final summary
            self._log_final_summary(final_results)
            
            return self.permanent_dir, final_results
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _archive_results(self):
        """Archive important results to permanent storage"""
        self.logger.info("📦 Archiving results to permanent storage...")
        
        # Files to archive
        important_files = [
            "experiment_config.yaml",
            "comprehensive_results.json",
            "dataset_quality_*.json",
            "model_info.json",
            "*.png",
            "test_predictions.json"
        ]
        
        for pattern in important_files:
            for file_path in self.experiment_dir.glob(pattern):
                if file_path.is_file():
                    dest_path = self.permanent_dir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    self.logger.debug(f"Archived: {file_path.name}")
        
        # Archive logs directory
        logs_src = self.experiment_dir / "logs"
        if logs_src.exists():
            logs_dest = self.permanent_dir / "logs"
            if logs_dest.exists():
                shutil.rmtree(logs_dest)
            shutil.copytree(logs_src, logs_dest)
        
        self.logger.info(f"✅ Results archived to: {self.permanent_dir}")
    
    def _log_final_summary(self, results):
        """Log comprehensive final summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎉 RESEARCH EXPERIMENT COMPLETED")
        self.logger.info("="*80)
        
        config = results['experiment_config']
        self.logger.info(f"📋 Experiment: {config['config_name']} with {config['model_choice']}")
        self.logger.info(f"🆔 ID: {config['experiment_id']}")
        self.logger.info(f"⏱️  Training time: {results['training_time_hours']:.2f} hours")
        
        # Dataset info
        sizes = results['dataset_sizes']
        self.logger.info(f"📊 Dataset: {sizes['train']:,} train, {sizes['valid']:,} valid, {sizes['test']:,} test")
        
        # Model info
        model_info = results['model_info']
        self.logger.info(f"🤖 Model: {model_info['total_parameters']:,} parameters")
        
        # Best metrics
        training_results = results['training_results']
        best_metrics = training_results.get('best_metrics', {})
        
        if best_metrics:
            self.logger.info("🏆 Best training metrics:")
            for metric_name, metric_info in best_metrics.items():
                if metric_name in ['bleu', 'rouge_l', 'vocab_overlap', 'generation_success_rate']:
                    self.logger.info(f"   {metric_name}: {metric_info['value']:.4f} (step {metric_info['step']})")
        
        # Test results
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            self.logger.info("🧪 Final test metrics:")
            for metric_name, value in test_metrics.items():
                if metric_name in ['bleu', 'rouge_l', 'vocab_overlap', 'generation_success_rate']:
                    self.logger.info(f"   {metric_name}: {value:.4f}")
        
        # Paths
        self.logger.info(f"📁 Results: {results['permanent_path']}")
        self.logger.info(f"🤖 Model: {results['model_path']}")
        
        # Research recommendations
        self._log_research_recommendations(results)
        
        self.logger.info("="*80)
    
    def _log_research_recommendations(self, results):
        """Log research insights and recommendations"""
        self.logger.info("\n📝 Research Insights:")
        
        training_results = results.get('training_results', {})
        best_metrics = training_results.get('best_metrics', {})
        test_metrics = results.get('test_metrics', {})
        
        # BLEU analysis
        if 'bleu' in best_metrics and 'bleu' in test_metrics:
            train_bleu = best_metrics['bleu']['value']
            test_bleu = test_metrics['bleu']
            
            if test_bleu < train_bleu * 0.8:
                self.logger.info("   ⚠️  Potential overfitting detected (test BLEU much lower than train)")
            elif test_bleu >= 0.3:
                self.logger.info("   ✅ Good BLEU performance on test set")
            
        # Generation success rate
        if 'generation_success_rate' in test_metrics:
            success_rate = test_metrics['generation_success_rate']
            if success_rate >= 0.8:
                self.logger.info("   ✅ High generation success rate - model produces valid outputs")
            elif success_rate >= 0.5:
                self.logger.info("   ⚠️  Moderate generation success - consider more training")
            else:
                self.logger.info("   ❌ Low generation success - model needs significant improvement")
        
        # Vocabulary overlap
        if 'vocab_overlap' in test_metrics:
            vocab_overlap = test_metrics['vocab_overlap']
            if vocab_overlap >= 0.6:
                self.logger.info("   ✅ Good vocabulary overlap with references")
            else:
                self.logger.info("   📝 Consider improving vocabulary alignment")

def main():
    """Main function with proper UTF-8 setup"""
    # Ensure UTF-8 environment is set up before anything else
    setup_utf8_environment()
    
    parser = argparse.ArgumentParser(description='Complete Research Pipeline for ProPicto Training')
    parser.add_argument('--config', required=True,
                       choices=['keywords_to_sentence', 'pictos_tokens_to_sentence', 
                               'hybrid_to_sentence', 'direct_to_sentence'],
                       help='Data configuration to use')
    parser.add_argument('--model', choices=['barthez', 'french_t5'], 
                       default='barthez', help='French model to use')
    parser.add_argument('--max-samples', type=int, 
                       help='Limit training samples (for experimentation)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--experiment-name', default='propicto_research',
                       help='Experiment name for organization')
    parser.add_argument('--test-run', action='store_true',
                       help='Run in test mode with limited data and epochs')
    
    args = parser.parse_args()
    
    print("🔬 Complete Research Pipeline for ProPicto Training")
    print("=" * 60)
    print("✅ HPC optimized with VSC_SCRATCH integration")
    print("✅ Comprehensive experiment tracking and metrics")
    print("✅ Scalable to full datasets with proper resource management")
    print("✅ Research-grade logging and result archival")
    print("✅ Built-in test runs and validation")
    print("✅ FIXED: Proper UTF-8 encoding throughout pipeline")
    
    # Test UTF-8 encoding
    test_french = "Test français: café, être, déjà, naïve, mémoire"
    print(f"🔤 UTF-8 test: {test_french}")
    
    if args.test_run:
        print("🧪 TEST RUN MODE - Limited samples and epochs for quick validation")
    
    # Initialize pipeline
    pipeline = ResearchPipeline(args.experiment_name)
    
    try:
        # Run experiment
        results_dir, results = pipeline.run_experiment(
            config_name=args.config,
            model_choice=args.model,
            max_samples=args.max_samples,
            num_epochs=args.epochs,
            test_run=args.test_run
        )
        
        print(f"\n🎉 RESEARCH EXPERIMENT COMPLETED!")
        print(f"📁 Results: {results_dir}")
        print(f"🤖 Model: {results['model_path']}")
        
        # Show key metrics
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            print(f"\n📊 Key Results:")
            for metric in ['bleu', 'rouge_l', 'generation_success_rate']:
                if metric in test_metrics:
                    print(f"   {metric}: {test_metrics[metric]:.3f}")
        
        print(f"configurationreport:")
        print(f"   - Expe ID: {results['experiment_config']['experiment_id']}")
        print(f"   - Configuration: {args.config} with {args.model}")
        print(f"   - Training samples: {results['dataset_sizes']['train']:,}")
        print(f"   - Training time: {results['training_time_hours']:.2f} hours")

        
    except Exception as e:
        print(f"\EXPERIMENT FAILED: {e}")
        print("see logs ")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()