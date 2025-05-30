#!/usr/bin/env python3
"""
Simple Production ProPicto Trainer
- Uses ALL available data
- Comprehensive logging
- Built-in evaluation metrics (no external dependencies)
- Works with any environment
"""

import logging
import torch
import json
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import math

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, TrainerCallback
)
from datasets import Dataset

def setup_logging(log_dir: str = "logs"):
    """Setup comprehensive logging"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"propicto_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging to: {log_file}")
    return logger

class SimpleEvaluator:
    """Simple built-in evaluator with no external dependencies"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics using built-in implementations"""
        results = {}
        
        # Basic length statistics
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        results.update({
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
            'length_std': np.std(pred_lengths)
        })
        
        # Built-in BLEU score
        results['bleu'] = self._calculate_bleu(predictions, references)
        
        # Built-in ROUGE-L score
        results['rouge_l'] = self._calculate_rouge_l(predictions, references)
        
        # Custom French-specific metrics
        results.update(self._calculate_french_metrics(predictions, references))
        
        return results
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Built-in BLEU calculation"""
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
                
                if total == 0:
                    scores.append(0.0)
                else:
                    scores.append(overlap / total)
            
            if all(s > 0 for s in scores):
                bleu = math.exp(sum(math.log(s) for s in scores) / len(scores))
                # Brevity penalty
                bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
                total_score += bp * bleu
                valid_count += 1
        
        return total_score / valid_count if valid_count > 0 else 0.0
    
    def _calculate_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """Built-in ROUGE-L calculation"""
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
                rouge_scores.append(0.0)
                continue
            
            lcs_len = lcs_length(pred_tokens, ref_tokens)
            
            if len(pred_tokens) == 0:
                precision = 0.0
            else:
                precision = lcs_len / len(pred_tokens)
            
            recall = lcs_len / len(ref_tokens)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            rouge_scores.append(f1)
        
        return np.mean(rouge_scores) if rouge_scores else 0.0
    
    def _calculate_french_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate French-specific quality metrics"""
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
        
        # French article usage
        french_articles = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'des'}
        article_usage = []
        for pred in predictions:
            pred_words = set(pred.lower().split())
            article_count = len(pred_words & french_articles)
            total_words = len(pred.split())
            if total_words > 0:
                article_ratio = article_count / total_words
                article_usage.append(article_ratio)
        
        metrics['french_article_ratio'] = np.mean(article_usage) if article_usage else 0.0
        
        # Fluency (no repeated tokens)
        fluency_scores = []
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                fluency_scores.append(unique_ratio)
        
        metrics['fluency_score'] = np.mean(fluency_scores) if fluency_scores else 0.0
        
        # Average sentence length
        avg_lengths = []
        for pred in predictions:
            avg_lengths.append(len(pred.split()))
        
        metrics['avg_sentence_length'] = np.mean(avg_lengths) if avg_lengths else 0.0
        
        return metrics

class SimpleTrainingCallback(TrainerCallback):
    """Simple training callback with built-in evaluation"""
    
    def __init__(self, evaluator, eval_dataset, output_dir, generation_samples=5):
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.generation_samples = generation_samples
        self.logger = logging.getLogger(__name__)
        
        # Tracking
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.generation_history = []
        
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        """Enhanced evaluation with generation testing"""
        
        if state.global_step % (args.eval_steps * 2) == 0:  # Less frequent for performance
            self.logger.info(f"📊 Running evaluation at step {state.global_step}")
            
            try:
                # Generate predictions on eval samples
                predictions, references = self._generate_eval_predictions(
                    model, tokenizer, self.eval_dataset, num_samples=min(100, len(self.eval_dataset))
                )
                
                # Validate predictions and references
                if not predictions or not references:
                    self.logger.warning("⚠️  No predictions generated, skipping evaluation")
                    return
                
                if len(predictions) != len(references):
                    self.logger.warning(f"⚠️  Prediction count mismatch: {len(predictions)} vs {len(references)}")
                    min_len = min(len(predictions), len(references))
                    predictions = predictions[:min_len]
                    references = references[:min_len]
                
                # Calculate metrics
                metrics = self.evaluator.evaluate_predictions(predictions, references)
                
                # Log metrics
                self.logger.info("📈 Evaluation Metrics:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.metrics_history[metric_name].append({
                            'step': state.global_step,
                            'epoch': state.epoch,
                            'value': float(value)  # Ensure serializable
                        })
                        self.logger.info(f"   {metric_name}: {value:.4f}")
                
                # Track best metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]['value']:
                            self.best_metrics[metric_name] = {
                                'value': float(value),
                                'step': state.global_step,
                                'epoch': state.epoch
                            }
                
                # Save sample generations
                self._save_generation_samples(predictions, references, state.global_step)
                
            except Exception as e:
                self.logger.error(f"❌ Evaluation failed: {e}")
                # Continue training even if evaluation fails
                import traceback
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    def _generate_eval_predictions(self, model, tokenizer, eval_dataset, num_samples=100):
        """Generate predictions for evaluation"""
        model.eval()
        predictions = []
        references = []
        
        # FIX: Convert numpy integers to regular Python integers
        sample_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
        sample_indices = [int(idx) for idx in sample_indices]  # Convert to Python int
        
        with torch.no_grad():
            for idx in sample_indices:
                example = eval_dataset[idx]
                
                # Prepare input
                input_ids = torch.tensor([example['input_ids']]).to(model.device)
                attention_mask = torch.tensor([example['attention_mask']]).to(model.device)
                
                # Generate
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=2,
                    length_penalty=1.0,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
                
                # Decode prediction
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Decode reference
                label_ids = [l for l in example['labels'] if l != -100]
                reference = tokenizer.decode(label_ids, skip_special_tokens=True)
                
                predictions.append(prediction)
                references.append(reference)
        
        model.train()
        return predictions, references
    
    def _save_generation_samples(self, predictions, references, step):
        """Save generation samples for manual inspection"""
        samples = []
        for i, (pred, ref) in enumerate(zip(predictions[:self.generation_samples], references[:self.generation_samples])):
            samples.append({
                'step': step,
                'sample_id': i,
                'prediction': pred,
                'reference': ref
            })
        
        self.generation_history.extend(samples)
        
        # Save to file
        with open(self.output_dir / "generation_samples.json", 'w', encoding='utf-8') as f:
            json.dump(self.generation_history, f, ensure_ascii=False, indent=2)
    
    def save_training_results(self):
        """Save comprehensive training results"""
        results = {
            'metrics_history': dict(self.metrics_history),
            'best_metrics': self.best_metrics,
            'generation_history': self.generation_history,
            'summary': self._create_summary()
        }
        
        with open(self.output_dir / "comprehensive_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Create plots
        self._create_evaluation_plots()
        
        return results
    
    def _create_summary(self):
        """Create training summary"""
        summary = {
            'total_evaluations': len(self.metrics_history.get('bleu', [])),
            'best_bleu': self.best_metrics.get('bleu', {}).get('value', 0.0),
            'best_rouge_l': self.best_metrics.get('rouge_l', {}).get('value', 0.0),
            'final_metrics': {}
        }
        
        # Final metrics (last evaluation)
        for metric_name, history in self.metrics_history.items():
            if history:
                summary['final_metrics'][metric_name] = history[-1]['value']
        
        return summary
    
    def _create_evaluation_plots(self):
        """Create evaluation plots"""
        if not self.metrics_history:
            return
        
        try:
            # Plot key metrics over time
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # BLEU score
            if 'bleu' in self.metrics_history:
                bleu_data = self.metrics_history['bleu']
                steps = [item['step'] for item in bleu_data]
                values = [item['value'] for item in bleu_data]
                axes[0, 0].plot(steps, values, 'b-', marker='o')
                axes[0, 0].set_title('BLEU Score')
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('BLEU')
                axes[0, 0].grid(True, alpha=0.3)
            
            # ROUGE-L score
            if 'rouge_l' in self.metrics_history:
                rouge_data = self.metrics_history['rouge_l']
                steps = [item['step'] for item in rouge_data]
                values = [item['value'] for item in rouge_data]
                axes[0, 1].plot(steps, values, 'r-', marker='o')
                axes[0, 1].set_title('ROUGE-L Score')
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('ROUGE-L')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Length ratio
            if 'length_ratio' in self.metrics_history:
                length_data = self.metrics_history['length_ratio']
                steps = [item['step'] for item in length_data]
                values = [item['value'] for item in length_data]
                axes[1, 0].plot(steps, values, 'g-', marker='o')
                axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                axes[1, 0].set_title('Length Ratio (Pred/Ref)')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Ratio')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Vocabulary overlap
            if 'vocab_overlap' in self.metrics_history:
                vocab_data = self.metrics_history['vocab_overlap']
                steps = [item['step'] for item in vocab_data]
                values = [item['value'] for item in vocab_data]
                axes[1, 1].plot(steps, values, 'm-', marker='o')
                axes[1, 1].set_title('Vocabulary Overlap')
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Overlap Ratio')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not create plots: {e}")

class SimpleProPictoTrainer:
    """Simple production trainer with no external dependencies"""
    
    def __init__(self, data_root="data/processed_propicto", results_dir="results"):
        self.logger = setup_logging()
        self.data_root = Path(data_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create experiment directory
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🧪 Experiment ID: {self.experiment_id}")
        self.logger.info(f"📁 Results directory: {self.experiment_dir}")
        
    def load_full_dataset(self, config_name: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load ALL data for a configuration"""
        config_path = self.data_root / config_name
        
        # Load all splits
        train_path = config_path / "train" / "data.json"
        valid_path = config_path / "valid" / "data.json"
        test_path = config_path / "test" / "data.json"
        
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(valid_path, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            
        test_data = []
        if test_path.exists():
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        
        self.logger.info(f"📊 Full dataset loaded for {config_name}:")
        self.logger.info(f"   Train: {len(train_data):,} samples")
        self.logger.info(f"   Valid: {len(valid_data):,} samples") 
        self.logger.info(f"   Test:  {len(test_data):,} samples")
        
        return train_data, valid_data, test_data
    
    def train_production_model(self, config_name: str, model_choice: str = 'barthez',
                              max_train_samples: Optional[int] = None,
                              num_epochs: int = 5):
        """Train production model with full evaluation"""
        
        experiment_config = {
            'config_name': config_name,
            'model_choice': model_choice,
            'max_train_samples': max_train_samples,
            'num_epochs': num_epochs,
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Save experiment config
        with open(self.experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        self.logger.info(f"🚀 Production training: {config_name}")
        self.logger.info(f"📋 Config: {experiment_config}")
        
        # Load model
        self.logger.info(f"🤖 Loading {model_choice}...")
        if model_choice == 'barthez':
            tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
            model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez")
        else:  # french_t5
            tokenizer = AutoTokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
            model = AutoModelForSeq2SeqLM.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(self.device)
        
        # Load FULL dataset
        train_data, valid_data, test_data = self.load_full_dataset(config_name)
        
        # Optionally limit training data
        if max_train_samples and max_train_samples < len(train_data):
            self.logger.info(f"🧪 Limiting training to {max_train_samples:,} samples")
            train_data = train_data[:max_train_samples]
        
        # Tokenize datasets
        train_dataset = self._tokenize_dataset(train_data, tokenizer, config_name)
        valid_dataset = self._tokenize_dataset(valid_data, tokenizer, config_name)
        test_dataset = self._tokenize_dataset(test_data, tokenizer, config_name) if test_data else None
        
        # Setup evaluation
        evaluator = SimpleEvaluator(tokenizer)
        
        # Setup callbacks
        training_callback = SimpleTrainingCallback(
            evaluator=evaluator,
            eval_dataset=valid_dataset,
            output_dir=self.experiment_dir,
            generation_samples=10
        )
        
        # Training arguments for production
        steps_per_epoch = len(train_dataset) // 8  # Assuming batch size 8
        eval_steps = max(50, steps_per_epoch // 4)  # Evaluate 4 times per epoch
        save_steps = eval_steps
        total_steps = steps_per_epoch * num_epochs
        
        training_args = TrainingArguments(
            output_dir=str(self.experiment_dir / "checkpoints"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_steps=min(500, total_steps // 10),
            logging_steps=max(10, steps_per_epoch // 10),
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=3,
            fp16=False,
            dataloader_pin_memory=False,
            report_to=[],
            remove_unused_columns=True,
            run_name=f"{config_name}_{model_choice}_{self.experiment_id}"
        )
        
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
                training_callback
            ]
        )
        
        # Train
        self.logger.info(f"🏋️ Starting production training...")
        self.logger.info(f"   Total steps: {total_steps:,}")
        self.logger.info(f"   Eval every: {eval_steps} steps")
        
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Final evaluation on test set
        final_results = {}
        if test_dataset:
            self.logger.info("🧪 Final evaluation on test set...")
            test_predictions, test_references = training_callback._generate_eval_predictions(
                model, tokenizer, test_dataset, num_samples=len(test_dataset)
            )
            final_results['test_metrics'] = evaluator.evaluate_predictions(test_predictions, test_references)
            
            # Save test predictions
            test_results = []
            for i, (pred, ref) in enumerate(zip(test_predictions, test_references)):
                test_results.append({
                    'sample_id': i,
                    'prediction': pred,
                    'reference': ref
                })
            
            with open(self.experiment_dir / "test_predictions.json", 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        # Save final model
        final_model_dir = self.experiment_dir / "final_model"
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Save comprehensive results
        training_results = training_callback.save_training_results()
        
        # Final summary
        final_results.update({
            'experiment_config': experiment_config,
            'training_time_hours': training_time / 3600,
            'training_results': training_results,
            'model_path': str(final_model_dir),
            'dataset_sizes': {
                'train': len(train_dataset),
                'valid': len(valid_dataset),
                'test': len(test_dataset) if test_dataset else 0
            }
        })
        
        with open(self.experiment_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Log summary
        self._log_final_summary(final_results)
        
        return self.experiment_dir, final_results
    
    def _tokenize_dataset(self, data, tokenizer, config_name):
        """Tokenize dataset with proper task formatting"""
        def get_task_prefix(config_name: str, input_text: str):
            if config_name == 'keywords_to_sentence':
                clean_input = input_text.replace("mots:", "").strip()
                return f"Corriger et compléter: {clean_input}"
            elif config_name == 'pictos_tokens_to_sentence':
                clean_input = input_text.replace("tokens:", "").strip()
                return f"Transformer pictogrammes: {clean_input}"
            elif config_name == 'hybrid_to_sentence':
                clean_input = input_text.replace("hybrid:", "").strip()
                return f"Transformer texte mixte: {clean_input}"
            elif config_name == 'direct_to_sentence':
                clean_input = input_text.replace("direct:", "").strip()
                return f"Corriger texte: {clean_input}"
            else:
                return f"Transformer: {input_text}"
        
        def tokenize_function(examples):
            inputs = []
            targets = []
            
            for input_text, target_text in zip(examples['input_text'], examples['target_text']):
                french_input = get_task_prefix(config_name, input_text)
                inputs.append(french_input)
                targets.append(target_text)
            
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
            labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
            
            model_inputs["labels"] = [
                [t if t != tokenizer.pad_token_id else -100 for t in label_ids]
                for label_ids in labels["input_ids"]
            ]
            
            return model_inputs
        
        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        return tokenized_dataset
    
    def _log_final_summary(self, results):
        """Log comprehensive final summary"""
        self.logger.info(f"\n🎉 PRODUCTION TRAINING COMPLETE")
        self.logger.info("=" * 60)
        
        config = results['experiment_config']
        self.logger.info(f"📋 Configuration: {config['config_name']} with {config['model_choice']}")
        self.logger.info(f"🕐 Training time: {results['training_time_hours']:.2f} hours")
        
        # Dataset info
        sizes = results['dataset_sizes']
        self.logger.info(f"📊 Data: {sizes['train']:,} train, {sizes['valid']:,} valid, {sizes['test']:,} test")
        
        # Best metrics
        training_results = results['training_results']
        best_metrics = training_results.get('best_metrics', {})
        
        self.logger.info("🏆 Best metrics during training:")
        for metric_name, metric_info in best_metrics.items():
            if metric_name in ['bleu', 'rouge_l', 'vocab_overlap', 'fluency_score']:
                self.logger.info(f"   {metric_name}: {metric_info['value']:.4f} (step {metric_info['step']})")
        
        # Test results
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            self.logger.info("🧪 Final test metrics:")
            for metric_name, value in test_metrics.items():
                if metric_name in ['bleu', 'rouge_l', 'vocab_overlap', 'fluency_score']:
                    self.logger.info(f"   {metric_name}: {value:.4f}")
        
        self.logger.info(f"📁 Results saved to: {results.get('model_path', 'N/A')}")
        self.logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Simple Production ProPicto Training')
    parser.add_argument('--config', required=True,
                       choices=['keywords_to_sentence', 'pictos_tokens_to_sentence', 
                               'hybrid_to_sentence', 'direct_to_sentence'],
                       help='Data configuration to use')
    parser.add_argument('--model', choices=['barthez', 'french_t5'], 
                       default='barthez', help='French model to use')
    parser.add_argument('--max-samples', type=int, 
                       help='Limit training samples (for experimentation)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    
    args = parser.parse_args()
    
    print("🏭 Simple Production ProPicto Trainer")
    print("=" * 50)
    print("✅ Uses ALL available data")
    print("✅ Built-in evaluation (BLEU, ROUGE-L, French metrics)")
    print("✅ Comprehensive logging and result tracking")
    print("✅ No external dependencies")
    
    trainer = SimpleProPictoTrainer()
    
    try:
        experiment_dir, results = trainer.train_production_model(
            config_name=args.config,
            model_choice=args.model,
            max_train_samples=args.max_samples,
            num_epochs=args.epochs
        )
        
        print(f"\n🎉 TRAINING SUCCESSFUL!")
        print(f"📁 Results: {experiment_dir}")
        print(f"📊 Check comprehensive_results.json for detailed metrics")
        print(f"📈 Check evaluation_metrics.png for training curves")
        print(f"🔍 Check generation_samples.json for sample outputs")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        print("🔧 Check logs for detailed error information")

if __name__ == "__main__":
    main()