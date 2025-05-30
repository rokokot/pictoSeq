#!/usr/bin/env python3
"""
French-Specific ProPicto Training
Using French-native models for French-to-French text normalization
Based on research recommendations for monolingual seq2seq tasks
"""

import logging
import torch
import json
import time
from pathlib import Path
from transformers import (
    # French-specific models
    AutoTokenizer, AutoModelForSeq2SeqLM,
    # Training infrastructure  
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class FrenchProPictoTrainer:
    """
    Trainer using French-native models for French text normalization
    """
    def __init__(self):
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # French-specific model options (in order of preference)
        self.french_models = {
            'barthez': 'moussaKam/barthez',  # Best for French generation
            'french_t5': 'plguillou/t5-base-fr-sum-cnndm',  # French T5
            'mbarthez': 'moussaKam/mbarthez',  # Enhanced BARThez
        }
        
    def load_french_model(self, model_choice='barthez'):
        """Load French-specific model"""
        model_name = self.french_models[model_choice]
        self.logger.info(f"🇫🇷 Loading French model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Ensure proper padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model.to(self.device)
            self.logger.info(f"✅ French model loaded successfully")
            
            return model, tokenizer, model_choice
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to load {model_choice}: {e}")
            if model_choice != 'french_t5':
                self.logger.info("🔄 Falling back to French T5...")
                return self.load_french_model('french_t5')
            else:
                raise e
    
    def prepare_french_data(self, data_path: str, num_samples: int = 100):
        """Prepare data with French-specific task formulation"""
        self.logger.info(f"📊 Loading French normalization data...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data[:num_samples]
        
        # Analyze the actual task
        self.logger.info("🔍 French Task Analysis:")
        for i, sample in enumerate(samples[:3]):
            input_text = sample['input_text'] 
            target_text = sample['target_text']
            
            # Remove "mots:" prefix to see the actual keywords
            keywords = input_text.replace("mots:", "").strip()
            
            self.logger.info(f"Example {i+1}:")
            self.logger.info(f"  Keywords:   '{keywords}'")
            self.logger.info(f"  Normalized: '{target_text}'")
            
            # Show the transformation type
            if len(target_text.split()) > len(keywords.split()):
                task_type = "expansion + correction"
            else:
                task_type = "correction + reordering"
            self.logger.info(f"  Task type:  {task_type}")
        
        return samples
    
    def tokenize_french_data(self, samples, tokenizer, model_choice):
        """Tokenize with French-appropriate task prefixes"""
        
        def get_french_task_prefix(model_choice, keywords):
            """Get appropriate French task prefix"""
            if model_choice == 'barthez':
                # BARThez works well with direct French instructions
                return f"Corriger et compléter: {keywords}"
            elif model_choice == 'french_t5':
                # French T5 trained on summarization, adapt the pattern
                return f"normaliser: {keywords}"
            else:
                # Generic French instruction
                return f"transformer en phrase: {keywords}"
        
        def tokenize_function(examples):
            inputs = []
            targets = []
            
            for input_text, target_text in zip(examples['input_text'], examples['target_text']):
                # Clean input and create French task instruction
                keywords = input_text.replace("mots:", "").strip()
                french_input = get_french_task_prefix(model_choice, keywords)
                
                inputs.append(french_input)
                targets.append(target_text)
            
            # Tokenize
            model_inputs = tokenizer(
                inputs,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            # Tokenize targets
            labels = tokenizer(
                targets,
                max_length=128,
                truncation=True,
                padding='max_length', 
                return_tensors=None
            )
            
            # Process labels
            model_inputs["labels"] = []
            for label_ids in labels["input_ids"]:
                if hasattr(label_ids, 'tolist'):
                    label_ids = label_ids.tolist()
                
                processed_labels = [
                    token_id if token_id != tokenizer.pad_token_id else -100 
                    for token_id in label_ids
                ]
                model_inputs["labels"].append(processed_labels)
            
            return model_inputs
        
        # Show the new French approach
        sample = samples[0]
        keywords = sample['input_text'].replace("mots:", "").strip()
        french_input = get_french_task_prefix(model_choice, keywords)
        
        self.logger.info("🇫🇷 FRENCH Task Formulation:")
        self.logger.info(f"  Original:     {sample['input_text']}")
        self.logger.info(f"  French input: {french_input}")
        self.logger.info(f"  Target:       {sample['target_text']}")
        
        # Create and tokenize dataset
        dataset = Dataset.from_list(samples)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_french_model(self, num_samples=200, num_epochs=4):
        """Train with French-optimized configuration"""
        
        # Try French models in order of preference
        for model_choice in ['barthez', 'french_t5']:
            try:
                model, tokenizer, used_model = self.load_french_model(model_choice)
                break
            except Exception as e:
                self.logger.warning(f"Failed to load {model_choice}: {e}")
                if model_choice == 'french_t5':  # Last option
                    raise e
        
        # Load and prepare data
        train_path = "data/processed_propicto/keywords_to_sentence/train/data.json"
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Data not found: {train_path}")
        
        train_data = self.prepare_french_data(train_path, num_samples)
        eval_data = train_data[-20:]  # Use last 20 for eval
        train_data = train_data[:-20]  # Remove eval from train
        
        # Tokenize with French-specific approach
        train_dataset = self.tokenize_french_data(train_data, tokenizer, used_model)
        eval_dataset = self.tokenize_french_data(eval_data, tokenizer, used_model)
        
        self.logger.info(f"📊 French Training: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Output directory
        output_dir = f"models/french_propicto_{used_model}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # FIXED: French-optimized training arguments with aligned steps
        eval_steps = max(5, len(train_dataset) // 4)
        save_steps = eval_steps * 2  # FIXED: Make save_steps a multiple of eval_steps
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,  # Conservative for pretrained French model
            weight_decay=0.01,
            warmup_steps=len(train_dataset) // 8,
            logging_steps=max(1, len(train_dataset) // 10),
            eval_steps=eval_steps,
            save_steps=save_steps,  # FIXED: Now a multiple of eval_steps
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            fp16=False,
            dataloader_pin_memory=False,
            report_to=[],
            remove_unused_columns=True
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train
        self.logger.info(f"🚀 Starting French-specific training with {used_model}...")
        start_time = time.time()
        
        trainer.train()
        training_time = time.time() - start_time
        
        # Test French generation
        self.test_french_generation(model, tokenizer, used_model, train_data[:3])
        
        # Save
        final_dir = f"models/french_propicto_final_{used_model}"
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        self.logger.info(f"✅ French training completed in {training_time/60:.1f} minutes")
        self.logger.info(f"🇫🇷 French model saved to: {final_dir}")
        
        return final_dir, used_model
    
    def test_french_generation(self, model, tokenizer, model_choice, test_samples):
        """Test French generation with appropriate prompts"""
        self.logger.info("🎯 Testing French generation...")
        
        model.eval()
        
        def get_french_task_prefix(model_choice, keywords):
            if model_choice == 'barthez':
                return f"Corriger et compléter: {keywords}"
            elif model_choice == 'french_t5':
                return f"normaliser: {keywords}"
            else:
                return f"transformer en phrase: {keywords}"
        
        for i, sample in enumerate(test_samples):
            keywords = sample['input_text'].replace("mots:", "").strip()
            french_input = get_french_task_prefix(model_choice, keywords)
            
            self.logger.info(f"\n🇫🇷 French Test {i+1}:")
            self.logger.info(f"  Keywords: {keywords}")
            self.logger.info(f"  Expected: {sample['target_text']}")
            
            # Test different generation strategies
            strategies = {
                'greedy': {'do_sample': False, 'num_beams': 1},
                'beam': {'do_sample': False, 'num_beams': 3, 'length_penalty': 1.2},
                'nucleus': {'do_sample': True, 'top_p': 0.9, 'temperature': 0.8}
            }
            
            for strategy_name, params in strategies.items():
                try:
                    inputs = tokenizer(french_input, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=100,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            **params
                        )
                    
                    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean output (remove task prefix if echoed)
                    if prediction.startswith(french_input):
                        prediction = prediction[len(french_input):].strip()
                    
                    self.logger.info(f"  {strategy_name:8}: {prediction}")
                    
                except Exception as e:
                    self.logger.warning(f"  {strategy_name:8}: Error - {e}")

def main():
    print("🇫🇷 French-Specific ProPicto Trainer")
    print("=" * 50)
    print("✅ Using French-native models (BARThez/French-T5)")
    print("✅ Proper French task formulation") 
    print("✅ Monolingual French-to-French normalization")
    
    trainer = FrenchProPictoTrainer()
    
    try:
        model_path, model_used = trainer.train_french_model(
            num_samples=200,  # Reasonable amount
            num_epochs=4      # Enough for fine-tuning
        )
        
        print(f"\n🎉 FRENCH TRAINING SUCCESS!")
        print(f"📁 Model: {model_path}")
        print(f"🤖 Used: {model_used}")
        print("\n🔍 Expected improvements:")
        print("   1. No more <extra_id_0> outputs")
        print("   2. Proper French text generation")
        print("   3. Better keyword expansion")
        print("   4. Native French understanding")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 Suggestions:")
        print("   1. Check if BARThez/French-T5 models are accessible")
        print("   2. Verify data paths")
        print("   3. Ensure sufficient GPU memory")

if __name__ == "__main__":
    main()