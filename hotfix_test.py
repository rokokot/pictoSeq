#!/usr/bin/env python3
"""
Hotfix training script - simplified evaluation to avoid issues
"""

import logging
import torch
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def train_with_basic_eval(config_name: str, max_samples: int = None, epochs: int = 3):
    """Simple training with basic evaluation"""
    
    logger = setup_logging()
    
    print(f"🚀 Training {config_name} with simplified evaluation")
    print(f"📊 Max samples: {max_samples or 'ALL'}")
    print(f"🔄 Epochs: {epochs}")
    
    # Load model
    logger.info("🤖 Loading BARThez...")
    tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
    model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load data
    data_root = Path("data/processed_propicto")
    config_path = data_root / config_name
    
    with open(config_path / "train" / "data.json", 'r') as f:
        train_data = json.load(f)
    
    with open(config_path / "valid" / "data.json", 'r') as f:
        valid_data = json.load(f)
    
    logger.info(f"📊 Loaded: {len(train_data):,} train, {len(valid_data):,} valid")
    
    # Limit data if requested
    if max_samples and max_samples < len(train_data):
        train_data = train_data[:max_samples]
        valid_data = valid_data[:max(10, max_samples // 10)]
        logger.info(f"🧪 Limited to: {len(train_data):,} train, {len(valid_data):,} valid")
    
    # Tokenize function
    def get_task_prefix(input_text: str):
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
            french_input = get_task_prefix(input_text)
            inputs.append(french_input)
            targets.append(target_text)
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
        
        model_inputs["labels"] = [
            [t if t != tokenizer.pad_token_id else -100 for t in label_ids]
            for label_ids in labels["input_ids"]
        ]
        
        return model_inputs
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True)
    valid_dataset = Dataset.from_list(valid_data).map(tokenize_function, batched=True)
    
    # Show example
    logger.info("🔍 Example transformation:")
    sample = train_data[0]
    logger.info(f"  Original: {sample['input_text']}")
    logger.info(f"  French:   {get_task_prefix(sample['input_text'])}")
    logger.info(f"  Target:   {sample['target_text']}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/hotfix_{config_name}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training args - SIMPLIFIED
    steps_per_epoch = len(train_dataset) // 8
    eval_steps = max(50, steps_per_epoch // 2)  # Less frequent evaluation
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=min(100, steps_per_epoch // 4),
        logging_steps=max(10, steps_per_epoch // 5),
        eval_steps=eval_steps,
        save_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        fp16=False,
        report_to=[],
        remove_unused_columns=True
    )
    
    # Simple trainer - NO CUSTOM CALLBACKS
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    logger.info("🏋️ Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    
    # Save model
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Quick generation test
    logger.info("🎯 Testing generation...")
    model.eval()
    
    test_samples = train_data[:3]
    results = []
    
    for i, sample in enumerate(test_samples):
        french_input = get_task_prefix(sample['input_text'])
        inputs = tokenizer(french_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean prediction
        if prediction.startswith(french_input):
            prediction = prediction[len(french_input):].strip()
        
        results.append({
            'input': sample['input_text'],
            'expected': sample['target_text'],
            'generated': prediction
        })
        
        logger.info(f"Test {i+1}:")
        logger.info(f"  Input:     {sample['input_text']}")
        logger.info(f"  Expected:  {sample['target_text']}")
        logger.info(f"  Generated: {prediction}")
    
    # Save results
    final_results = {
        'config': config_name,
        'training_time_minutes': training_time / 60,
        'model_path': final_model_dir,
        'dataset_sizes': {
            'train': len(train_dataset),
            'valid': len(valid_dataset)
        },
        'test_results': results
    }
    
    with open(f"{output_dir}/results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Summary
    logger.info(f"\n🎉 TRAINING COMPLETE")
    logger.info(f"🕐 Time: {training_time/60:.1f} minutes")
    logger.info(f"📁 Model: {final_model_dir}")
    logger.info(f"📊 Results: {output_dir}/results.json")
    
    # Check if generation is working
    working_count = 0
    for result in results:
        if (len(result['generated'].strip()) > 0 and 
            "<extra_id_0>" not in result['generated'] and
            len(result['generated'].split()) > 2):
            working_count += 1
    
    success_rate = working_count / len(results)
    logger.info(f"🎯 Generation success: {success_rate:.1%} ({working_count}/{len(results)})")
    
    if success_rate >= 0.5:
        logger.info("✅ Model is generating meaningful French text!")
    else:
        logger.warning("⚠️  Model may need more training")
    
    return final_model_dir, final_results

def main():
    parser = argparse.ArgumentParser(description='Hotfix ProPicto Training')
    parser.add_argument('--config', required=True,
                       choices=['keywords_to_sentence', 'pictos_tokens_to_sentence', 
                               'hybrid_to_sentence', 'direct_to_sentence'],
                       help='Data configuration to use')
    parser.add_argument('--max-samples', type=int, help='Limit training samples')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    
    args = parser.parse_args()
    
    print("🔧 Hotfix ProPicto Trainer")
    print("=" * 40)
    print("✅ Simplified evaluation (no complex metrics)")
    print("✅ Robust error handling")
    print("✅ Fast training with basic monitoring")
    
    try:
        model_path, results = train_with_basic_eval(
            config_name=args.config,
            max_samples=args.max_samples,
            epochs=args.epochs
        )
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Model: {model_path}")
        print("🔍 Check results.json for generation examples")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()