#!/usr/bin/env python3
"""
Quick fix for the French training - just the strategy mismatch
"""

import logging
import torch
import json
import time
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset

def quick_french_test():
    """Quick test with fixed training arguments"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("🇫🇷 Quick French Fix Test")
    print("=" * 30)
    
    # Load BARThez (already downloaded)
    logger.info("🤖 Loading BARThez...")
    tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
    model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Quick data prep
    logger.info("📊 Loading data...")
    with open("data/processed_propicto/keywords_to_sentence/train/data.json", 'r') as f:
        data = json.load(f)
    
    # Use small sample
    train_samples = data[:50]  # Small for quick test
    eval_samples = data[50:60]
    
    def tokenize_function(examples):
        inputs = []
        targets = []
        
        for input_text, target_text in zip(examples['input_text'], examples['target_text']):
            keywords = input_text.replace("mots:", "").strip()
            french_input = f"Corriger et compléter: {keywords}"
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
    train_dataset = Dataset.from_list(train_samples).map(tokenize_function, batched=True)
    eval_dataset = Dataset.from_list(eval_samples).map(tokenize_function, batched=True)
    
    logger.info(f"📊 Quick test: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # FIXED training arguments
    training_args = TrainingArguments(
        output_dir="models/quick_french_test",
        num_train_epochs=2,  # Quick test
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=3e-5,
        logging_steps=5,
        eval_steps=10,
        evaluation_strategy="steps",  # Match save strategy
        save_strategy="steps",        # FIXED: Both use "steps"
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        fp16=False,
        report_to=[]
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
    logger.info("🚀 Starting quick French training...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    
    # Quick generation test
    logger.info("🎯 Testing French generation...")
    model.eval()
    
    test_sample = train_samples[0]
    keywords = test_sample['input_text'].replace("mots:", "").strip()
    french_input = f"Corriger et compléter: {keywords}"
    
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
    
    print(f"\n🎯 FRENCH GENERATION TEST:")
    print(f"Keywords:   {keywords}")
    print(f"Expected:   {test_sample['target_text']}")
    print(f"Generated:  {prediction}")
    
    # Check if it's working
    is_working = (
        len(prediction.strip()) > 0 and
        prediction != french_input and
        "<extra_id_0>" not in prediction and
        len(prediction.split()) > 2
    )
    
    print(f"\n🎉 RESULT: {'SUCCESS' if is_working else 'NEEDS_WORK'}")
    print(f"⏱️  Training time: {training_time:.1f} seconds")
    
    if is_working:
        print("✅ French model is generating sensible text!")
        print("✅ Ready for full training with more data")
    else:
        print("⚠️  May need more training data or epochs")
    
    return is_working

if __name__ == "__main__":
    success = quick_french_test()
    if success:
        print("\n🚀 Run full training: python french_specific_trainer.py")
    else:
        print("\n🔧 Investigate generation quality")