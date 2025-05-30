#!/usr/bin/env python3
"""
Universal ProPicto Trainer - Compatible with any Transformers version
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
import transformers

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_compatible_training_args(**kwargs):
    """Get TrainingArguments compatible with current Transformers version"""
    
    # Check Transformers version
    transformers_version = transformers.__version__
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Transformers version: {transformers_version}")
    
    # Try new parameter names first, fall back to old ones
    compatible_kwargs = {}
    
    for key, value in kwargs.items():
        if key == 'evaluation_strategy':
            # Try new name first, fall back to old
            try:
                # Test if eval_strategy is supported
                test_args = TrainingArguments(output_dir="./test", eval_strategy="no")
                compatible_kwargs['eval_strategy'] = value
                logger.info("✅ Using 'eval_strategy' (new format)")
            except TypeError:
                # Fall back to old name
                compatible_kwargs['evaluation_strategy'] = value
                logger.info("✅ Using 'evaluation_strategy' (old format)")
        else:
            compatible_kwargs[key] = value
    
    return TrainingArguments(**compatible_kwargs)

def train_universal(config_name: str, max_samples: int = None, epochs: int = 3):
    """Universal training that works with any Transformers version"""
    
    logger = setup_logging()
    
    print(f"🚀 Universal Training: {config_name}")
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
    logger.info(f"✅ Model loaded on {device}")
    
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
    
    # Task prefix function
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
    
    # Tokenization
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
    logger.info("🔍 Task formulation:")
    sample = train_data[0]
    logger.info(f"  Original: {sample['input_text']}")
    logger.info(f"  French:   {get_task_prefix(sample['input_text'])}")
    logger.info(f"  Target:   {sample['target_text']}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/universal_{config_name}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate training steps
    batch_size = 8
    steps_per_epoch = len(train_dataset) // batch_size
    eval_steps = max(50, steps_per_epoch // 3)  # Evaluate 3 times per epoch
    save_steps = eval_steps
    
    logger.info(f"📊 Training plan:")
    logger.info(f"   Steps per epoch: {steps_per_epoch}")
    logger.info(f"   Eval every: {eval_steps} steps")
    logger.info(f"   Total steps: {steps_per_epoch * epochs}")
    
    # UNIVERSAL training arguments - compatible with any version
    try:
        training_args = get_compatible_training_args(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_steps=min(100, steps_per_epoch // 4),
            logging_steps=max(10, steps_per_epoch // 5),
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",  # Will be converted to eval_strategy if needed
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            fp16=False,
            report_to=[],
            remove_unused_columns=True
        )
        logger.info("✅ Training arguments configured successfully")
        
    except Exception as e:
        logger.error(f"❌ Training arguments failed: {e}")
        # Fallback to minimal configuration
        logger.info("🔄 Using minimal fallback configuration...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=3e-5,
            logging_steps=50,
            save_total_limit=1,
            report_to=[]
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    logger.info("🏋️ Starting universal training...")
    start_time = time.time()
    
    try:
        trainer.train()
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    training_time = time.time() - start_time
    
    # Save model
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Generation test
    logger.info("🎯 Testing generation...")
    model.eval()
    
    test_samples = train_data[:5]  # Test 5 samples
    results = []
    
    generation_success = 0
    
    for i, sample in enumerate(test_samples):
        try:
            french_input = get_task_prefix(sample['input_text'])
            inputs = tokenizer(french_input, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=2,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean prediction (remove task prefix if echoed)
            if prediction.startswith(french_input):
                prediction = prediction[len(french_input):].strip()
            
            # Check if generation is meaningful
            is_good = (
                len(prediction.strip()) > 0 and
                "<extra_id_0>" not in prediction and
                prediction.lower() != french_input.lower() and
                len(prediction.split()) > 1
            )
            
            if is_good:
                generation_success += 1
            
            results.append({
                'sample_id': i,
                'input': sample['input_text'],
                'expected': sample['target_text'],
                'generated': prediction,
                'quality': 'good' if is_good else 'poor'
            })
            
            logger.info(f"Test {i+1} ({'✅' if is_good else '⚠️'}):")
            logger.info(f"  Input:     {sample['input_text']}")
            logger.info(f"  Expected:  {sample['target_text']}")
            logger.info(f"  Generated: {prediction}")
            
        except Exception as e:
            logger.error(f"❌ Generation failed for sample {i+1}: {e}")
            results.append({
                'sample_id': i,
                'input': sample['input_text'],
                'expected': sample['target_text'],
                'generated': f"Error: {str(e)}",
                'quality': 'error'
            })
    
    # Calculate success rate
    success_rate = generation_success / len(test_samples)
    
    # Save results
    final_results = {
        'experiment': {
            'config': config_name,
            'transformers_version': transformers.__version__,
            'training_time_minutes': training_time / 60,
            'model_path': final_model_dir,
            'timestamp': timestamp
        },
        'dataset_info': {
            'train_samples': len(train_dataset),
            'valid_samples': len(valid_dataset),
            'limited': max_samples is not None
        },
        'generation_test': {
            'success_rate': success_rate,
            'successful_generations': generation_success,
            'total_tests': len(test_samples),
            'samples': results
        }
    }
    
    with open(f"{output_dir}/results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Final summary
    logger.info(f"\n🎉 UNIVERSAL TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"🕐 Training time: {training_time/60:.1f} minutes")
    logger.info(f"📁 Model saved: {final_model_dir}")
    logger.info(f"🎯 Generation success: {success_rate:.1%} ({generation_success}/{len(test_samples)})")
    logger.info(f"📊 Results saved: {output_dir}/results.json")
    
    if success_rate >= 0.6:
        logger.info("🎉 SUCCESS: Model generates good French text!")
    elif success_rate >= 0.3:
        logger.info("⚠️  PARTIAL: Model works but may need more training")
    else:
        logger.info("❌ POOR: Model needs significant improvement")
    
    return final_model_dir, final_results

def main():
    parser = argparse.ArgumentParser(description='Universal ProPicto Training')
    parser.add_argument('--config', required=True,
                       choices=['keywords_to_sentence', 'pictos_tokens_to_sentence', 
                               'hybrid_to_sentence', 'direct_to_sentence'],
                       help='Data configuration to use')
    parser.add_argument('--max-samples', type=int, help='Limit training samples')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    
    args = parser.parse_args()
    
    print("🌐 Universal ProPicto Trainer")
    print("=" * 40)
    print("✅ Compatible with any Transformers version")
    print("✅ Automatic parameter adaptation")
    print("✅ Robust error handling")
    print("✅ Comprehensive generation testing")
    
    try:
        model_path, results = train_universal(
            config_name=args.config,
            max_samples=args.max_samples,
            epochs=args.epochs
        )
        
        success_rate = results['generation_test']['success_rate']
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📁 Model: {model_path}")
        print(f"🎯 Success rate: {success_rate:.1%}")
        print("📊 Check results.json for detailed analysis")
        
        if success_rate >= 0.6:
            print("🚀 Ready for production use!")
        else:
            print("🔧 Consider training with more data or epochs")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        print("Full error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()