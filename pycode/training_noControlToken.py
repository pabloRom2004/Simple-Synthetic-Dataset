import torch
from transformers import (
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
)
from config import OUTPUT_PATH, MODE, CONTROL_TOKENS
from dataset_creation import create_dataset_dict, preprocess_function
from tokenization import load_tokenizer_and_model, update_tokenizer
from huggingface_hub import login 

def train_model():
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model()
    
    # Create and preprocess the dataset
    dataset = create_dataset_dict()
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), 
        batched=True,
        remove_columns=dataset['train'].column_names  # Clean up extra columns
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Calculate warmup steps based on training examples and epochs
    num_train_examples = len(dataset['train'])
    batch_size = 8
    num_epochs = 8
    total_steps = (num_train_examples * num_epochs) // batch_size
    warmup_steps = total_steps // 10

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_PATH,
        eval_strategy="epoch",
        learning_rate=0.0001,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        fp16=False,
        load_best_model_at_end=True,  # Save the best model based on eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        logging_steps=50,
        logging_first_step=True,
        push_to_hub=True,
        hub_model_id="pabRomero/BART-Firefox-Simplification-Advanced",
        hub_token=None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['valid'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and evaluate the model
    trainer.train()
    trainer.evaluate(tokenized_dataset['valid'])
    
    # Push final model, tokenizer, and generation configuration to the hub
    trainer.push_to_hub()
    tokenizer.push_to_hub("pabRomero/BART-Firefox-Simplification-Advanced")

if __name__ == '__main__':
    login(token="token-here", add_to_git_credential=True)
    train_model()
