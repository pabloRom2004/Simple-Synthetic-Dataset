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

def create_control_token_hook(control_ids, factor=100):
    """
    Returns a hook function that scales the gradient for rows corresponding
    to control token ids by the specified factor.
    """
    def hook(grad):
        # Scale the gradient rows for each control token id.
        # Note: modifying in-place is acceptable here.
        grad[control_ids] = grad[control_ids] * factor
        return grad
    return hook

def train_model():
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model()
    tokenizer.clean_up_tokenization_spaces = True  # Explicit setting for cleaner tokenisation
    update_tokenizer(tokenizer, model)
    
    # Register a gradient hook on the embedding weights so that the control token gradients get scaled.
    control_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in CONTROL_TOKENS]
    print("Control token IDs:", control_token_ids)
    embedding_weight = model.get_input_embeddings().weight
    embedding_weight.register_hook(create_control_token_hook(control_token_ids, factor=100))
    
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
    batch_size = 32
    num_epochs = 15
    total_steps = (num_train_examples * num_epochs) // batch_size
    warmup_steps = total_steps // 10

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_PATH,
        eval_strategy="epoch",
        eval_steps=250,
        learning_rate=0.000025, 
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
        hub_model_id="pabRomero/BART-Firefox-Simplification",
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
    tokenizer.push_to_hub("pabRomero/BART-Firefox-Simplification")

if __name__ == '__main__':
    login(token="token-here", add_to_git_credential=True)
    train_model()
