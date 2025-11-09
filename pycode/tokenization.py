from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import TOKENIZER_TO_LOAD, MODEL_TO_LOAD, OUTPUT_PATH, CONTROL_TOKENS

def load_tokenizer_and_model():
    """
    Loads the tokenizer and model from pretrained sources with clean generation config
    """
    # First load with no generation config
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TO_LOAD)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TO_LOAD)
    
    return tokenizer, model

def update_tokenizer(tokenizer, model):
    """
    Adds additional control tokens (e.g. for reading levels) to the tokenizer,
    resizes the model embeddings accordingly, and saves the updated tokenizer.
    """
    tokenizer.add_tokens(CONTROL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(OUTPUT_PATH)
    print('Tokenizer updated and the vocabulary size is:', len(tokenizer))