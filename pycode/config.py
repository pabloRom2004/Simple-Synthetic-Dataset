# Description: Configuration file for the project
MODE = 'joint'
DATASET_DIR = 'dataset/Synthetic_Data_TXT_Split/advanced/'
OUTPUT_PATH = f'model/bart_base_{MODE}/'
MODEL_TO_LOAD = 'facebook/bart-base'
TOKENIZER_TO_LOAD = 'facebook/bart-base'

CONTROL_TOKENS = [
    '<LEVEL_ADVANCED>',      # College and above
    '<LEVEL_SECONDARY>',     # 8th-12th grade
    '<LEVEL_ELEMENTARY>'     # 5th-7th grade
]

# Constants
PHASE = ['train', 'valid', 'test']
SPECIAL_TOKEN_REGEX = r'<[a-zA-Z\-\d\.]+>'  # kept one version only
max_input_length = 80
max_target_length = 80

#---------------Constants--------------------
SUFFIX_NAMES = ['.txt','_LR.txt', '_RL.txt', '_WR.txt', '_DTD.txt']
FASTTEXT_EMBEDDINGS_DIR = './dataset/fasttext_vectors'
VALUES = [str(round(0.05*x+0.2, 2)) for x in range(0, 27)]
