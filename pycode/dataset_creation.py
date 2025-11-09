from datasets import Dataset, DatasetDict
from config import DATASET_DIR, max_input_length, max_target_length, PHASE

def create_dataset(task):
    complex_list = []
    simple_list = []
    if task in PHASE:
        with open(DATASET_DIR + task + '_complex_processed.txt', encoding='utf-8') as complex_f:
            for i in complex_f.readlines():
                complex_list.append(i.strip())
        with open(DATASET_DIR + task + '_simple_processed.txt', encoding='utf-8') as simple_f:
            for i in simple_f.readlines():
                simple_list.append(i.strip())
        if len(complex_list) == len(simple_list):
            return Dataset.from_dict({'inputs': complex_list, 'targets': simple_list})
        else:
            raise ValueError("Complex and simple lists have different lengths!")
    elif task == 'sample':
        with open(DATASET_DIR + 'test_sample', encoding='utf-8') as sample_f:
            for i in sample_f:
                complex_list.append(i.strip())
                simple_list.append('N/A')
        return Dataset.from_dict({'inputs': complex_list, 'targets': simple_list})
    else:
        raise TypeError('Invalid input!')

def create_dataset_dict():
    return DatasetDict({
        'train': create_dataset('train'),
        'valid': create_dataset('valid'),
        'test': create_dataset('test')
    })

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples['inputs'], max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['targets'], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
