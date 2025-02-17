import json
import os
import glob
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

def read_json_files(directory: str) -> List[Dict]:
    """Read and combine all JSON files in the directory"""
    all_data = []
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for json_file in tqdm(json_files, desc="Reading JSON files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                all_data.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                continue
    
    return all_data

def create_pairs(data: List[Dict]) -> List[Tuple[str, str]]:
    """Create pairs of complex-simple sentences with level tokens"""
    pairs = []
    for entry in tqdm(data, desc="Creating pairs"):
        # Create three pairs for each entry
        pairs.extend([
            (f"<LEVEL_ELEMENTARY> {entry['COMPLEX']}", entry['ELEMENTARY']),
            (f"<LEVEL_SECONDARY> {entry['COMPLEX']}", entry['SECONDARY']),
            (f"<LEVEL_ADVANCED> {entry['COMPLEX']}", entry['ADVANCED'])
        ])
    return pairs

def split_data(pairs: List[Tuple[str, str]], train_ratio=0.99, test_ratio=0.008):
    """Split data into train/test/valid sets"""
    random.shuffle(pairs)  # Shuffle the pairs
    
    n = len(pairs)
    train_size = int(n * train_ratio)
    test_size = int(n * test_ratio)
    
    train = pairs[:train_size]
    test = pairs[train_size:train_size + test_size]
    valid = pairs[train_size + test_size:]
    
    return train, test, valid

def write_to_files(output_dir: str, train: List[Tuple], test: List[Tuple], valid: List[Tuple]):
    """Write the splits to appropriate files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to write pairs to files
    def write_split(pairs: List[Tuple], complex_file: str, simple_file: str):
        with open(complex_file, 'w', encoding='utf-8') as fc, \
             open(simple_file, 'w', encoding='utf-8') as fs:
            for complex_sent, simple_sent in pairs:
                fc.write(complex_sent + '\n')
                fs.write(simple_sent + '\n')
    
    # Write train/test/valid files
    write_split(train, 
               os.path.join(output_dir, 'train_complex_processed.txt'),
               os.path.join(output_dir, 'train_simple_processed.txt'))
    
    write_split(test,
               os.path.join(output_dir, 'test_complex_processed.txt'),
               os.path.join(output_dir, 'test_simple_processed.txt'))
    
    write_split(valid,
               os.path.join(output_dir, 'valid_complex_processed.txt'),
               os.path.join(output_dir, 'valid_simple_processed.txt'))

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Input and output directories
    input_dir = "dataset/Synthetic_Data"
    output_dir = "dataset/Synthetic_Data_TXT"
    
    # Process the data
    print("Starting data processing...")
    
    # Read all JSON files
    data = read_json_files(input_dir)
    print(f"Read {len(data)} examples from JSON files")
    
    # Create pairs with level tokens
    pairs = create_pairs(data)
    print(f"Created {len(pairs)} pairs")
    
    # Split the data
    train, test, valid = split_data(pairs)
    print(f"Split sizes: Train={len(train)}, Test={len(test)}, Valid={len(valid)}")
    
    # Write to files
    write_to_files(output_dir, train, test, valid)
    print("Finished writing files to", output_dir)

if __name__ == "__main__":
    main()