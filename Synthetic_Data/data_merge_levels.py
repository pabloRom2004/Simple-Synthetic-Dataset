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

def create_level_pairs(data: List[Dict]) -> Dict[str, List[Tuple[str, str]]]:
    """Create separate pairs for each level, following the natural progression"""
    pairs = {
        'elementary': [],
        'secondary': [],
        'advanced': []
    }
    
    for entry in tqdm(data, desc="Creating level-specific pairs"):
        # Advanced level - just COMPLEX->ADVANCED
        pairs['advanced'].append((entry['COMPLEX'], entry['ADVANCED']))
        
        # Secondary level - follows the progression
        pairs['secondary'].extend([
            (entry['COMPLEX'], entry['SECONDARY']),  # Original complex->secondary
            (entry['ADVANCED'], entry['SECONDARY'])  # Advanced->secondary
        ])
        
        # Elementary level - all possible simplifications following the progression
        pairs['elementary'].extend([
            (entry['COMPLEX'], entry['ELEMENTARY']),   # Original complex->elementary
            (entry['ADVANCED'], entry['ELEMENTARY']),  # Advanced->elementary
            (entry['SECONDARY'], entry['ELEMENTARY'])  # Secondary->elementary
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

def write_to_files(base_output_dir: str, level: str, train: List[Tuple], test: List[Tuple], valid: List[Tuple]):
    """Write the splits to appropriate files in level-specific directories"""
    output_dir = os.path.join(base_output_dir, level)
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
    base_output_dir = "dataset/Synthetic_Data_TXT_Split"
    
    # Process the data
    print("Starting data processing...")
    
    # Read all JSON files
    data = read_json_files(input_dir)
    print(f"Read {len(data)} examples from JSON files")
    
    # Create level-specific pairs
    level_pairs = create_level_pairs(data)
    
    # Process each level separately
    for level, pairs in level_pairs.items():
        print(f"\nProcessing {level} level...")
        print(f"Total pairs for {level}: {len(pairs)}")
        
        # Split the data
        train, test, valid = split_data(pairs)
        print(f"Split sizes for {level}:")
        print(f"Train={len(train)}, Test={len(test)}, Valid={len(valid)}")
        
        # Write to files
        write_to_files(base_output_dir, level, train, test, valid)
        print(f"Finished writing {level} files")

if __name__ == "__main__":
    main()