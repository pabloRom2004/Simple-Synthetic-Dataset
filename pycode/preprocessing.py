# pycode/preprocessing.py

from preprocessors import ReadingLevelPreprocessor
import os
from config import PHASE

def preprocess_dataset(dataset_dir: str):
    """Preprocess dataset with reading level tokens"""
    preprocessor = ReadingLevelPreprocessor()
    
    for phase in PHASE:
        complex_path = os.path.join(dataset_dir, f'{phase}_complex.txt')
        simple_path = os.path.join(dataset_dir, f'{phase}_simple.txt')
        
        if not os.path.exists(complex_path) or not os.path.exists(simple_path):
            print(f"Missing files for {phase} phase")
            continue
            
        output_complex = os.path.join(dataset_dir, f'{phase}_complex_processed.txt')
        output_simple = os.path.join(dataset_dir, f'{phase}_simple_processed.txt')
        
        print(f"Processing {phase} files...")
        
        # Read files and process line pairs
        with open(complex_path, 'r', encoding='utf-8') as cf, \
             open(simple_path, 'r', encoding='utf-8') as sf, \
             open(output_complex, 'w', encoding='utf-8') as cof, \
             open(output_simple, 'w', encoding='utf-8') as sof:
            
            for complex_line, simple_line in zip(cf, sf):
                if complex_line.strip() and simple_line.strip():
                    # Calculate grade level of simple text
                    score = preprocessor.calculate_reading_metrics(simple_line.strip())
                    target_level = preprocessor.get_grade_level(score)
                    
                    # Add target level to complex text
                    processed_complex = f"{target_level} {complex_line.strip()}"
                    
                    # Write to output files
                    cof.write(processed_complex + '\n')
                    sof.write(simple_line.strip() + '\n')