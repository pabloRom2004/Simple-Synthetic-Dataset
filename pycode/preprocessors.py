# pycode/preprocessors.py

from abc import ABC
import hashlib
from utils import store_args, get_temp_filepath, yield_lines_in_parallel
from typing import Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

class ReadingLevelPreprocessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        
    def calculate_reading_metrics(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            words = [word for word in words if any(c.isalnum() for c in word)]
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            total_syllables = sum(self.sylco(word) for word in words)
            
            score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
            return max(0, min(100, score))
        
        except Exception as e:
            print(f"Error processing text: {e}")
            return 0

    def get_grade_level(self, score: float) -> str:
        """Map Flesch score to grade level token"""
        if score >= 70:
            return '<LEVEL_ELEMENTARY>'    # 5th-7th grade
        elif score >= 50:
            return '<LEVEL_SECONDARY>'     # 8th-12th grade
        else:
            return '<LEVEL_ADVANCED>'      # College and above

    def encode_sentence(self, sentence: str) -> str:
        """Add reading level token to sentence"""
        score = self.calculate_reading_metrics(sentence)
        level_token = self.get_grade_level(score)
        return f"{level_token} {sentence}"

    def encode_file_pair(self, complex_filepath: str, simple_filepath: str, 
                        output_complex_filepath: str, output_simple_filepath: str):
        """Process file pairs and add reading level tokens"""
        with open(complex_filepath, 'r', encoding='utf-8') as cf, \
             open(simple_filepath, 'r', encoding='utf-8') as sf, \
             open(output_complex_filepath, 'w', encoding='utf-8') as cof, \
             open(output_simple_filepath, 'w', encoding='utf-8') as sof:
            
            for complex_line, simple_line in zip(cf, sf):
                if complex_line.strip() and simple_line.strip():
                    encoded_complex = self.encode_sentence(complex_line.strip())
                    encoded_simple = self.encode_sentence(simple_line.strip())
                    cof.write(encoded_complex + '\n')
                    sof.write(encoded_simple + '\n')

    def sylco(self, word: str) -> int:
        """Calculate number of syllables in a word"""
        word = word.lower()
        
        # exception_add are words that need extra syllables
        # exception_del are words that need less syllables
        exception_add = ['serious','crucial']
        exception_del = ['fortunately','unfortunately']
        
        co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
        co_two = ['coapt','coed','coinci']
        pre_one = ['preach']
        
        syls = 0  # added syllable number
        disc = 0  # discarded syllable number
        
        # 1) if letters < 3 : return 1
        if len(word) <= 3:
            return 1
            
        # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end
        if word[-2:] == "es" or word[-2:] == "ed":
            doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
            if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1:
                if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies":
                    pass
                else:
                    disc += 1
                    
        # 3) discard trailing "e", except where ending is "le"
        le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']
        if word[-1:] == "e":
            if word[-2:] == "le" and word not in le_except:
                pass
            else:
                disc += 1
                
        # 4) check if consecutive vowels exists, triplets or pairs, count them as one
        doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
        tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
        disc += doubleAndtripple + tripple
        
        # 5) count remaining vowels in word
        numVowels = len(re.findall(r'[eaoui]',word))
        
        # 6) add one if starts with "mc"
        if word[:2] == "mc":
            syls += 1
            
        # 7) add one if ends with "y" but is not surrounded by vowel
        if word[-1:] == "y" and word[-2] not in "aeoui":
            syls += 1
            
        # 8) add one if "y" is surrounded by non-vowels and is not in the last word
        for i,j in enumerate(word):
            if j == "y":
                if (i != 0) and (i != len(word)-1):
                    if word[i-1] not in "aeoui" and word[i+1] not in "aeoui":
                        syls += 1
                        
        # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one
        if word[:3] == "tri" and word[3] in "aeoui":
            syls += 1
        if word[:2] == "bi" and word[2] in "aeoui":
            syls += 1
            
        # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
        if word[-3:] == "ian":
            if word[-4:] == "cian" or word[-4:] == "tian":
                pass
            else:
                syls += 1
                
        # 11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary
        if word[:2] == "co" and word[2] in 'eaoui':
            if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
                syls += 1
            elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one:
                pass
            else:
                syls += 1
                
        # 12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary
        if word[:3] == "pre" and word[3] in 'eaoui':
            if word[:6] in pre_one:
                pass
            else:
                syls += 1
                
        # 13) check for "-n't" and cross match with dictionary to add syllable
        negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]
        if word[-3:] == "n't":
            if word in negative:
                syls += 1
                
        # 14) Handling the exceptional words
        if word in exception_del:
            disc += 1
        if word in exception_add:
            syls += 1
            
        return numVowels - disc + syls