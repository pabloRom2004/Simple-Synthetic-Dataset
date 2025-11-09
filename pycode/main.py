# pycode/main.py

from preprocessing import preprocess_dataset
from training import train_model
import torch
from huggingface_hub import login 

def main():
    login(token="hugging-face-token-here", add_to_git_credential=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    preprocess_dataset('dataset/OneStopEnglish')
    train_model()

if __name__ == '__main__':
    main()