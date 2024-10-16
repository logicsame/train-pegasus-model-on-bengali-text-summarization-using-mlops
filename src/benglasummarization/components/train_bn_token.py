import sentencepiece as spm
from src.benglasummarization.logging import logger
from tqdm.notebook import tqdm
import os
from benglasummarization.entity.config_entity import BanTokenTrainConfig
class TrainTokenize:
    def __init__(self, config: BanTokenTrainConfig):
        self.config = config
        
    def train_tokenizer(self):
        with open(self.config.input_file_dir, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)

        with tqdm(total=total_lines, desc='Preparing Sentence for Training', unit='lines') as pbar:
            with open(self.config.input_file_dir, 'r', encoding='utf-8') as f:
                for _ in f:
                    pbar.update(1)
                    
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(self.config.save_file), exist_ok=True)
        
        # Training Arguments
        train_params = {
            'input': str(self.config.input_file_dir),
            'model_prefix': os.path.join(self.config.save_file, self.config.model_prefix),
            'vocab_size': self.config.vocab_size,
            'model_type': self.config.model_type,
            'character_coverage': 1.0,
            'input_sentence_size': 1000000,
            'shuffle_input_sentence': True
        }
        
        spm.SentencePieceTrainer.train(**train_params)
        logger.info(f'Tokenizer model saved to {train_params["model_prefix"]}.model')
        logger.info(f'Tokenizer vocabulary saved to {train_params["model_prefix"]}.vocab')
    
    