from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_dir : Path
    local_data_file : Path
    unzip_dir : Path
    
    
@dataclass(frozen=True)
class BanTokenizationConfig:
    root_dir : Path
    source_dir : Path
    save_dir : Path
    output_file : str
    
@dataclass(frozen=True)
class BanTokenTrainConfig:
    root_dir : Path
    input_file_dir : Path
    save_file : Path
    model_prefix : str
    model_type : str
    vocab_size : int
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir : Path
    data_dir : Path 
    ben_tokenizer_dir : Path
    save_trained_model_dir : Path
    max_input_length : int
    max_output_length : int
    batch_size : int
    num_epochs : int
    accumulator_steps : int
    max_grad_norm : float
    early_stopping_patience : int
    patience_counter : int
    model_name : str
    learning_rate : float
    