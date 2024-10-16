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
    