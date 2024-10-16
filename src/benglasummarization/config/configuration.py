from src.benglasummarization.constants import *
from src.benglasummarization.utils.common import read_yaml, create_directories
from benglasummarization.entity.config_entity import DataIngestionConfig
from src.benglasummarization.entity.config_entity import BanTokenizationConfig
from src.benglasummarization.entity.config_entity import BanTokenTrainConfig, ModelTrainingConfig
class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_dir=config.source_dir,
            local_data_file=config.local_data_file,
            unzip_dir= config.unzip_dir
        )
        
        return data_ingestion_config
    
    
    def get_ben_tokenization_config(self) -> BanTokenizationConfig:
        config = self.config.ban_tokenization
        params = self.params.pre_tokenize
        create_directories([config.root_dir])

        ben_tokenization_config = BanTokenizationConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            save_dir= config.save_dir,
            output_file= params.output_file
        )
 
        return ben_tokenization_config
    
    
    def get_train_token_config(self) -> BanTokenTrainConfig:
        config = self.config.train_tokenize
        params = self.params.train_tokenize
        create_directories([config.root_dir])
        
        train_token_config = BanTokenTrainConfig(
            root_dir= config.root_dir,
            input_file_dir= config.input_file_dir,
            save_file= config.save_file,
            model_prefix= params.model_prefix,
            model_type= params.model_type,
            vocab_size= params.vocab_size
        )
        return train_token_config
    
    def get_model_trainer_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        param = self.params.training_model
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainingConfig(
            root_dir= config.root_dir,
            data_dir= config.data_dir,
            ben_tokenizer_dir= config.ben_tokenizer_dir,
            save_trained_model_dir= config.save_trained_model_dir,
            max_input_length = param.max_input_length,
            max_output_length = param.max_output_length,
            batch_size = param.batch_size,
            num_epochs = param.num_epochs,
            accumulator_steps = param.accumulator_steps,
            max_grad_norm = param.max_grad_norm,
            early_stopping_patience = param.early_stopping_patience,
            patience_counter = param.patience_counter,
            model_name = param.model_name,
            learning_rate = param.learning_rate
            
        )
        return model_trainer_config