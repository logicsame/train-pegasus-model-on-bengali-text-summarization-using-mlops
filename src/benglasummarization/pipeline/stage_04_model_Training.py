from benglasummarization.components.model_training import ModelTraining
from benglasummarization.config.configuration import ConfigurationManager

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTraining(config=model_training_config)
        model_trainer.train()