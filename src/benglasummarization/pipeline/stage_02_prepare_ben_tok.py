from src.benglasummarization.components.prepare_ben_token import BanTokenization
from src.benglasummarization.config.configuration import ConfigurationManager


class BenTokenizationPreparePipeLine:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_ben_tok_config = config.get_ben_tokenization_config()  
        ben_data_tok = BanTokenization(config=prepare_ben_tok_config)
        ben_data_tok.combine_text_columns()