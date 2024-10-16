from benglasummarization.config.configuration import ConfigurationManager
from benglasummarization.components.train_bn_token import TrainTokenize

class TrainTokenizePipeLine:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        train_ban_tok = config.get_train_token_config()
        train_tok = TrainTokenize(config=train_ban_tok)
        train_tok.train_tokenizer()
        