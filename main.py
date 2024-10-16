from src.benglasummarization.logging import logger
from src.benglasummarization.pipeline.stage01_data_ingestion import DataIngestionPipeline
from src.benglasummarization.pipeline.stage_02_prepare_ben_tok import BenTokenizationPreparePipeLine
from src.benglasummarization.pipeline.stage_03_train_ban_token import TrainTokenizePipeLine
STAGE_NAME = 'Data Ingestion Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
STAGE_NAME = 'Prepare Ban Tokeniation Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   Ban_Token = BenTokenizationPreparePipeLine()
   Ban_Token.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
STAGE_NAME = 'Training Bengla Tokenization Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   Train_Ban_Token = TrainTokenizePipeLine()
   Train_Ban_Token.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

