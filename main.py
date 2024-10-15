from src.benglasummarization.logging import logger
from src.benglasummarization.pipeline.stage01_data_ingestion import DataIngestionPipeline

STAGE_NAME = 'Data Ingestion Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e