import pandas as pd
from pathlib import Path
from src.benglasummarization.logging import logger
from tqdm.notebook import tqdm
from src.benglasummarization.entity.config_entity import BanTokenizationConfig
class BanTokenization:
    def __init__(self, config: BanTokenizationConfig):
        self.config = config

    def combine_text_columns(self, text_columns=['main']):
        df = pd.read_csv(self.config.source_dir)

        # Ensure save_dir is a Path object
        save_dir = Path(self.config.save_dir)
        
        # Create the directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Combine save_dir and output_file to form the output path
        output_txt_file = save_dir / self.config.output_file
        
        # Write the combined text data to the output file
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            for index, row in tqdm(df.iterrows(), total=len(df)):
                combined_text = ' '.join(str(row[col]) for col in text_columns)
                f.write(combined_text + '\n')

        # Log the success of the operation
        logger.info(f"All text data has been combined into {output_txt_file}")