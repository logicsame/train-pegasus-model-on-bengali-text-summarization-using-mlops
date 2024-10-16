from torch.utils.data import Dataset
from transformers import PegasusTokenizer
import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from src.benglasummarization.logging import logger
from src.benglasummarization.entity.config_entity import ModelTrainingConfig


class BengaliSummaryDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer: PegasusTokenizer, config: ModelTrainingConfig):
        self.config = config
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_input_length,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            summary,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_output_length,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = labels['input_ids'].squeeze()

        # Replace padding token id's with -100 to ignore them during loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        df = pd.read_csv(self.config.data_dir)
        df = df.head(1000)
        texts = df['main'].tolist()
        summaries = df['sum3'].tolist()
        return train_test_split(texts, summaries, test_size=0.1, random_state=42)

    def create_datasets(self, train_texts, train_summaries, val_texts, val_summaries):
        tokenizer = PegasusTokenizer.from_pretrained(self.config.ben_tokenizer_dir)
        train_dataset = BengaliSummaryDataset(train_texts, train_summaries, tokenizer, self.config)
        val_dataset = BengaliSummaryDataset(val_texts, val_summaries, tokenizer, self.config)
        return train_dataset, val_dataset, tokenizer

    def train(self):
        # Load and split data
        train_texts, val_texts, train_summaries, val_summaries = self.load_data()
        
        # Create datasets and tokenizer
        train_dataset, val_dataset, tokenizer = self.create_datasets(train_texts, train_summaries, val_texts, val_summaries)
        
        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialize model
        model = PegasusForConditionalGeneration.from_pretrained(self.config.model_name).to(self.device)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=len(train_dataloader) * self.config.num_epochs)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / self.config.accumulator_steps
                loss.backward()

                total_loss += loss.item()

                if (step + 1) % self.config.accumulator_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({'loss': total_loss / (step + 1)})

            progress_bar.close()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()

            val_loss /= len(val_dataloader)
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

           

        logger.info(f"Training Completed")
        save_path = os.path.join(self.config.save_trained_model_dir)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f'Model Saved to {self.config.save_trained_model_dir}')

