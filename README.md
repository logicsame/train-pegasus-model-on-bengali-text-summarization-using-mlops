
# 🚀 Bengali Text Summarization with Pegasus

This project demonstrates the training of a **Pegasus large model** on Bengali text summarization from scratch. Pegasus is a powerful model specifically designed for text summarization, but it lacks pretraining on Bengali text, which motivated us to adapt it for Bengali language tasks. The project aims to enhance Pegasus's performance on Bengali summarization using a custom Bengali tokenizer and dataset. 

## Model Overview

The Pegasus model is known for its state-of-the-art performance in text summarization for various languages, but we extended its capabilities by adapting it for Bengali text, with the following key highlights:
- **Custom Bengali Tokenization**: Created with a vocabulary of approximately 91,000 unique tokens for handling Bengali characters and words efficiently.
- **Training Details**:
    - Trained the model for **1 epoch on a 10k subset of data**, with promising results even at this small scale.
    - Built a scalable batch processing pipeline to accommodate **160k Bengali data samples** for further training.
    - Managed the training on limited GPU resources, optimizing batch sizes and model parameters accordingly.

## Training Setup

Due to the limited availability of pretrained models for Bengali summarization, this project initiates training from scratch:
1. **Data Preparation**: Preprocessed and tokenized a dataset of 160,000 Bengali text samples for training. Training began with 10,000 samples, yielding strong initial results.
2. **Tokenizer**: We built and trained a custom tokenizer tailored to Bengali, yielding a vocabulary of around 91,000 tokens.
3. **Model Training**: Initially trained for 1 epoch on a small subset, showing promising accuracy. Future iterations will scale to the full dataset.

## Results and Future Work

Initial testing with 10,000 samples over one epoch demonstrated Pegasus’s potential in Bengali summarization tasks. Future objectives include:
- **Increasing Training Epochs**: Extend training on the full 160,000-sample dataset.
- **Fine-Tuning and Hyperparameter Optimization**: Further improve summarization quality by refining hyperparameters and model architecture.

## Installation and Usage

To set up this project, you will need to install the necessary packages and run the app using Streamlit:

1. **Clone the Repository**:
    ```bash
    git clone github.com/logicsame/train-pegasus-model-on-bengali-text-summarization-using-mlops
    cd bengali-text-summarization
    ```

2. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## License

This project is licensed under the MIT License.

## Contributing

Feel free to contribute! Whether it’s improving the Bengali tokenizer, fine-tuning the model, or exploring additional datasets, your contributions are welcomed.




---
title: Bengali Text Summarization
emoji: 🚀
colorFrom: purple
colorTo: yellow
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

