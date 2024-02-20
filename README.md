# Turkish audio to audio translation
### Scalable machine learning deep learning course - lab2 - turkish audio to multi language audio translator with whisper 

This project is based on Hugging Face transformer that transcribes Turkish audios from a voice recorder, a file or a YouTube video, then generates the translated version in a selection of different languages. 

Hugging Face space for translator UI is [here](https://huggingface.co/spaces/nelanbu/turkish-audio-to-audio-translator).

## Model Performance Improvement Strategies

### Overview

This document outlines two key approaches to enhance the performance of the Whisper ASR model: a model-centric approach and a data-centric approach. By focusing on tuning model-related parameters and exploring additional data sources, the model can be optimized for improved results.

## (a) Model-Centric Approach

### 1. Hyperparameter Tuning

Fine-tuning hyperparameters is a crucial step to optimize the performance of the Whisper ASR model. Experiment with the following hyperparameters:

- **Learning Rate:** Adjust the learning rate to find an optimal balance between convergence speed and fine-tuning stability.

- **Batch Size:** Test different batch sizes to observe their impact on both memory usage and model performance.

- **Regularization:** Introduce regularization techniques such as dropout or weight decay to prevent overfitting.

#### LoRA: Low-Rank Adaptation of Large Language Models
LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

## (b) Data-Centric Approach

1. Identify New Data Sources
To enhance the model's ability to generalize, identify and incorporate new data sources. Explore publicly available datasets or crowd-sourced data to supplement the Common Voice dataset.

2. Data Augmentation
Implement data augmentation techniques to artificially increase the size of the training dataset. This helps improve the model's robustness to variations in speech patterns. Consider the following techniques:

