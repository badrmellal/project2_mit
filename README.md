# MIT RAG System with Neural RLHF

A Retrieval-Augmented Generation (RAG) system with Neural Network-based Reinforcement Learning from Human Feedback (RLHF) for improved document question-answering.

**Authors**: Badr Mellal and Iliass Benayed.

**Date**: 30 April 2025.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/badrmellal/project2_mit/blob/main/mitrag_rlhf.ipynb)

[🎥 Watch System Demo Video](https://drive.google.com/file/d/10_64G6wrzoX1TDWNj-UjsOdQ-hDJ6kuV/view?usp=drive_link)


## Project Overview

This project implements a question-answering system that combines:

1. **Document Processing & Chunking**: Intelligent document segmentation for effective retrieval
2. **Vector Embeddings & Semantic Search**: Using transformer models to understand document meaning
3. **Language Model Integration**: Generating relevant and accurate answers
4. **Neural RLHF**: Learning from user feedback to continuously improve system outputs

The RAG approach augments the capabilities of language models with external knowledge, while RLHF enables the system to learn from user interactions, creating a self-improving question-answering system.

## Features

- **Document Support**: Process PDF, DOCX, and TXT files
- **Adaptive Chunking**: Dynamic chunk sizing based on document structure and length
- **Vector Embeddings**: Semantic understanding with SentenceTransformer
- **Vector Database**: Efficient storage and retrieval with ChromaDB
- **Answer Generation**: Using Hugging Face models (default: Flan-T5)
- **Neural Reward Model**: DistilBERT model trained to predict user satisfaction
- **PPO Implementation**: Policy optimization based on human feedback
- **RLHF Fine-tuning**: Two-phase training (supervised fine-tuning followed by PPO)
- **Interactive UI**: Built with IPython widgets for easy interaction
- **Feedback Collection**: Star-based rating system with analytics
- **Minimal Feedback Requirements**: Works with as few as 10 feedback samples

## Architecture

The system consists of the following components:

1. **DocumentProcessor**: Handles document parsing and chunking
2. **MitRetriever**: Manages vector embeddings and search
3. **HuggingFaceLLM**: Integrates with language models for answer generation
4. **FeedbackSystem**: Collects ratings and trains RLHF models
5. **MitRAGSystem**: Orchestrates all components into a unified system

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- SentenceTransformer
- ChromaDB
- pandas, numpy, matplotlib
- PyPDF2, python-docx
- ipywidgets
- Google Colab (recommended for GPU access)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/badrmellal/project2_mit.git
cd mit-rag-rlhf
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook in Google Colab or locally:
```bash
jupyter notebook MitRag&Rlhf.ipynb
```

## Usage Guide

### 1. Upload Document

Click the "Upload Document" button and select a PDF, DOCX, or TXT file to process.

### 2. Process Document

Click "Process Document" to chunk the document and create vector embeddings.

### 3. Ask Questions

Type your question in the text field and click "Search" to get an answer based on the document.

### 4. Provide Feedback

Rate the quality of the answer from 1-5 stars and click "Submit" to provide feedback.

### 5. Train RLHF Model

After collecting at least 10 feedback samples, click "Train RLHF Model" to create a personalized model based on your preferences.

### 6. Toggle RLHF

Use the checkbox to switch between the base model and your RLHF-enhanced model.

### 7. View Analytics

Click "Show Analytics" to see feedback statistics and insights.

## RLHF Implementation Details

The RLHF implementation consists of two main components:

1. **Neural Reward Model**: 
   - Based on DistilBERT architecture
   - Fine-tuned to predict user ratings (1-5) for question-answer pairs
   - Takes both the question and generated answer as input
   - Outputs a scalar reward value

2. **PPO Fine-tuning**:
   - Two-phase approach: supervised fine-tuning followed by PPO
   - Initial alignment on high-rated examples
   - Policy optimization with KL divergence penalty to prevent over-optimization
   - Reference model used to maintain output quality

The RLHF training process:
1. Collects user feedback (minimum 10 samples)
2. Trains the reward model to predict ratings
3. Fine-tunes the base LLM using a weighted approach favoring high-rated examples
4. Performs PPO optimization to maximize predicted rewards
5. Saves the resulting model for inference

## Future Improvements

- Integration with more advanced language models
- Distributed training for larger feedback datasets
- Hybrid retrieval methods (keyword + semantic)
- Active learning approach to optimize feedback collection


## Acknowledgments

- Hugging Face for model implementations
- The transformers, SentenceTransformer, and ChromaDB communities
- Our module instructor Abdelhak Mahmoudi for guidance and support
