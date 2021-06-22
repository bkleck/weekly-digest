# weekly-digest
*Text classification, summarization, NER and semantic similarity with transformer models*
<br/>  
<img src='https://user-images.githubusercontent.com/77097236/122026118-8d3d7580-cdfc-11eb-9613-c7fc3fe20f81.png' width="500" height="300">
<br/>
## Table of Contents
* [Introduction](#introduction)
* [Workflow](#workflow)
  * [Cleaning Scraped News](#1-cleaning-scraped-news)
  * [BERT Signal Classification](#2-text-classification-of-signals)
  * [Data Aggregation](#3-data-aggregation)
  * [T5 Summarization](#4-summarization)
  * [NER](#5-named-entity-recognition-ner)
  * [Semantic Similarity](#6-semantic-similarity)
* [File Descriptions](#file-descriptions)
* [Setup](#setup)
* [How to Use](#how-to-use)
* [Improvements](#improvements)
* [Data Flow](#data-flow)

## Introduction
With the vast number of news articles related to portfolio companies each week, it is **_time-consuming for end users_** to go through each of these articles one by one. Thus, this Weekly Digest aims to accelerate this process. The data input will come from scrapers deployed on Google and Baidu APIs.  

Firstly, **_text classification of the presence of various important signals_** for investment decisions such as revenue, growth, etc... will be implemented at the sentence level using a **_BERT model with neural network head_**. Next, a **_T5 model_** will be utilized to perform the task of **_summarizing the large chunks of text_** at the company and signal level into readable outputs for end-users. 

After that, **_HuggingFace's NER pipeline_** is used to identify if the company being described in the article is the correct one. Lastly, a **_roBERTa SentenceTransformer_** will help to compare **_semantic similarities between summaries_**, removing duplicates for end users.  
<br/>

## Workflow

### 1) Cleaning Scraped News
Input data will come from **_Google and Baidu scrapers_**, with the Chinese articles translated to English. We will only be scraping for articles for our portfolio companies. Firstly, we will perform cleaning at the article level. Many of the articles scraped are actually **_market reports_**, with not much information conveyed about the company in mind, thus these reports will be removed with keyword search. 

There are also many instances of articles only mentioning the company in a small section, hence we will remove articles with **_less than 2 mentions of the company name_**. Finally, articles will be **_converted to sentence level_** for input into our signal classifier models.
<br/>
<br/>
  
### 2) Text Classification of Signals

#### Training
For this section, we will be building a neural network layer on top of BERT's architecture. We will be downloading the **_BERT-based uncased 12 layer_** from [Tensorflow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2). The reason for doing so is so that we can **_build Keras layers on top of BERT_** to classify it into our required outputs. 

We will slowly drop the number of nodes through **_Dense layers_** to give a final output of 1 or 0 for the signal, supported by **_Dropout layers_** to drop training data to prevent overfitting. The model architecture is shown in the `news_revenue.ipynb` notebook with the following codes:  

`model = build_model(bert_layer, max_len=max_len)`  
`model.summary()`

Training is done with **_patience = 5_**, hence **_early-stopping_** will be activated if the validation categorical accuracy does not improve after 5 epochs. This was done for all 7 signals, with the results being shown below:

Signals | Revenue | Product | Market | Partnership | Management | Clinical | Fundraising |
:------ | :-----: | :-----: | :----: | :--------: | :-------: | :------: | :---------: |
F1 score | 0.99   | 0.935   | 0.796  | 0.958       | 0.811       | 0.97    | 0.90        |
macro avg | 0.99   | 0.787   | 0.799  | 0.954       | 0.800       | 0.83    | 0.90        |
weighted avg | 0.99   | 0.929   | 0.795  | 0.958    | 0.806       | 0.97    | 0.90        |

The models were trained on a **_labelled dataset of 3000 rows_** for each signal. Labelling was standardized across the team, with reference to the `Signal Classifier Coding Frame` file.
<br/>
<br/>
#### Inference
With the trained models saved, we will run inference using our models through the `run_models.py` file. However, we decided to keep the original logit score instead of encoding it into 1/0 in the notebook. Afterwards, I performed a **_min-max normalization_** for each of the 7 signals, and kept only the largest score. This is for the purpose of **_cross-comparison across different models_**. This change was made to **_prevent repitition of sentences across summaries_**, causing inconvenience for the end users.

A for loop was used to run the 7 models. Runtime was **_15mins for 1 week of news articles_**, on GeForce RTX 2070, 8GB RAM.
<br/>
<br/>

### 3) Data Aggregation
Firstly, I created short-form names for all portfolio companies in the `vpc_master_070621.xlsx` file for keyword search within the sentences. Afterwards, I will **_keep only the sentences with matches with the name_**, together with **_1 sentence before and after_** if it came from the same search query. The purpose for keeping sentences around the main sentence was to **_give the user context_** about the situation, else it will be hard for he/she to evaluate the piece of information. 

For each signal, we will keep all the rows with **_signal = 1, together with 1 sentence before and after_**, only if it came from the same search query. Then, a **_groupby will be done at the company and signal level_**, with the sentences joined together as a string and the URLs to the articles joined in a list.

An example of how this output will look like:
Company | Sentences | URL | Signal | 
:------ | :-----: | :-----: | :----: | 
Grab | ....   | [URL1, URL2]   | revenue  | 
Grab | ....   | [URL1]   | partnership  | 
Innoviz | ....   | [URL1, URL2]  | fundraising  | 
<br/>

### 4) Summarization
With the large chunks of sentences for each signal after aggregation, I will make use of summarization to convert it to a **_readable format_** for the user, while **_retaining key points_**.

We will be using the T5 model ([Text-To-Text Transfer Transformer](https://huggingface.co/transformers/model_doc/t5.html#tft5forconditionalgeneration)). This text-to-text framework suggests using the same model for all NLP tasks, where inputs are modelled such that the **_model will recognise the task_**, and the output is a **_"text" version_** of the outcome. Hence, for our task, we will have to add the string **_'summarize: '_** infront of each block of text to let the model know the task we want to perform.
![image](https://user-images.githubusercontent.com/77097236/122148216-b73e7880-ce8c-11eb-884f-9f752964a7d7.png)

Another reason why we are using the T5 is because it is an **_abstractive summarizer_**. Hence, it will **_rewrite sentences to make it more coherent_** for the user. This differs from extractive summarizers, where they just pick up sentences directly from text.

The predictions of the model can be created using the `model.generate()` function. With the vast amount of arguments that can be passed in, I referred to this [guide](https://huggingface.co/blog/how-to-generate) for the tuning of the hyperparameters. I made use of **_min_length = 100 and max_length = 300_** for the most informative and concise summaries. **_Beam search_** was used to reduce the risk of missing hidden high probability word sequences. **_Repetition and length penalty_** was used to prevent the repeat of sentences, which often happens for short texts trying to reach the minimum length.
<br/>
<br/>

### 5) Named Entity Recognition (NER)
With the scraping done from an API, it was hard for us to tweak the parameters for control of the output. Hence, there was a lot of **_noise in our dataset_** (*e.g. search for Bites company, get spider bites article, search for Aruna company, get Aruna person*). Hence, I made use of NER to **_remove such irrelevant articles_**.

Trying out both NER capabilities on HuggingFace and SpaCy, I have concluded that [HuggingFace pipeline](https://huggingface.co/transformers/usage.html) outperformed the latter, hence we will be using it instead. After NER labelling is done, we will keep only those summaries with the company name labelled as **_'I-ORG' entities_**. 
<br/>
<br/>

### 6) Semantic Similarity
Another problem that we face is that some of these **_summaries end up pretty similar_** to each other, due to the fact that they came from the same sources. Thus, we will make use of the **_SentenceTransformer with large roBERTa_** from [HuggingFace](https://huggingface.co/sentence-transformers/stsb-roberta-large) to get **_sentence embeddings_**. 

With these embeddings, we will then calculate the **_cosine score between pairs of sentences_** for each company. Hence, we will be dropping duplicate summaries with more than **_70% threshold_** similarity, so that end users do not have to go through the frustration of reading repititions.  
<br/>
<br/>

## File Descriptions
The ipynb notebooks help to illustrate the processes I went through, only use for reference.
- `news_revenue.ipynb` shows the **_training process_** for 2 of 7 signals.
- `news_pipeline.ipynb` shows the **_inference step_** used to run predictions across all 7 signals.
- `summary_transformer.ipynb` shows the rest of the steps from **_Summarization to Semantic Similarity_**.

The python files are used to run the entire process on input dataset of news articles.
- Files should be run in the following order: `clean_scraped_news.py`, `run_models.py`, `prep_summary.py`, `summary_transformer.py`.
<br/>

## Setup
To accelerate the run-time, please ensure CUDA has been activated to support deep-learning processes.
- For `run_models.py`, ensure CUDA is installed to **_activate GPU on Tensorflow_** by referring [here](https://www.tensorflow.org/install/gpu).
  - Install CUDA toolkit version 11.0 [here](https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal).
  - Rename cusolver64_10.dll to cusolver64_11.dll in bin folder.
  - Install cuDNN [here](https://developer.nvidia.com/user).
  - Copy the file cudnn64_8.dll and put into the CUDA bin folder.
  
- For `summary_transformer.py`, ensure CUDA is installed to **_activate GPU on Pytorch_**.
  - Download CUDA 10.1 [here](https://developer.nvidia.com/cuda-10.1-download-archive-base) as Pytorch does not support version 11 and above.
  - Follow the steps [here](https://varhowto.com/install-pytorch-cuda-10-1/).
<br/>

## How to Use
If you just want to use the models for inference on a new dataset, follow the steps here.
1. Command prompt into the folder and create the virtual environment with `python -m venv venv/` (only do it one time).
2. Activate the environment with `{directory}\venv\Scripts\activate`.
3. If it is your first time, install libraries with `pip install -r requirements.txt`.
4. Input the newly scraped data files into the **_data/scrape folder_** in the following format: **_150621_google.csv and 150621_baidu.csv_**.
5. Run `clean_scraped_news.py`. Output files will be in the same folder with same date: **_150621_clean_google.csv and 150621_clean_baidu.csv_**.
6. Run `run_models.py`. Runtime is ~15mins. Please ensure GPU is activated. Output files will be in **_data folder_**: **_150621_signal_google_predictions.csv and 150621_signal_baidu_predictions.csv_**.
7. Run `prep_summary.py`. Output file will be in **_data/summary folder_**: **_150621_before_summary.csv_**.
8. Run `summary_transformer.py`. Runtime is ~20mins. Please ensure GPU is activated. Output file will be in same folder: **_150621_after_summary.csv_**. 
<br/>

## Improvements
*Improvements made here have been updated into the most recent python files. Old files are in the archive folder.*
- [x] **Few-shot learning**  
With the model architecture in place, we made use of few-shot learning to improve our output. Using 10 manually created summaries, I ran a small training loop on 10 epochs to **_finetune the model weights_** to suit our use case.  
This **_improved the coherence_** of the summaries and helped to extract the correct information, as well as **_improve grammar, punctuation and readability_**. Hence, we will **_load weights from the summary_weights folder_** instead of from HuggingFace. This is shown in the `summary_tuning.ipynb` notebook. 

- [ ] **PEGASUS model**  
I also tried the PEGASUS model trained for [financial summarization](https://huggingface.co/human-centered-summarization/financial-summarization-pegasus). However, given the **_small model size and limited training data_**, it did not perform very well. Furthermore, it was trained to **_generate 1 sentence outputs_**, which led to alot of noise when generating longer summaries, hence it did not suit our use case.

- [x] **Logic change**  
Looking at the output, we realise that our original intention to group by signals so that users can see all relevant informationg pertaining to one signal, has caused a problem. The **_summaries contain information from various contexts_** originating from numerous sources of news, which may lead to some inaccuracy in presentation. Hence, we have decided to change the logic and do a **_clustering by indexes_** instead.  
<img src='https://user-images.githubusercontent.com/77097236/122890063-ef0a5c00-d375-11eb-9b47-2bdfdcfbffa5.png' width="300" height="200" align="left">

<br/>

> Hence, this new logic serves to **_group up continuous columns of text_** if they belong to the same article, **_retaining the context_** of the situation. This will introduce greater variance in lengths of text, hence **_2 separate summarizer arguments_** will be used to handle short and long texts. **_Min_length and length_penalty arguments will be removed_** to reduce instances where the ending of summaries are cut off.
<br/>
<br/>

## Data Flow
The image below shows the full flow of the data between python files:
<img src='https://user-images.githubusercontent.com/77097236/122183428-d0abe880-cebd-11eb-8a34-f9d408554a5d.jpg' width="800" height="600">
