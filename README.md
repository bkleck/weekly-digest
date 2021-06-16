# weekly-digest
*Text classification, summarization, NER and semantic similarity with transformer models*
<br/>  
<img src='https://user-images.githubusercontent.com/77097236/122026118-8d3d7580-cdfc-11eb-9613-c7fc3fe20f81.png' width="500" height="300">
<br/>

## Introduction
With the vast number of news articles related to portfolio companies each week, it is **_time-consuming for end users_** to go through each of these articles one by one. Thus, this Weekly Digest aims to accelerate this process. The data input will come from scrapers deployed on Google and Baidu APIs.  

Firstly, **_text classification of the presence of various important signals_** for investment decisions such as revenue, growth, etc... will be implemented at the sentence level using a **_BERT model with neural network head_**. Next, a **_T5 model_** will be utilized to perform the task of **_summarizing the large chunks of text_** at the company and signal level into readable outputs for end-users. 

After that, **_HuggingFace's NER pipeline_** is used to identify if the company being described in the article is the correct one. Lastly, a **_roBERTa SentenceTransformer_** will help to compare **_semantic similarities between summaries_**, removing duplicates for end users.  
<br/>

## Workflow

### 1) Cleaning Scraped News
Input data will come from **_Google and Baidu scrapers_**, with the Chinese articles translated to English. We will only be scraping for articles for our portfolio companies. Firstly, we will perform cleaning at the article level. Many of the articles scraped are actually **_market reports_**, with not much information conveyed about the company in mind, thus these reports will be removed with keryword search. 

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

