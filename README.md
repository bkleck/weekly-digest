# weekly-digest
Text classification, summarization, NER and semantic similarity with transformer models. 
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
For this section, we will be building a neural network layer on top of BERT's architecture. We will be downloading the **_BERT-based uncased 12 layer_** from [Tensorflow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2). The reason for doing so is so that we can **_build Keras layers on top of BERT_** to classify it into our required outputs. 

We will slowly drop the number of nodes through 3 **_Dense layers_** to give a final output of 1 or 0 for the signal, supported by **_Dropout layers_** to drop training data to prevent overfitting. The model architecture is shown in the `news_revenue.ipynb` notebook with the following codes:  

`model = build_model(bert_layer, max_len=max_len)`  
`model.summary()`

Training is done with **_patience = 5_**, hence early-stopping will be activated if the validation categorical accuracy does not improve after 5 epochs. This was done for all 7 signals, with the results being shown below:

Signals | Revenue | Product | Market | Partnership | Management | Clinical | Fundraising |
:------ | :-----: | :-----: | :----: | :--------: | :-------: | :------: | :---------: |
F1 score | 0.99   | 0.935   | 0.796  | 0.958       | 0.811       | 0.97    | 0.90        |
macro avg | 0.99   | 0.787   | 0.799  | 0.954       | 0.800       | 0.83    | 0.90        |
weighted avg | 0.99   | 0.929   | 0.795  | 0.958    | 0.806       | 0.97    | 0.90        |
