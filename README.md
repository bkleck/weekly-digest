# weekly-digest
Text classification, summarization, NER and semantic similarity with transformer models.  
<br/>

## Introduction
With the vast number of news articles related to portfolio companies each week, it is **_time-consuming for end users_** to go through each of these articles one by one. Thus, this Weekly Digest aims to accelerate this process. The data input will come from scrapers deployed on Google and Baidu APIs.  

Firstly, **_text classification of the presence of various important signals_** for investment decisions such as revenue, growth, etc... will be implemented at the sentence level using a **_BERT model with neural network head_**. Next, a **_T5 model_** will be utilized to perform the task of **_summarizing the large chunks of text_** at the company and signal level into readable outputs for end-users. 

After that, **_HuggingFace's NER pipeline_** is used to identify if the company being described in the article is the correct one. Lastly, a **_roBERTa SentenceTransformer_** will help to compare **_semantic similarities between summaries_**, removing duplicates for end users.  
<br/>

## Workflow

### 1) Cleaning scraped news

