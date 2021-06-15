# weekly-digest
Text classification, summarization, NER and semantic similarity with transformer models.  

  
## Introduction
With the vast number of news articles related to portfolio companies each week, it is time-consuming for end users to go through each of these articles one by one. Thus, this Weekly Digest aims to accelerate this process. The data input will come from scrapers deployed on Google and Baidu APIs. Firstly, text classification of the presence of various important signals for important investment decisions such as revenue, growth, etc... will be implemented at the sentence level using a BERT model with neural network head. Next, a T5 model will be utilized to perform the task of summarizing the large chunks of text at the company and signal level into readable outputs for end-users. After that, HuggingFace's NER pipeline is used to identify if the company being described in the article is the correct one. Lastly, a roBERTa SentenceTransformer will help to compare semantic similarities between summaries, removing duplicates for end users.
