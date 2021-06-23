import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import pipeline
import pandas as pd
import torch
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import datetime
from sentence_transformers import SentenceTransformer, util


## Load pre-trained model and data for summarization


# load data
print('')
print("Please type today's date in the following format (110621)")
# all files will be saved with today's date at the front
# input file must also be in the same format
date = input('Please enter date: ')
input_file = 'data/summary/' + date + '_before_summary.csv'

total_time = datetime.datetime.now() # track run-time
df = pd.read_csv(input_file)

# add in code-word for T5 to perform summarization
df['article'] = 'summarize: ' + df['article']


# choose which model you are using
model_name = 't5-base'

# ensure gpu is being utilized
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device) # send model to gpu

tokenizer = T5Tokenizer.from_pretrained(model_name)


## Run predictions


# https://huggingface.co/blog/how-to-generate

def summarize(text):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True, max_length=512)
  input_ids = input_ids.to(device) # ensure inputs and model on same device

  # generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
  generated_ids = model.generate(input_ids=input_ids, 
                                 min_length=100, 
                                 max_length=300, 
                                 length_penalty=2.0, # set penalty for increasing length (> 1 means force to increase length)
                                 num_beams=4, # reduces the risk of missing hidden high probability word sequences, reduces repeats, makes words more surprising and less probable 
                                #  early_stopping=True,
                                #  no_repeat_ngram_size=4 # no 4-gram appears twice
                                 repetition_penalty = 2.0,
                                #  do_sample=True,
                                #  top_p=0.92,
                                #  top_k=0
                                 )
  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]

begin_time = datetime.datetime.now() # track run-time
df['summary'] = df['article'].apply(lambda x: summarize(x))
print('Done with summarization!')
print(datetime.datetime.now() - begin_time)
print('')


## Perform NER with HuggingFace


nlp = pipeline("ner", device=0) # activate gpu on pipeline
print('This shows if you are using GPU:')
print(nlp.device)
print('')

begin_time_2 = datetime.datetime.now() # track run-time
df['ner'] = df['summary'].apply(lambda x: nlp(x))
print('Done with NER!')
print(datetime.datetime.now() - begin_time_2)
print('')

def extract_org(a_list):
  string = ''
  # keep the entities with the label 'Organization'
  for i in a_list:
    if i['entity'] == 'I-ORG':
      entity = ' ' + i['word']
      string += entity
  
  # join all those words with ##, it is Huggingface's way of saying it came from the same entity
  string = string.replace(' ##', '')
  return string

df['entity'] = df['ner'].apply(lambda x: extract_org(x))

# keep only those rows with their company names as 'ORG' entities
idx_list = []
for idx,row in df.iterrows():
  if row['Search Query'].lower() in row['entity'].lower():
    idx_list.append(idx)

final_df = df.iloc[idx_list]

final_df.drop(columns=['ner','entity'], inplace=True)
final_df.reset_index(drop=True, inplace=True)



##  perform semantic similarity with SentenceTransformer


st_model = SentenceTransformer('stsb-roberta-base')

# clean quotation marks from "" to '', or else model will throw error
final_df['summary'] = final_df['summary'].apply(lambda x: x.replace('"', "'"))

# this function returns the semantic similarity between 2 texts
def similar_sentence(sentence1, sentence2):
  embedding1 = st_model.encode(sentence1, convert_to_tensor=True)
  embedding2 = st_model.encode(sentence2, convert_to_tensor=True)
  cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
  return cosine_scores

# get the list of unique companies to iterate through them
unique_list = final_df['Search Query'].unique()

# this function utilizes semantic similarity to remove "similar" summaries across the dataframe
def remove_dups(unique_list, df):
  idx_list = []
  for name in unique_list:
    # get each group of dataframe by unique company name
    small_df = df[df['Search Query'] == name].reset_index()

    for idx,row in small_df.iterrows():
      text1 = row['summary']

      # compare each row with every other next row in the dataframe
      while idx < len(small_df) - 1:
        idx += 1
        # keep old index for reference to original df
        index2 = small_df.loc[idx, 'index']
        text2 = small_df.loc[idx, 'summary']
        score = similar_sentence(text1, text2)

        # if the semantics are more than 70% similar, append the old index to a list for dropping
        if score > 0.7:
          idx_list.append(index2)

  return idx_list

idx_list = remove_dups(unique_list, final_df)
df = final_df.drop(final_df.index[idx_list])
df = df.sort_values(by=['Search Query'])


output_file = 'data/summary/' + date + '_after_summary.csv'

df.to_csv(output_file)
print(' ')
print('predictions has been saved to data/summary folder!')
print('Total time taken:')
print(datetime.datetime.now() - total_time)
