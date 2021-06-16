import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

print("Please type today's date in the following format (110621)")
# all files will be saved with today's date at the front
# input file must also be in the same format
date = input('Please enter date: ')
input_file_1 = 'data/scrape/' + date + '_google.csv'
input_file_2 = 'data/scrape/' + date + '_baidu.csv'

google = pd.read_csv(input_file_1, index_col=0)
google.rename(columns={'Media':'Source', 'Seach Query':'Search Query'}, inplace=True)

baidu = pd.read_csv(input_file_2, index_col=0)
baidu.drop(columns=['Search Query','Abbreviated Search Query','Chinese Title','Chinese Article'], inplace=True)
baidu.rename(columns={'Media':'Source', 'Company Name':'Search Query'}, inplace=True)

# convert all columns to strings
eng_columns = list(google) # Creates list of all column headers
google[eng_columns] = google[eng_columns].astype(str)

chin_columns = list(baidu) # Creates list of all column headers
baidu[chin_columns] = baidu[chin_columns].astype(str)


## Clean articles
def clean_article(df):
  df['lower'] = df['Article'].apply(lambda x: x.lower())
  df['name'] = df['Search Query'].apply(lambda x: x.lower())
  print('')
  print('Original no. of articles: ' + str(df.shape))
  print('')

  # remove any article with these words inside
  # irrelevant articles
  substrings = ['market research', 'research study', 'research analysis',
                'market study', 'market report']
  report_count = []
  for idx, row in df.iterrows():
    total = 0
    string = row['lower']
    for substring in substrings:
      count = string.count(substring)
      total += count
    
    report_count.append(total)

  df['report_count'] = report_count
  df = df[df['report_count'] == 0]
  print('No. of articles after removing reports: ' + str(df.shape))
  print('')


  # remove those articles with less than 2 mentions of company name
  count_list = []
  for idx, row in df.iterrows():
    string = row['lower']
    substring = row['name']

    count = string.count(substring)
    count_list.append(count)

  df['count'] = count_list
  df = df[df['count'] >= 1]
  print('')
  print('No. of articles after removing less than 1 mention of company: ' + str(df.shape))
  print(' ')

  return df

print('Google news:')
clean_google = clean_article(google)
clean_google.drop(columns=['report_count', 'name', 'count'], inplace=True)

clean_google.rename(columns={'Article':'article'}, inplace=True)
df1 = clean_google.loc[:, 'Search Query': 'article']
df1.reset_index(inplace=True, drop=True)

print('')
print('Baidu news:')
clean_baidu = clean_article(baidu)
clean_baidu.drop(columns=['report_count', 'name', 'count'], inplace=True)

clean_baidu.rename(columns={'Article':'article'}, inplace=True)
df2 = clean_baidu.loc[:, 'Search Query': 'article']
df2.reset_index(inplace=True, drop=True)


# convert from article to sentence level to put into the classifier models
def articles_to_sentence(df):
    df = df.dropna()
    df["article"] = df["article"].apply(lambda x: sent_tokenize(x))
    df = df.explode("article")
    df.reset_index(inplace=True)

    # remove articles that contain the string 'http'
    df = df[~df['article'].str.contains('http')]
    df.drop(columns=['index', 'Source', 'Title'], inplace=True)

    # remove sentences that do not contain alphabets
    df =  df[df['article'].str.contains('[A-Za-z]')]

    # remove very short sentences that do not convey anything
    df = df[df.article.apply(lambda x: len(str(x))>20)]

    return df

google_sentences = articles_to_sentence(df1)
baidu_sentences = articles_to_sentence(df2)

output_file_1 = 'data/scrape/' + date + '_clean_google.csv'
google_sentences.to_csv(output_file_1)

output_file_2 = 'data/scrape/' + date + '_clean_baidu.csv'
baidu_sentences.to_csv(output_file_2)

print('Cleaned news files have been saved to data/scrape folder!')