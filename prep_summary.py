import pandas as pd
import numpy as np

## Load data
print("Please type today's date in the following format (110621)")
# all files will be saved with today's date at the front
# input file must also be in the same format
date = input('Please enter date: ')

input_file_1 = 'data/' + date + '_signal_google_predictions.csv'
df1 = pd.read_csv(input_file_1, index_col=0)
df1.reset_index(drop=True, inplace=True)
df1.dropna(subset=['Search Query'], inplace=True)

input_file_2 = 'data/' + date + '_signal_baidu_predictions.csv'
df2 = pd.read_csv(input_file_2, index_col=0)
df2.reset_index(drop=True, inplace=True)
df2.dropna(subset=['Search Query'], inplace=True)

# this is the file containing all the manually created short names
df3 = pd.read_excel(open('data/vpc_master_070621.xlsx', 'rb'), sheet_name='Master')

# remove companies we are not tracking
df3.dropna(subset=['Company Name for Search'], inplace=True)
df3 = df3[['Company Name for Search', 'short_names']]

# create a dictionary for mapping search query to short names
s = df3.groupby('Company Name for Search')['short_names'].apply(lambda x: x.tolist())
name_dict = s.to_dict()


# Add in short names to find match

def find_english_match(df1, name_dict):
    # map short names to search queries
    # remove all whitespaces
    df1['short_name'] = df1['Search Query'].apply(lambda x: name_dict[x][0].replace(' ',''))

    # match = 1 if short name appears in article
    match_list = []
    for idx,row in df1.iterrows():
        if row['short_name'] in row['article']:
            match_list.append(1)
        else:
            match_list.append(0)

    df1['match'] = match_list

    # Add in 1 sentence before and after match

    # remove indexes if they exceed this range (from original df)
    indexes = list(df1.index.values)
    first_idx = indexes[0]
    last_idx = indexes[-1]

    # retrieve indexes with match = 1
    df3 = df1[df1['match'] == 1.0]
    idx_list = list(df3.index.values)

    # add 1 sentence before and 1 sentence after to the list
    new_list = []
    for i in idx_list:
        prior = i - 1
        new_list.append(prior)

        after = i + 1
        new_list.append(after)

    idx_list = idx_list + new_list

    # remove duplicates
    idx_list = list(dict.fromkeys(idx_list))
    # sort in ascending order
    idx_list.sort()

    # remove first and last element if it exceeds the index range
    if idx_list[0] < first_idx:
        idx_list.pop(0)

    elif idx_list[-1] > last_idx:
        idx_list.pop(-1)

    for idx in idx_list:
        df1.loc[idx, 'match'] = 1

    return df1

def find_chinese_match(df1):

    # match = 1 if short name appears in article
    match_list = []
    for idx,row in df1.iterrows():
        if row['Search Query'] in row['article']:
            match_list.append(1)
        else:
            match_list.append(0)

    df1['match'] = match_list

    # Add in 1 sentence before and after match

    # remove indexes if they exceed this range (from original df)
    indexes = list(df1.index.values)
    first_idx = indexes[0]
    last_idx = indexes[-1]

    # retrieve indexes with match = 1
    df3 = df1[df1['match'] == 1.0]
    idx_list = list(df3.index.values)

    # add 1 sentence before and 1 sentence after to the list
    new_list = []
    for i in idx_list:
        prior = i - 1
        new_list.append(prior)

        after = i + 1
        new_list.append(after)

    idx_list = idx_list + new_list

    # remove duplicates
    idx_list = list(dict.fromkeys(idx_list))
    # sort in ascending order
    idx_list.sort()

    # remove first and last element if it exceeds the index range
    if idx_list[0] < first_idx:
        idx_list.pop(0)

    elif idx_list[-1] > last_idx:
        idx_list.pop(-1)

    for idx in idx_list:
        df1.loc[idx, 'match'] = 1

    return df1


df1 = find_english_match(df1, name_dict)
df2 = find_chinese_match(df2)

# combine english and chinese sentences together
df = pd.concat([df1, df2])

# Create summarization data with 1 sentence before and after


# this is the new logic: group up continous sentences instead of signals
# allow better context in summaries to prevent confusion

def prep_summary(df1):
    # remove sentences without matches
    df = df1[df1['match'] == 1]
    df.reset_index(inplace=True)

    # get dataframe full of signals
    signal_df = df.loc[:,'revenue':'fundraising']

    # remove rows with all signal = 0
    signal_df = signal_df.replace(0, np.nan)
    signal_df = signal_df.dropna(how='all', axis=0)
    signal_df = signal_df.replace(np.nan, 0)
    idx_list = list(signal_df.index.values)

    # keep only the indexes of rows with at least 1 signal
    df = df.iloc[idx_list]

    # aggregate data together by a multi-index of search query, followed by index
    # this is for the purpose of grouping up continuous texts together later on (e.g. 2731, 2732, 2733)
    final_df = df.groupby(['Search Query','index']).agg({
                    'article': ' '.join, # change from fullstop to space. Ensure summarizer reads sentence properly
                    'URL': list
                })
    final_df = pd.DataFrame(final_df)

    # reset the index so that we obtain the multi-indexes as columns instead
    final_df.reset_index(inplace=True)

    final_df.rename(columns={'index':'old_index'}, inplace=True)
    final_df.reset_index(inplace=True)

    # create a new empty column for input of values during iterrows 
    final_df['new_index'] = np.nan

    # this section is the part where we will use the following logic to join continuous texts together (e.g. 2731, 2732, 2733)
    # firstly, if it is the first row, we will just keep the old index as the new index
    # if not, we will implement the following logic:
    # if the old index of the current row is +1 away from the previous row, we set the new index of current row to be same as previous row
    # else, the new index will be set as old index
    # in this way, all continuous indexes will be set with the same new index, for grouping together later on
    for idx,row in final_df.iterrows():
        if idx == 0:
            final_df.loc[idx, 'new_index'] = row['old_index']

        else:
            previous = final_df.loc[idx-1, 'old_index']
            if row['old_index'] == previous + 1:
                final_df.loc[idx, 'new_index'] = final_df.loc[idx-1, 'new_index']

            else:
                final_df.loc[idx, 'new_index'] = final_df.loc[idx, 'old_index']
    
    df = final_df.groupby(['Search Query','new_index']).agg({
                'article': ' '.join,
                'URL': sum # use sum to join lists together
            })
    df = pd.DataFrame(df)
    return df


df  = prep_summary(df)


output_file = 'data/summary/' + date + '_before_summary.csv'

df.to_csv(output_file)
print(' ')
print('predictions has been saved to data folder!')