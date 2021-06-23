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


def prep_summary(df1):
    # remove sentences without matches
    df = df1[df1['match'] == 1]
    df.reset_index(drop=True, inplace=True)

    indexes = list(df.index.values)
    first_idx = indexes[0]
    last_idx = indexes[-1]

    signals = ['revenue','product','market','partnership','mgmt','clinical','fundraising']
    df_list = []

    for signal in signals:

        # retrieve indexes with signal = 1
        df1 = df[df[signal] == 1.0]
        idx_list = list(df1.index.values)

        # add 1 sentence before and 1 sentence after to the list
        new_list = []
        for i in idx_list:

            current_name = df.loc[i, 'short_name']

            # get the indexes of 1 sentence prior and after
            # get company name in before and after row
            # if index our of range error, just set name to 'NIL'
            # compare the name of company in that row to name of current row
            # if same, append the index to a new list

            prior = i - 1
            try:
                prior_name = df.loc[prior, 'short_name']
            except:
                prior_name = 'NIL'
            
            if prior_name == current_name:
                new_list.append(prior)
        
            after = i + 1
            try:
                after_name = df.loc[after, 'short_name']
            except:
                after_name = 'NIL'

            if after_name == current_name:
                new_list.append(after)

        idx_list = idx_list + new_list

        # remove duplicates
        idx_list = list(dict.fromkeys(idx_list))
        # sort in ascending order
        idx_list.sort()

        # remove first and last element if it exceeds the index range
        # only do this if the index list is not empty
        if len(idx_list) > 0:
            if idx_list[0] < first_idx:
                idx_list.pop(0)

            elif idx_list[-1] > last_idx:
                idx_list.pop(-1)

            signal_df = df.iloc[idx_list]
            # group by company names, aggregating all the texts into 1 row
            # aggregate all the URLs into a list
            signal_df = signal_df.groupby('Search Query').agg({
                'article': '.'.join,
                'URL': list
            })

            signal_df = pd.DataFrame(signal_df)
            signal_df['signal'] = signal
            df_list.append(signal_df)

    df = pd.concat(df_list)

    # dropping ALL duplicte values
    df.drop_duplicates(subset ="article",
                        keep = False, inplace = True)
    
    # remove duplicate URLs within list
    df['URL'] = df['URL'].apply(lambda x: list(set(x)))
    
    return df

df = prep_summary(df)

output_file = 'data/summary/' + date + '_before_summary.csv'

df.to_csv(output_file)
print(' ')
print('predictions has been saved to data folder!')
