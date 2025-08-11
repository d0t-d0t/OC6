from ollama import chat
import json
from ollama import Client
import pandas as pd

from Preproc import tokenize_and_preprocess

client = Client(
  host='http://localhost:11434',
  # headers={'x-some-header': 'some-value'}
)

def classify_tweet(tweet,
                   client=client,
                   model = 'gemma3:12b',#'deepseek-r1:14b'#
                     ):
    prompt = f"""Classify the following tweet mood in one token only. 
                the scale is 0=negative, 1=neutral, 2=positive.
                Do not answer anything else than 0, 1, or 2 
                Do not provide any explanation: {tweet} """
    response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    try:
        classification = int(response['message']['content'].strip())
        return classification
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
def merge_results(current_df,result_df):
    current_df = current_df.merge(result_df[['llm_tweet','llm_target']], how='left',
            on='llm_tweet')
    

def classify_in_batches(df, batch_size=1000):
    file_name = 'llm_reclassified_data.csv'

    remaining_df = df.loc[df['llm_target'].isna()]
    while len(remaining_df)>0:

        current_batch = remaining_df.sample(batch_size)

        print(f'On {len(remaining_df)} remaining tweet, batching {batch_size}')

        current_batch['llm_target'] = current_batch['llm_tweet'].apply(classify_tweet)

        merge_results(df,current_batch)
        remaining_df = df.loc[df['llm_target'].isna()]

        df.to_csv(file_name, mode='w', index=False)


if __name__ == "__main__":
    tokenize_and_preprocess(df, 'tweet', 
                            cible='llm_tweet',
                            actions=[
                                    'REPLACE_MENTIONS',
                                    'REPLACE_URLS',
                                    'FORMAT'
                                    ])
    try:
        try:
            existing_df = pd.read_csv('llm_reclassified_data.csv')
        except:
            existing_df = df.copy()

        classify_in_batches(existing_df)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
