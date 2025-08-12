from collections import Counter
# from sklearn.feature_extraction.text import CountVectorizer
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
# import tensorflow as tf
# nltk english lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

# def observ_token(df, 
#                  token_column='tokenised_desc'
#                  ):


#     word_freq_df = get_word_frequency(df,token_column)

#     # plt.figure(figsize=(10, 6))
#     # sns.histplot(data=word_freq_df.head(50), y='Word',x='Frequency',palette="hls",
#     #             multiple="stack",
#     #             log_scale=True
#     #             )
#     # plt.title(f'Visualisation des stop words')
#     # plt.show()
#     word_freq_df.head(50).plot(x='Word', y='Frequency', kind='bar',figsize=(8,4))
#     word_freq_df.tail(50).plot(x='Word', y='Frequency', kind='bar',figsize=(8,4))

#     word_dict = pd.Series(word_freq_df.Frequency.values,index=word_freq_df.Word).to_dict()
#     wordcloud = WordCloud(
#         width=600,
#         height=300,
#         max_words=100,
#         colormap='viridis',
#         background_color='white',
#         stopwords={'the', 'and', 'is', 'in', 'that', 'of', 'it'}
#     ).generate_from_frequencies(word_dict)

#     plt.figure(figsize=(8, 4))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
#     return word_freq_df


def get_word_frequency(df, token_column):
    # Assuming df is your existing DataFrame with a 'tokenised_desc' column
    all_tokens = [token for sublist in df[token_column] for token in sublist]

    overall_variety = len(set(all_tokens))
    overall_word_freq = dict(Counter(all_tokens))

    # print(f"Overall Variety: {overall_variety}")
    # print("Overall Word Frequency:", overall_word_freq)

    word_freq_df = pd.DataFrame(list(overall_word_freq.items()), columns=['Word', 'Frequency'])
    # sort df by frequency in descending order
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
    
    return word_freq_df


def regex_tokenize_df_text(df, text_column,
                           regex=None , 
                           cible=None):
    if not regex: regex = r'\b[^\W\d_]+\b'
    # Select only word without numbers and special characters may be a bad idea because of cpu model number
    tokenizer = nltk.RegexpTokenizer(regex)
    # #
    # tokenizer =  nltk.RegexpTokenizer(r'\w+')
    tokenized = df[text_column].apply(tokenizer.tokenize)

    #if inplace
    if not cible: cible = text_column

    df[cible] = tokenized


def format_text(df, text_column):
    """Format the text column in a DataFrame by converting to lowercase."""
    def format_row(row):
        format_row = [w.lower() for w in row]
        return format_row
    df[text_column] = df[text_column].apply(format_row)

    
def remove_words_from_tokens(df,token_column,
                    stop_words = stopwords.words('english')
                    ):
        
    def filter_row(token_list):
        filtered_list= []
        for w in token_list:
            if w not in stop_words:
                filtered_list.append(w)
        return filtered_list
    
    return df[token_column].apply(filter_row)


def lematize_tokens(df,token_column):
    def lematize_word_list(word_list):
        lemmatizer = WordNetLemmatizer()
        lematized_list = []
        for w in word_list:
            lematized_w = lemmatizer.lemmatize(w.lower())
            lematized_list.append(lematized_w)

        return lematized_list

    return df[token_column].apply(lematize_word_list)


def replace_re(df, column,re_tag =r'@\w+', placeholder = 'User' ):
    import re

    def replace_in_text(text):
        mentions = re.findall(re_tag, text)
        if not mentions:
            return text

        # Create a unique placeholder for each unique set of mentions
        # placeholder = 'User'
        new_text = text
        for i,mention in enumerate(mentions):
            new_text = new_text.replace(mention, placeholder#+str(i+1)
                                        )

        return new_text

    return df[column].apply(replace_in_text)
    



def tokenize_and_preprocess(df, text_column=None, 
                            cible = 'tokenised_desc',
                            reg_ex = None,
                            language = 'english',
                            min_size = 3,
                            max_freq = 400,
                            min_freq = 2,
                            actions = ['TOKENISE','FORMAT','STOPWORD'],
                            print_sample = 10,
                            return_frequ = False
                            ):
    if text_column is None:
        #if it is a pandas series, turn it into a DF
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, str):
            df = pd.DataFrame([df], columns=['text'])

        #assume there is only one column and take its name 
        text_column = list(df.columns)[0]


    current_cible = text_column
    for a in actions:
        match a:
            case 'TOKENISE':
               regex_tokenize_df_text(df,current_cible, cible = cible, regex = reg_ex)
               current_cible = cible

            case 'BAGG':
                df[cible] = df[current_cible].apply(lambda x: ' '.join(x))

            case 'FORMAT':
                format_text(df,current_cible)

            case 'REPLACE_MENTIONS':
                df[cible] = replace_re(df, current_cible,re_tag=r'@\w+')
                current_cible = cible
            
            case 'REPLACE_URLS':
                df[cible] = replace_re(df, current_cible,  r'http[s]?://\S+',
                                       placeholder='URL')
                current_cible = cible 

            case 'STOPWORD' :
                df[cible] = remove_words_from_tokens(df,current_cible,
                                stop_words = stopwords.words(language)
                                )
            case 'LEMATIZE':
                df[cible] = lematize_tokens(df,current_cible) 

            case 'WORD_FREQ':
                word_freq = get_word_frequency(df,current_cible)                            

            case 'TOO_FREQ' :
                # get word where freq is > max_freq
                too_frequ_words = word_freq.loc[word_freq['Frequency'] > max_freq, 'Word'].tolist()
                df[cible] = remove_words_from_tokens(df,current_cible,
                                stop_words = too_frequ_words )
                
            case 'TOO_RARE':
                not_freq_words = word_freq.loc[word_freq['Frequency'] < min_freq, 'Word'].tolist()
                df[cible] = remove_words_from_tokens(df,current_cible,
                                stop_words = not_freq_words )
                
            case 'TOO_SMALL':
                to_small_words = word_freq.loc[word_freq['Word'].str.len()< min_size, 'Word'].tolist()
                df[cible] = remove_words_from_tokens(df,current_cible,
                                stop_words = to_small_words )
                
            # case 'KERAS_FORMAT':
            #     ''' need to return tf.constant in a shape [["foo qux bar"], ["qux baz"]] '''
            #     return tf.constant(df[current_cible].tolist())
                 
                
        if print_sample>0:
            print(f'performing {a}')
            print(df[current_cible].head(print_sample))

    if return_frequ:
        return df[cible],word_freq
    return df[cible]