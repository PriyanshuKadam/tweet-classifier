import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv('../data/train.csv')

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text

def find_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'

def find_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'

def find_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'

def process_text(df):
    
    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))
    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))
    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))
    df['links'] = df['text'].apply(lambda x: find_links(x))
    
    return df
def transform_text(train, input_text):
    test = pd.DataFrame({'text': [input_text]})
    train= process_text(train)
    test = process_text(test)

    from sklearn.feature_extraction.text import CountVectorizer

    # Links
    vec_links = CountVectorizer(min_df=5, analyzer='word', token_pattern=r'https?://\S+')  # Only include those >=5 occurrences
    link_vec = vec_links.fit_transform(train['links'])
    link_vec_test = vec_links.transform(test['links'])
    X_train_link = pd.DataFrame(link_vec.toarray(), columns=vec_links.get_feature_names_out())
    X_test_link = pd.DataFrame(link_vec_test.toarray(), columns=vec_links.get_feature_names_out())

    # Mentions
    vec_men = CountVectorizer(min_df=5)
    men_vec = vec_men.fit_transform(train['mentions'])
    men_vec_test = vec_men.transform(test['mentions'])
    X_train_men = pd.DataFrame(men_vec.toarray(), columns=vec_men.get_feature_names_out())
    X_test_men = pd.DataFrame(men_vec_test.toarray(), columns=vec_men.get_feature_names_out())

    # Hashtags
    vec_hash = CountVectorizer(min_df=5)
    hash_vec = vec_hash.fit_transform(train['hashtags'])
    hash_vec_test = vec_hash.transform(test['hashtags'])
    X_train_hash = pd.DataFrame(hash_vec.toarray(), columns=vec_hash.get_feature_names_out())
    X_test_hash = pd.DataFrame(hash_vec_test.toarray(), columns=vec_hash.get_feature_names_out())

    from sklearn.feature_extraction.text import TfidfVectorizer

    vec_text = TfidfVectorizer(min_df=10, ngram_range=(1, 2), stop_words='english') 
    # Only include >=10 occurrences, use unigrams and bigrams
    text_vec = vec_text.fit_transform(train['text_clean'])
    text_vec_test = vec_text.transform(test['text_clean'])

    X_train_text = pd.DataFrame(text_vec.toarray(), columns=vec_text.get_feature_names_out())
    X_test_text = pd.DataFrame(text_vec_test.toarray(), columns=vec_text.get_feature_names_out())

    train = train.join(X_train_link, rsuffix='_link')
    train = train.join(X_train_men, rsuffix='_mention')
    train = train.join(X_train_hash, rsuffix='_hashtag')
    train = train.join(X_train_text, rsuffix='_text')
    test = test.join(X_test_link, rsuffix='_link')
    test = test.join(X_test_men, rsuffix='_mention')
    test = test.join(X_test_hash, rsuffix='_hashtag')
    test = test.join(X_test_text, rsuffix='_text')
    
    from sklearn.preprocessing import MinMaxScaler
    X = train.drop(columns = ['id', 'keyword','location','text','text_clean', 'hashtags', 'mentions','links', 'target'])
    X_test = test.drop(columns = ['text','text_clean', 'hashtags', 'mentions','links'])
    y = train.target
    return X, y, X_test    