import sys
import sqlite3
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import pickle

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

lemma = nltk.wordnet.WordNetLemmatizer()


def load_data(database_filepath):
    '''
    INPUT: data file path
    OUTPUT: X, Y (numerical for category), category_names
    
    STEPS:  
    1. reads in DisasterResponse db
    2. cleans text in messages and inserts in new column
    3. drops non-category columns from Y dataframe
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('data/DisasterResponse.db', engine)
    clean_messages = []
    for i in df['message']:
     #   clean_messages.append(tokenize(i))
         clean_messages.append((i))

    df['clean_messages'] = clean_messages
    
    not_y = ['index','id', 'message', 'original', 'genre', 'clean_messages']
    Y = df.drop(not_y, axis = 1)
    category_names = list(Y.columns)
    
    X = df['clean_messages']
    
    return(X, Y, category_names)
    pass

def tokenize (txt):  
    '''
    INPUT: 
    a line of text
    
    OUTPUT: 
    that same text cleaned:  
    
    STEPS:
        1. creates empty string to fill
        2. tokenizes and pos_tags
        3. lemmatizes all nouns
        4. doesn't lemmatize adj/adv bc they don't change
        5. only appends and lemmatizes longer verbs (theory that shorter verb forms are less regular and therefore more              likely to be common words/stop words)
    Chose to not use stopword list because it can be implemented from tfidf.
    POS_tagging can make this slow, but believe it is worth it for eventual output
    '''
    new_txt = ""
    tokens = word_tokenize(txt)
    pos_tagged = pos_tag(tokens)
    for z in pos_tagged:
        if (('NN' in z[1])):
            lem = lemma.lemmatize(z[0])
            new_txt= new_txt + " " + str(lem.lower())
        elif ('JJ' in z[1]) or ('RB' in z[1]):
            new_txt= new_txt + " " + str(z[0])
        elif ('VB' in z[1]) and len(z[0]) > 3:
            lem = lemma.lemmatize(z[0], 'v')
            new_txt= new_txt + " " + str(lem.lower())
 #   print(new_txt)
    return(new_txt)
    pass


def build_model():
    '''
    INPUT: none
    
    OUTPUT: GridSearchCV obj
    
    In the interest of time, only min samples split and max features are varied, for Random Forests
    '''
    parameters = {
        #originally included 'True', but this is not possible because raw tfidf
        #is sparse.  Might be worth trying adding some kind of decomposition like
        #lda or nmf to address this
        'clf__estimator__min_samples_split': [2,15, 50],
        'clf__estimator__max_features': [5,10,50]
        }
    
    pipeline = Pipeline([
        ('tfidf_vec', TfidfVectorizer()),
        ('scaler', StandardScaler(with_mean = False)),
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(),n_jobs=-1))
        ])
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 3)
    
    return(cv)
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    cv model, X_test, Y_test, list of category names
    
    OUTPUT:
    precision, f1 score, recall for each category
    '''
    y_pred = model.predict(Y_test)
    prfs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=category_names)
    prfs_df = pd.DataFrame(prfs_df)
    print(prfs_df.head())
    print("\nBest Parameters:", cv.best_params_)

    pass


def save_model(model, model_filepath):
    '''
    INPUT: model and filepath
    
    OUTPUT: None
    Saves model at given file path
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    pass


'''def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()'''
