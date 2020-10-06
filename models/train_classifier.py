import sys
import sqlite3
from sqlalchemy import create_engine

import pandas as pd
import numpy as np

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

lemma = nltk.wordnet.WordNetLemmatizer()


def load_data(database_filepath):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('data/DisasterResponse.db', engine)
    clean_messages = []
    for i in df['message']:
     #   clean_messages.append(tokenize(i))
         clean_messages.append((i))

    df['clean_messages'] = clean_messages
    not_y = ['index', 'message', 'original', 'genre']
    Y = df.drop(not_y, axis = 1)
    category_names = list(Y.columns)
    Y = pd.melt(Y, id_vars = 'id')
    Y = Y[Y['value'] == 1]
    final_X = df.merge(Y, right_on = 'id', left_on = 'id', how = 'inner')
    X = final_X['clean_messages']  
    final_X["variable"] = final_X["variable"].astype('category')
    Y_coded = final_X["variable"].cat.codes
    return(X, Y_coded, category_names)
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
        5. only appends and lemmatizes longer verbs (theory that shorter verb forms are less regular and therefore more likely to be common words/stop words)
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
    parameters = {
        'scaler__with_mean': [True, False],
        'clf__kernel': ['linear', 'rbf', 'sigmoid'],
        'clf__C':[0.001, 0.1, 1, 10, 100],
        'clf__gamma': np.arange(0.1, 0.9, 0.05)
        }
    
    pipeline = Pipeline([
        ('tfidf_vec', TfidfVectorizer()),
        ('scaler', StandardScaler()),
        ('clf', svm.SVC()),
        ])
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return(cv)
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(Y_test)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=category_names)
    accuracy = (y_pred == y_test).mean()
    
    print("Labels:", category_names)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)

    pass


def save_model(model, model_filepath):
    pass


def main():
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
    main()
