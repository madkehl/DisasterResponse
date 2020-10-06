import sys
import sqlite3
import pandas as pd


from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk

lemma = nltk.wordnet.WordNetLemmatizer()

def lem_prettify (txt):  
    '''
    INPUT: 
    a line of text
    
    OUTPUT: 
    that same text cleaned:  
    
    STEPS:
        1. creates empty string to fill
        2. tokenizes and pos_tags
        3. lemmatizes all nouns, adjectives, adverbs
        4. only lemmatizes longer verbs (theory that shorter verb forms are less regular and therefore more likely to be common words/stop words)
    Chose to not use stopword list because the context of disaster oriented tweets is very different from general language usage.
    '''
    new_txt = ""
    tokens = word_tokenize(txt)
    pos_tagged = pos_tag(tokens)
    for z in pos_tagged:
        if (('NN' in z[1]) or ('JJ' in z[1]) or ('RB' in z[1])):
            lem = lemma.lemmatize(z[0])
            new_txt= new_txt + " " + str(lem.lower())
        elif len(z[0]) > 3:
            lem = lemma.lemmatize(z[0], 'v')
            new_txt= new_txt + " " + str(lem.lower())
 #   print(new_txt)
    return(new_txt)
    pass

def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    clean_messages = []
    for i in df['message']:
        clean_messages.append(lem_prettify(i))
    df['clean_messages'] = clean_messages
    return(data)
    pass


def tokenize(text):
    pass


def build_model():
    pipeline
    pass


def evaluate_model(model, X_test, Y_test, category_names):
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