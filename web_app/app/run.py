import json
import plotly
import pandas as pd
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
sno = nltk.stem.SnowballStemmer('english')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from plotly.subplots import make_subplots
import plotly.graph_objects as goplot

app = Flask(__name__)


def get_col_sample(df, samplen):
    return(df.sample(n=samplen, replace=True, random_state=1).reset_index(drop = True))

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
    
    '''
    new_txt = ""
    tokens = word_tokenize(txt)
    pos_tagged = pos_tag(tokens)
    for z in pos_tagged:
        if (('NN' in z[1])):
            lem = lemma.lemmatize(z[0])
            new_txt= new_txt + " " + str(lem.lower())
        elif ('JJ' in z[1]):
            new_txt= new_txt + " " + str(z[0].lower())
        elif ('VB' in z[1]) and (len(z[0]) > 3):
            lem = lemma.lemmatize(z[0], 'v')
            new_txt= new_txt + " " + str(lem.lower())
    return(new_txt)
    pass
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    not_y = ['index','id', 'message', 'original', 'genre']
    Y = df.drop(not_y, axis = 1)
    category_names = list(Y.columns)
    category_counts = list(Y.sum(axis = 0).values)
    
    agg = df.drop(not_y, axis = 1)
    agg_sum = agg.sum(axis = 1).values
    
    number_samples = int(np.median(category_counts))
    
    g1 = {      "type": "pie",
                "domain": {
                    "x": [0,1],
                    "y": [0,1]
                    },
                "marker": {
                    "colors": genre_counts
                
                    },
                "hoverinfo": "label+value",
                "labels": genre_names,
                "values": genre_counts
           }

        
    
    g2 =  {     'type': 'bar',
                 'x':category_names,
                 'y': category_counts,
                "hoverinfo": "x+y",
                 "marker": {
                    "color": category_counts,
                    "colorscale":"Viridis"
                    }
           
              
        }
  
    
    # create visuals
    fig = make_subplots(rows=2, cols=2,
          specs=[[{"type": "pie"}, {"type": "histogram"}] ,[{"type": "bar", "colspan": 2},None]],
          subplot_titles=("Genre Breakdown (Raw)","Number of Categories per ID (Raw)", "Category Counts (Raw)"))
    
    fig.add_trace(goplot.Pie(g1), row=1, col=1)
    fig.add_trace(goplot.Histogram(x = agg_sum), row=1, col=2)
    fig.add_trace(goplot.Bar(g2), row=2, col=1)
    
    fig.update_layout(width = 1000, height = 1000, margin=dict(l=20, r=20, b=20, t=100),  showlegend = False)
    
    balanced_list = []
    for i in category_names[1:len(category_names)-1]:
        for_balance = df[df[i] == 1].reset_index(drop = True)
        balanced_list.append(get_col_sample(for_balance, number_samples)) 
        
    balanced_df = pd.concat(balanced_list, axis = 0)
    
    agg_bal = balanced_df.drop(not_y, axis = 1)
    agg_bal_sum = agg_bal.sum(axis = 1).values
    
    Y_bal = balanced_df.drop(not_y, axis = 1)
    X_bal = balanced_df['message']
    
    category_names_bal = list(Y_bal.columns)
    category_counts_bal = list(balanced_df.groupby('id').agg('sum', axis = 0).values) 
    
    Y_long_bal = pd.melt(Y_bal.reset_index(), id_vars='index')
    Y_long_bal = Y_long_bal[Y_long_bal['value'] == 1]
    Y_long_bal = Y_long_bal.groupby('index').agg('sum').reset_index()
    
    g3 =  {     'type': 'bar',
                 'x':category_names_bal,
                 'y': category_counts_bal,
                "hoverinfo": "x+y",
                 "marker": {
                    "color": category_counts_bal,
                    "colorscale":"blackbody"
                    }
           
              
        }
    
    
    fig1 = make_subplots(rows=1, cols=2,
          specs=[[{"type": "histogram"},{"type": "bar"}]],
          subplot_titles=( "Category Counts (Balanced)","Number of Categories per ID (Balanced)"))
    
    fig1.add_trace(goplot.Histogram(x = agg_bal_sum), row=1, col=2)
    fig1.add_trace(goplot.Bar(g3), row=1, col=1)
    
    fig1.update_layout(width = 1000, height = 500, margin=dict(l=20, r=20, b=20, t=100),  showlegend = False)
    
    graphs = [
        fig1,
        fig
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    port = int(os.environ.get('PORT', 5000))#comment this out
    app.run(host='0.0.0.0', port=port)#comment this out
  #  app.run(host='0.0.0.0', port=3001, debug=True) #uncomment this


if __name__ == '__main__':
    main()
