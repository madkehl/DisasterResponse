import json
import plotly
import pandas as pd
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
            lem = sno.stem(lem)
            new_txt= new_txt + " " + str(lem.lower())
        elif ('JJ' in z[1]):
            lem = sno.stem(z[0])
            new_txt= new_txt + " " + str(lem.lower())
        elif ('VB' in z[1]) and (len(z[0]) > 3):
            lem = lemma.lemmatize(z[0], 'v')
            lem = sno.stem(lem)
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
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    not_y = ['index','id', 'message', 'original', 'genre']
    Y = df.drop(not_y, axis = 1)
    category_names = list(Y.columns)
    category_counts = list(Y.sum(axis = 0).values)
    Y_long = pd.melt(Y.reset_index(), id_vars='index')
    Y_long = Y_long[Y_long['value'] == 1]
    Y_long = Y_long.groupby('index').agg('sum').reset_index()
    
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
          subplot_titles=("Genre Breakdown","Number of Categories per ID", "Category Counts"))
    
    fig.add_trace(goplot.Pie(g1), row=1, col=1)
    fig.add_trace(goplot.Histogram(x = Y_long['value']), row=1, col=2)
    fig.add_trace(goplot.Bar(g2), row=2, col=1)
    
    fig.update_layout(width = 1000, height = 1000, margin=dict(l=20, r=20, b=20, t=100),  showlegend = False)
    
    
    graphs = [
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
