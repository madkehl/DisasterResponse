import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from plotly.subplots import make_subplots
import plotly.graph_objects as goplot

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

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
                "values": genre_counts,
           }

        
    
    g2 =  {     'type': 'bar',
                 'x': category_names,
                 'y': category_counts,
                 "marker": {
                    "color": category_counts,
                    "colorscale":"Viridis"
                    }
              
        }
    
    
    # create visuals
    fig = make_subplots(rows=1, cols=2,
          specs=[[{"type": "pie"}, {"type": "bar"}]])
    fig.add_trace(goplot.Pie(g1), row=1, col=1)
    fig.add_trace(goplot.Bar(g2), row=1, col=2)
    
    
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
