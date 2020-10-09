# Disaster Response Pipeline Project

### Instructions:
Go to: https://protected-waters-31003.herokuapp.com/

or

0. **The run.py script in the Github currently has its ports set up to interface with Heroku.  If you want to run this on your own machine**
   you will have to go into the run.py file, df __main__() and comment out/disable the first two lines, and un-comment out the third (notes are in file)

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
    	`python -m nltk.downloader averaged_perceptron_tagger`**
        `python models/train_classifier.py data/Dispython -m nltk.downloader averaged_perceptron_taggerasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python install plotly --upgrade`**
    `python run.py`

3. Go to http://0.0.0.0:3001/

** apply if you are working in the Udacity workspace environment. 


# Folders and files included:

* **app**:
	* **run.py**:  This will use the .db file and model to display a web page using Flask.  
	* **templates** (from Udacity Data Science Nanodegree):
		* **master.html**: contains the basic page html template
		* **go.html**: contains html template  
* **data**:  
	* **process_data.py**: This file contains the ETL pipeline that will convert and clean the two csv files in the folder to produce the .db file in the folder
	* **disaster_categories.csv**:  contains category information on disaster messages (from Udacity Data Science Nanodegree)
	* **disaster_messages.csv**: contains message information (from UDSN) 
	* **DisasterResponse.db**: expected output of process_data.py
  
* **models**: 
	* **train_classifier.py**: This file contains the code to use GridSearchCV to train an SVM classifier on tfidf vectorized messages.
	* **classifier.pkl**: This is the model itself.

# Current Requirements:
flask==0.12.5
json5==0.8.5
pandas==0.23.3
numpy==1.12.1
nltk==3.2.5
plotly==4.11.0
scikit-learn==0.19.1
SQLalchemy==1.2.19
gunicorn==19.10.0
scipy==1.2.1

# Results:
  


# Contact: 

Madeline Kehl (mad.kehl@gmail.com)

# Acknowledgements:

* Kaggle 
* Udacity Data Science Nanodegree



# MIT License

Copyright (c) 2020 Madeline Kehl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

