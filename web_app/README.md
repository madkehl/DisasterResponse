# Disaster Response Pipeline Project

### Instructions:
[app hosted in Heroku](https://protected-waters-31003.herokuapp.com/)

or

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
    	`python -m nltk.downloader averaged_perceptron_tagger`**
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app. **NOTE SECOND ARG**: this is because the default is set to run smoothly with heroku, port 5000
    `python install plotly --upgrade`**
    `python run.py 3001`

3. Go to http://0.0.0.0:3001/

** these steps apply if you are working in the Udacity workspace environment. 


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
  
see accuracy_metrics.txt

# Discussion/Notes:

**The child_alone category, as it was found to be empty (no training data available for it) was dropped from the 36 possible labels**

The data was not evenly split between categories, leading to poor original performance by the classifier.  This was somewhat remedied by finding the median number of samples in a category and taking a bootstrapped sample of each category equal to this median (sampling with replacement). This was superior to simply subsampling to the size of the smallest category because it required us to throw out less data (the min # per category was 117, not including 0 for child_alone). Sampling with replacement allowed us to oversample the smaller categories. 

Because of the multioutput nature of the data, there is still an overrepresentation of things marked as 'related'. This is because in our random subsamples, most items "drawn" for the other categories were also marked as related. Therefore this classifier is very likely to mark everything as at least 'related' and is not very useful for determining whether something is related or not. Additionally, the smaller the root category, the more likely it is to be overfit too the training data.  
 
# Contact: 

Madeline Kehl (mad.kehl@gmail.com)

# Acknowledgements:

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

