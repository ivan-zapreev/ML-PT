# ML-PT
The test task for ML-PT With the task desacription available via the [link](https://heyiamsasha.notion.site/ML-PT-0bc4ce5012604ed397f040a1bdc29858).

The main content can be reached by browsing through the following notebooks:
* ``01_data_exploration.ipynb``
    - Initial data exploration
    - ***Data wrangling***
    - Preliminary analysis of the data columns
* ``02_data_analysis.ipynb`` **(!)**
    - In depth investigation of the data column values
    - *PCA analysis and feature space visualization*
    - Extracting and dumping the features
    - ***Storing the events feature extractor*** for use in online events classification
* ``03_tune_dbscan.ipynb`` **(!)**
    - Using DBSCAN for optimal feature clustering
    - DBSCAN hyperparameters tuning, *finding the optimal number of clusters*
    - *Feature space visualization with clusters*
    - ***Identifying GOOD, ATTACK and Unclassified (Noize) clusters***
* ``04_select_classifier.ipynb``
    - Choosing a classifier for online event classification
    - Training and comparing default performance of 13 classifiers
        - *Using k-fold cross-validation*
    - ***Motivating selection of Random Forest Classifier***
* ``05_tune_classifier.ipynb`` **(!)**
    - Hyperparameters tuning of the Random Forest Classifier
        - *Using Random and Grid search with k-fold cross-validation*
    - Training the Random Forest Classifier with best hyperparameters
    - ***Storing the events classifier*** to be used in online events classification
* ``06_run_test_client.ipynb`` **(!)**
    - ***Runs the client side to evaluate the online classifier ran as a Restful (Flask) Application***
    - Loads the initially provided raw events data and queries the online service for classification
    - Compares the resulting event classes with the ones generated with Random Forest Classifier
        - This is done for the sanity check to make sure data wrangling and featurization work well

**NOTE 01:** The notebooks ``01`` to ``05`` above can be run sequentially, if needed to re-produce the results or re-generate the classifier and feature extractor models.

**NOTE 02:** To test the service one does not have to run notebooks ``01`` to ``05`` but shall:
1. Run the event classification restful service (see sections below)
2. Execute the ``06_run_test_client.ipynb`` notebook or use his/her own client to send requests to:
    * ``http://127.0.0.1:8080/predict``

# Repository structure

* **ML-PT/** - the root folder
    * **docs/** - stores the paper used for inspiration and the task description
    * **data/** - the provided raw data csv along with files generated from notebooks
    * **src/** - the main source code folde
        * **wrangler/** - data wrangling related files
        * **features/** - feature extraction related file
        * **model/** - packages related to clustering and classification model training/tuning
            * **dbscan/** - DBSCAN tuning related sources
            * **classifier/** - files related to selecting and training/tuning the classifier
        * **service/** - Flask-based event classification service
        * **utils/** - various utility source files
   * *READRE.md* - this read-me file
   * *LICENSE* - licensing information file
   * *Dockerfile* - docker image building file
   * *requirements.txt* - the python package list needed for running
   * *\*.ipynb* - multiple notebooks containing the main research work and testing

# How to run the Event Classification Service
Requires *Python 3.8*

## Using local server

1. Install packages
    * ``ML-PT % pip install -r requirements.txt``
2. Run server
    * ``ML-PT % python ./src/service/flask_app.py``
3. Run test notebook
    * ``ML-PT % 06_run_test_client.ipynb``

## Using Docker container

1. Install Docker
    * ``https://docs.docker.com/get-docker/``
2. Pull docker image:
    * ``docker image pull zapreevis/python-ml-pt:latest``
4. Run docker container:
    * ``docker run -p 127.0.0.1:8080:8080 -i -t zapreevis/python-ml-pt:latest``
5. Run test notebook
    * ``ML-PT % 06_run_test_client.ipynb``

# Restful API examples

# Request
Hereby an example request sent via ``curl``:

```
 % curl -X 'POST' \
'http://127.0.0.1:8080/predict' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '[{"data": "{\"CLIENT_IP\": \"188.138.92.55\", \"CLIENT_USERAGENT\": NaN, \"REQUEST_SIZE\": 166, \"RESPONSE_CODE\": 404, \"MATCHED_VARIABLE_SRC\": \"REQUEST_URI\", \"MATCHED_VARIABLE_NAME\": NaN, \"MATCHED_VARIABLE_VALUE\": \"//tmp/20160925122692indo.php.vob\", \"EVENT_ID\": \"AVdhXFgVq1Ppo9zF5Fxu\"}"}, \
     {"data": "{\"CLIENT_IP\": \"93.158.215.131\", \"CLIENT_USERAGENT\": \"Mozilla/5.0 (Windows NT 6.3; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0\", \"REQUEST_SIZE\": 431, \"RESPONSE_CODE\": 302, \"MATCHED_VARIABLE_SRC\": \"REQUEST_GET_ARGS\", \"MATCHED_VARIABLE_NAME\": \"url\", \"MATCHED_VARIABLE_VALUE\": \"http://www.galitsios.gr/?option=com_k2%5C%5C\", \"EVENT_ID\": \"AVdcJmIIq1Ppo9zF2YIp\"}"}]'
```

# Response
The example service response structure is:

```
[
{
"EVENT_ID": "AVdhXFgVq1Ppo9zF5Fxu",
"LABEL_PRED": 42
},
{
"EVENT_ID": "AVdcJmIIq1Ppo9zF2YIp",
"LABEL_PRED": 3
}
]
```

# Utility information

## Re-generate requirements file
``ML-PT % pipreqs . --force``

## Re-generate docker image
1. ``ML-PT % docker build -t zapreevis/python-ml-pt:latest .``
2. ``ML-PT % docker push zapreevis/python-ml-pt:latest``

# TODOs
1. Event though we get at most 2.8% of noize when running DBSCAN, one can try to reduce the noize levels by reducing *min_samples*
