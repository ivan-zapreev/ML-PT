# ML-PT
The test task for ML-PT With the task desacription available via the [link](https://heyiamsasha.notion.site/ML-PT-0bc4ce5012604ed397f040a1bdc29858).

The main content can be reached by browsing through the following notebooks:
* ``01_data_exploration.ipynb``
    - Initial data exploration
    - Data wrangling
    - Preliminary analysis of the data columns
* ``02_data_analysis.ipynb`` **(!)**
    - In depth investigation of the data columns value
    - PCA analysis and feature space visualization
    - Extracting and dumping the features
    - *Storing the features extractor* for use in online events classification
* ``03_tune_dbscan.ipynb`` **(!)**
    - Using DBSCAN for feature clustering
    - Tuning DBSCAN for optimal clustering
    - Identifying GOOD, ATTACK and Unclassified (Noize) clusters
* ``04_select_classifier.ipynb``
    - Choosing a classifier for online event classification
    - Training and comparing default performance of 13 classifiers
    - Motivating selection of Random Forest Classifier 
* ``05_tune_classifier.ipynb`` **(!)**
    - Hyperparameters tuning of the Random Forest Classifier
    - Training the Random Forest Classifier with best hyperparameters
    - *Storing the classifier* for use in online events classification
* ``06_run_test_client.ipynb`` **(!)**
    - Running the client side to evaluate the online classifier ran separately as a Flask server
    - Loads the initially provided raw events data and queries the online service for classification
    - Compares the resulting event classes with the ones generated with Random Forest Classifier
        - This is done for the sanity check to make sure data wrangling and featurization work well

**NOTE 01:** The notebooks ``01`` to ``05`` above can be run sequentially, if needed to re-produce the results or re-generate the classifier and feature extractor models.

**NOTE 02:** To test the service one does not have to run notebooks ``01`` to ``05`` but shall:
1. Run the event classification restul service (see sections below)
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
        * **service/** - Flask service related file for the server
        * **utils/** - various utility source files
   * *READRE.md* - this read-me file
   * *LICENSE* - licensing information file
   * *Dockerfile* - docker image building file
   * *requirements.txt* - the python package list needed for running
   * *\*.ipynb* - multiple notebooks containing the main research work and testing

# How to run the Service
Requires *Python 3.8*

## Using local server

1. Install packages
    * ``ML-PT % pip install -r requirements.txt``
2. Run server
    * ``ML-PT % python ./src/service/flask_app.py``
3. Run test notebook
    * 06_run_test_client.ipynb

## Using Docker container

1. Install Docker
     ``https://docs.docker.com/get-docker/``
3. Run docker image:
    * ``ML-PT % docker run -i -t python-ml-pt``
4. Run test notebook
    * 06_run_test_client.ipynb

## Re-generate requirements
``ML-PT % pipreqs . --force``

## Re-generate docker image
``ML-PT % docker build -t python-ml-pt .``

# TODOs
1. Event though we get at most 2.8% of noize when running DBSCAN, one can try to reduce the noize levels by reducing *min_samples*
