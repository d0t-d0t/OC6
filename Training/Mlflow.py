import mlflow
import subprocess
import pandas as pd
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score,confusion_matrix
import time

from TweetClassifier import TweetClassifierPipeline
from Visualisation_tools import plot_pred_per_type

def start_local_experiment( host='127.0.0.1',
                            port='8080',
                            uri=r'/mlruns',
                            experiment_name="Analyse sentiments Tweet"
                            ):
    command = f'''mlflow server --host {host}  --port {port} \n
                mlflow ui --backend-store-uri {uri}'''
    print(command)

    result = subprocess.Popen(command, shell=True)

    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

    mlflow.set_experiment(experiment_name)



def mlflow_logged_model_training(df,
                                 feature = 'tweet',
                                 cible = 'target',
                                 preproc_actions = [],
                                 vector_lib = '',
                                 vector_name = '',
                                 validation_split = 0.2,
                                 test_split = None,
                                 vectorizer_func=CountVectorizer,
                                 vectorizer_params = {},
                                 metric_funcs = [accuracy_score, 
                                                 precision_score, 
                                                 f1_score,
                                                 recall_score],
                                 model_lib='SKLEARN',
                                 model_func=LogisticRegression,
                                #  class_name = '',
                                 model_name = 'TweetClassifierPipeline',
                                 training_description = 'some_info_about this_training',
                                 model_params = {
                                                    "solver": "lbfgs",
                                                    "max_iter": 1000,
                                                    "multi_class": "auto",
                                                    "random_state": 8888,
                                                },
                                 observ_probabillity = False,
                                 debug = False,
                                 sign_model = False

                                   ):
    
    start_time = time.time()

    class_name = model_func.__name__
    if vectorizer_func:
        vector_name =vectorizer_func.__name__
        
    


    X = df[feature]
    y = df[cible]

    # Split the data into training and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=validation_split, 
        random_state=42
    )
    if test_split:
        X_val, X_test, y_val, y_test = train_test_split(
            X_val, y_val, 
            test_size=test_split, 
            random_state=42)


    

    # Train the model
    model = TweetClassifierPipeline(preproc_actions=preproc_actions,
                                      vectorizer_func=vectorizer_func,
                                      vectorizer_params=vectorizer_params,
                                      model_lib=model_lib,
                                      model_func=model_func,
                                      model_params=model_params,
                                      vector_lib=vector_lib)
    model.fit(X_train, y_train,x_test=X_val,y_test=y_val)
    temp_X = X_val.copy()
    if isinstance(temp_X, pd.Series):
        temp_X = temp_X.to_frame()
    # Predict on the test set
    temp_X['prediction'] = model.predict(temp_X[feature])

    plot_dic = {}

    if observ_probabillity:        
        # temp_X = X_val.copy()
        # model.predict_proba(temp_X[feature])
        proba_predict = model.predict_proba(temp_X[feature])
        if proba_predict.shape[1] == 2:
            temp_X['negative_probability'] = proba_predict[:, 0]
        else:
            temp_X['negative_probability'] = proba_predict
        temp_X['target'] =  y_val
        # temp_X['prediction'] = y_pred
        temp_X['prediction_type']= 'TP'
        temp_X.loc[(temp_X['prediction']==1) & (temp_X['target']==0),'prediction_type']= 'FP'
        temp_X.loc[(temp_X['prediction']==0) & (temp_X['target']==1),'prediction_type']= 'FN'
        temp_X.loc[(temp_X['prediction']==0) & (temp_X['target']==0),'prediction_type']= 'TN'
        fig = plot_pred_per_type(temp_X,
                                 cible='prediction_type',
                                 bin_cible='negative_probability')
        plot_dic['probability_plot'] = fig

        
    # Calculate metrics
    metric_dic =  {}
    # test_slices = []
    # for t in test_slices:
    for mf in metric_funcs:
        mf_name = mf.__qualname__
        mfr = mf(y_val, temp_X['prediction'])
        metric_dic[mf_name]= mfr



    process_time = time.time() - start_time


    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        if vectorizer_func:
            mlflow.log_params(vectorizer_params)
        # mlflow.pyfunc.load_model(vector_info.model_uri)
        # Log the hyperparameters
        mlflow.log_params(model_params)

        # Log the loss metric
        for metric, value in metric_dic.items():
            mlflow.log_metric(metric, value)


        # log the plots
        for plot_name,plot in plot_dic.items():
            mlflow.log_figure(plot, f"{plot_name}.png")

        signature = None
        # Infer the model signature
        if sign_model:
            signature = infer_signature(X_train, model.predict(X_train), model_params )

        # if  class_name == '':


        # Log the model, which inherits the parameters and metric

        model_info  = model_info = mlflow.sklearn.log_model(
                                            sk_model=model,        
        # mlflow.pyfunc.log_model(
        #                                   python_model=model,
                                            name=model_name,
                                            signature=signature,
                                            input_example=X_train.iloc[0:5].to_frame(),
                                            registered_model_name=f"{model_name}_{vector_name}_{class_name}",
                                            )

        # Log other information about the model
        mlflow.log_params({ 'Vectorization Library': vector_lib,
                            'Vectorisation Process': vector_name,                            
                            'Input_column':feature,
                            "Process Time": process_time,
                            "Preprocess Actions": preproc_actions,
                            "Model Library": model_lib,})

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": training_description,
                                  
                                  }
        )

        mlflow.pyfunc.load_model(model_info.model_uri)

        # Return the run ID for future reference
        # return run_id