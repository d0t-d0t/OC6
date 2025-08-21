import optuna
from mlflow import MlflowClient
from Mlflow import mlflow_logged_model_training, start_local_experiment

def objective(trial,df,
              target_metrics = 'val_accuracy',
              **kwargs):
              
    params = { 
        # 'param1' : trial.suggest_float('param1', 0.0, 1.0),
        # 'param2' : trial.suggest_int('param2', 1, 10)
        }

    for k,v in kwargs:
        match v[0]:
            case float():
                func = trial.suggest_float
            case int():
                func = trial.suggest_int
        params[k]= func(k,v[0],v[1])

    # Call the original function with the hyperparameters suggested by Optuna
    mlflow_logged_model_training(df, **kwargs)

    # Get run metrics
    client = MlflowClient()
    run_info = client.get_latest_runs()[0]
    metrics = run_info.data.metrics

    #Metric cible
    score = metrics[target_metrics]

    return score  # Minimize or maximize this value depending on your goal

def main():
    study = optuna.create_study(direction='maximize')
    start_local_experiment(experiment_name="Optuna study")
    study.optimize(objective, n_trials=100)

if __name__ == "__main__":
    main()