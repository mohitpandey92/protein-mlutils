import mlflow
import os




def log_experiment_params(params: dict):
    """
    Log experiment parameters to MLflow.

    Args:
        params (dict): Dictionary of parameters to log.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)

def start_mlflow_tracking_for_pytorch_lightning(location_db: str, experiment_name: str, 
                                                run_name: str, params_dict: dict, trainer, model, train_loader, val_loader, test_loader=None):
    """
    Start MLflow tracking server with a SQLite backend.

    Args:
        location_db (str): The file path for the SQLite database.
        experiment_name (str): The name of the MLflow experiment.
        run_name (str): The name of the MLflow run.
        params_dict (dict): Dictionary of parameters to log.
        trainer: PyTorch Lightning Trainer object.
        model: PyTorch Lightning model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    
    TODO:
        register ML models to MLflow model registry
    Returns:
        None
    """

    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + location_db
    
    
    mlflow.set_experiment(experiment_name)
    if trainer.global_rank == 0:
        print(f"MLflow tracking server started at {location_db}")
        mlflow.pytorch.autolog(log_models=False, log_every_n_epoch=1)
    
        with mlflow.start_run(run_name=run_name):
            
        
            log_experiment_params(params_dict)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            trainer.test(model, dataloaders=test_loader)
        
    mlflow.end_run()

