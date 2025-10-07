import mlflow
import os
from pytorch_lightning.loggers import MLFlowLogger


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
    
    # 1. Set the tracking URI globally
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + location_db
    
    # 2. Instantiate the MLFlowLogger
    # NOTE: The logger will set the experiment and start the run itself when trainer.fit is called.
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        # log_model=True or 'all' to log checkpoints
    )
    mlflow.pytorch.autolog(log_models=False, disable=False)
    
    # 4. Use the logger with the Trainer
    trainer.logger = mlf_logger 
    
    # For parameters that you want to log *outside* of autologging and before training starts, 
    # you must use the rank_zero_only utility or check the rank.
    if trainer.global_rank == 0:
        mlflow.set_experiment(experiment_name) 
        with mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
                mlflow.log_params(params_dict)

    # The Trainer.fit and Trainer.test calls will use the logger and autologging 
    # will only log on global rank 0.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model, dataloaders=test_loader)
    