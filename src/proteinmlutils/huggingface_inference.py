#import os
#import mlflow
import torch
import numpy as np
import time
import json
import multiprocessing
from evaluate import load
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

t1=time.time()


metric = load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")




class Custom_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ***
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        ***
       
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
    
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            #loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
           
            class_weights = torch.tensor([0.2, 0.8]).to(outputs[0].device)
            logits=outputs["logits"]
            loss = torch.nn.functional.cross_entropy(logits, inputs["labels"], weight=class_weights)
            
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    """
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        #if self.is_fsdp_xla_v2_enabled:
        #    eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        #start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        return output



def model_loader_fn(model_path):
    """
    Load the model for inference
    """
    print("transformers version", transformers.__version__, "PyTorch version", torch.__version__)   

    #model_path = os.path.join(model_dir, 'model/')
    
    # Load PyTorch HF tokenizer from disk.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    # Load PyTorch HF model from disk.
    model = AutoModelForSequenceClassification.from_pretrained(model_path)


    model_dict = {'model': model, 'tokenizer':tokenizer}
    
    return model_dict



def tokenizer_fn(input_data, model):
    tokenizer = model['tokenizer']
    model = model['model']
    
    encoded_input = tokenizer(input_data, return_tensors='pt', padding=True)
    eval_dataset = Dataset.from_dict(encoded_input)
    return eval_dataset

def predict_classifier_fn(eval_dataset, model_dict):
    """
    It provides the HF classifier predictions in a batch size of 24
    """
    
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    
    #encoded_input = tokenizer(input_data, return_tensors='pt', padding=True)
    
    #eval_dataset = Dataset.from_dict(encoded_input)
    batch_size = 24
    output_dir="."

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        push_to_hub=False,report_to=None, logging_steps=250, dataloader_num_workers=multiprocessing.cpu_count(), logging_strategy="steps", logging_dir=None, use_cpu=False,  fp16=True, save_only_model=True)
    
    
    custom_trainer_instance=Custom_Trainer(model,
        args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    output=custom_trainer_instance.evaluate()
    
    probs=torch.nn.functional.softmax(torch.tensor(output.predictions), dim=1).numpy()
    y_pred=np.argmax(probs, axis=1)
    
    return y_pred, probs



def predict_regressor_fn(eval_dataset, model_dict):
    
    """
    It provides the HF regression predictions in a batch size of 24
    """
    
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    
    #encoded_input = tokenizer(input_data, return_tensors='pt', padding=True)
    
    #eval_dataset = Dataset.from_dict(encoded_input)
    batch_size = 24
    output_dir="."
    #eval_dataset = Dataset.from_dict(encoded_input)
    
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        push_to_hub=False,report_to=None, logging_steps=250, dataloader_num_workers=multiprocessing.cpu_count(), logging_strategy="steps", logging_dir=None, use_cpu=False,  fp16=True, save_only_model=True)
    
    
    custom_trainer_instance=Custom_Trainer(model,
        args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    output=custom_trainer_instance.evaluate()
    
    
    #probs=torch.nn.functional.softmax(torch.tensor(output.predictions), dim=1).numpy()
    #y_pred=np.argmax(probs, axis=1)
    
    return output.predictions.squeeze(-1)

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response