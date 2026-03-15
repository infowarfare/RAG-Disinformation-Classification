import datetime
import uuid
import json

def save_metrics_to_file(clf_name, model_name, pred_metrics, folder_path: str) -> None:
        """
    Saves classification performance metrics to a structured file.

    This function takes model metadata and evaluation results, formats them, 
    and persists them to a designated directory for later analysis or 
    benchmarking.

    Args:
        clf_name (str): The name of the classifier or algorithm used.
        model_name (str): The specific version or name of the model.
        pred_metrics (dict): A dictionary containing metric names as keys and 
            their calculated values (e.g., {'accuracy': 0.95, 'macro-f1': 0.94}).
        folder_path (str): The directory path where the output file should be saved.

    Returns:
        None: The function writes to the filesystem and does not return a value.

    Raises:
        Exeception: If the directory cannot be created or the file cannot be written.
    """
        # get pred_metrics
        try:
            if pred_metrics is None:
                raise ValueError("Predictions metrics not found.")
            
            

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8] # Short UUID for extra safety

            filename = f"{folder_path}\\metrics_{clf_name}_{unique_id}"

            meta_data = {
                "meta_data": {
                    "classifier_name": clf_name,
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "run_id": unique_id
                }
            }

            complete_metrics = meta_data | pred_metrics

            with open(f"{filename}.json", "w") as f:
                json.dump(complete_metrics, f, indent=4)   

        except Exception as e:
            raise Exception(f"An unexpected Error occurred while saving metrics: {str(e)}")