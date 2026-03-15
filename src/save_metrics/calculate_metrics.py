from sklearn.metrics import f1_score,  precision_score, recall_score, matthews_corrcoef

def evaluate(predictions: list, gold_labels: list) -> dict:
    
    pred_metrics = {
        "predictions": predictions,
        "Macro-F1": f1_score(gold_labels, predictions, average='macro'),
        "Precision": precision_score(gold_labels, predictions, average='macro'),
        "Recall": recall_score(gold_labels, predictions, average='macro'),
        "MCC": matthews_corrcoef(gold_labels, predictions)
    }

    return pred_metrics