import json

CLASS_LABELS = "../../data/class_labels.json"

def get_class_labels(class_labels_path: str = CLASS_LABELS) -> dict:

    labels = {}

    with open(CLASS_LABELS, 'r') as f:
        labels = json.load(class_labels_path)

    return labels
