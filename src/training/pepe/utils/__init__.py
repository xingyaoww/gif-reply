import pandas as pd
import ast


def load_dataset(from_path):
    dataset = pd.read_csv(from_path)
    dataset["tags"] = dataset["tags"].apply(ast.literal_eval)

    dataset_info = {}
    dataset_info['id_to_label'] = list(
            set(dataset['tags'].to_list()))
    dataset_info['label_to_id'] = dict(
            zip(dataset_info['id_to_label'], range(0, len(dataset_info['id_to_label']))))
    dataset_info['n_labels'] = len(dataset_info['id_to_label'])
    return dataset, dataset_info
