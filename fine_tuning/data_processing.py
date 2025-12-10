from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pandas
from datasets import Dataset, DatasetDict
import torch

def load_split_dataset(dataset_path):
    """
    Loads up the CSV file and splits the data into train, validation, and test sets. 
    70% of the data goes to training, 20% goes to validation, and 10% goes to testing.

    Returning a DatasetDict which has the three different splits.
    """

    df = pandas.read_csv(dataset_path)

    print("--------Dataset Info--------")
    print(f"Sample Size: {len(df)}")
    print(f"Column names: {df.columns.to_list()}")

    count_per_level = df["Bloom Level"].value_counts().to_dict()
    print(f"Count for each level: {count_per_level}")

    train_data, val_and_test_data = train_test_split(
        df,
        test_size=0.3,
        stratify=df["Bloom Level"],
        random_state=42
    )

    validation_data, test_data = train_test_split(
        val_and_test_data,
        test_size=0.333,
        stratify=val_and_test_data["Bloom Level"],
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_data)
    validation_dataset = Dataset.from_pandas(validation_data)
    test_dataset = Dataset.from_pandas(test_data)

    print("\nData split is complete")
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(validation_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    return DatasetDict({
        "train" : train_dataset,
        "validation" : validation_dataset,
        "test" : test_dataset
    })


def dataset_weights(dataset, level_to_id):
    """
    Calculates the weights for the different Bloom Levels.

    Returns a PyTorch tensor of the weights for each bloom level
    """

    num_level = []
    for level in dataset['train']['Bloom Level']:
        num_id = level_to_id[level]
        num_level.append(num_id)

    level_counts = {}
    for level_number in num_level:
        level_counts[level_number] = level_counts.get(level_number, 0) + 1
    
    num_examples = len(num_level)

    level_weights = []
    for level_number in range(6):
        weight = num_examples / (6 * level_counts[level_number])
        level_weights.append(weight)
    
    print(f"The level weights are {level_weights}")
    print("Remember, Understand, Apply, Analyze, Evaluate, Create")

    level_weights = torch.tensor(level_weights, dtype=torch.float32)

    return level_weights


def data_prep(model_name, dataset_path):
    """
    Loads, splits, and tokenizes the dataset for training the model

    Returns:
        tokenized_dataset: DatasetDict with tokenized splits
        level_to_id: dict mapping bloom level to number
        id_to_level: dict mapping number to bloom level
        tokenizer: the tokenizer
        level_weights: PyTorch tensor of the weights
    """

    dataset = load_split_dataset(dataset_path)

    level_to_id = {
        "Remember" : 0,
        "Understand" : 1,
        "Apply" : 2,
        "Analyze" : 3,
        "Evaluate" : 4, 
        "Create" : 5
    }
    
    id_to_level = {
        0 : "Remember",
        1 : "Understand",
        2 : "Apply",
        3 : "Analyze",
        4 : "Evaluate", 
        5 : "Create"
    }

    level_weights = dataset_weights(dataset, level_to_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # padding for llamas 3 tokenizer.pad_token = tokenizer.eos_token
 
    print("\nStarting to tokenize the dataset")

    def tokenize_data(data):
        data_tokenized = tokenizer(
            data["Questions"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

        labels = []
        for level in data["Bloom Level"]:
            num_label = level_to_id[level]
            labels.append(num_label)
        
        data_tokenized["labels"] = labels
        return data_tokenized

    tokenized_dataset = DatasetDict({
        "train" : dataset["train"].map(tokenize_data, batched=True),
        "validation" : dataset["validation"].map(tokenize_data, batched=True),
        "test" : dataset["test"].map(tokenize_data, batched=True)
    })

    print("\nTokenization of the dataset is complete")
    print("Example of tokenized dataset")
    print(tokenized_dataset["train"][0])
    
    return tokenized_dataset, level_to_id, id_to_level, tokenizer, level_weights

