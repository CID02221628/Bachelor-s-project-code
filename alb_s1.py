import concurrent.futures
import re
import emoji
import torch
import pandas as pd
from datasets import load_dataset, Dataset
import torch.nn.functional as F
import json
import os
import time
import random
import numpy as np
from transformers import (
    AlbertTokenizer,
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AlbertConfig
)
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score


class ALBERTSentimentBaseProcessor:
    """
    Base class containing shared functionality for both fine-tuning and inference.
    """
    def __init__(self, max_seq_length=128, model_size='xlarge'):
        self.max_seq_length = max_seq_length
        self.model_size = model_size
        model_name = 'albert-xlarge-v2' if model_size == 'xlarge' else 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = None
        self.output_mode = None     # Will be set based on model's num_labels
        self.num_labels = None      # To store num_labels from config
        self.dataset_config = None  # To store dataset configurations

    def clean_text(self, text, remove_mentions=True, remove_urls=True, segment_hashtags=True, replace_emojis=True):
        """
        Cleans text for sentiment analysis.
        """
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        if remove_urls:
            text = re.sub(r'http\S+|www.\S+', '', text)
        if segment_hashtags:
            text = re.sub(r'#(\w+)', lambda x: ' '.join(re.findall(r'[A-Z][a-z]+|\w+', x.group(1))), text)
        if replace_emojis:
            text = emoji.demojize(text)

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize_text(self, text):
        """
        Tokenizes text using the ALBERT tokenizer with fixed padding and truncation.
        """
        return self.tokenizer(
            text,
            truncation=True,          # Truncate sequences longer than max_length
            padding='max_length',     # Pad sequences shorter than max_length
            max_length=self.max_seq_length,  # The fixed maximum length for all sequences
            return_tensors='pt'
        )

    def parallel_tokenization(self, texts):
        """
        Tokenizes texts in parallel using concurrent futures.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tokenized_texts = list(executor.map(self.tokenize_text, texts))
        return tokenized_texts

    def load_dataset_config(self, dataset_name):
        """
        Loads dataset-specific configurations from a JSON file.
        """
        with open('dataset_configs.json', 'r') as f:
            configs = json.load(f)
        if dataset_name in configs:
            self.dataset_config = configs[dataset_name]
            self.num_labels = self.dataset_config.get('num_labels')  # Set num_labels here
            return self.dataset_config
        else:
            raise ValueError(f"Dataset configuration for '{dataset_name}' not found.")

    def initialize_model(self, sentiment_context=None):
        model_name = 'albert-xlarge-v2' if self.model_size == 'xlarge' else 'albert-base-v2'

        if sentiment_context is not None:
            print(f"Loading fine-tuned weights for sentiment context '{sentiment_context}'")
            try:
                # Load weights_filepath and output_mode from metadata.json
                metadata_file_path = 'metadata.json'
                with open(metadata_file_path, 'r') as f:
                    metadata_list = json.load(f)
                # Find the entry with the matching sentiment_context
                metadata = next((item for item in metadata_list if item['version_name'] == sentiment_context), None)
                if metadata is None:
                    raise FileNotFoundError(f"No metadata found for sentiment context '{sentiment_context}'.")
                weights_filepath = metadata['weights_filepath']
                weights_dir = os.path.dirname(weights_filepath)
                if not os.path.isdir(weights_dir):
                    raise FileNotFoundError(f"The specified model directory '{weights_dir}' does not exist.")
                # Load the model from the specified directory
                self.output_mode = metadata.get('output_mode')
                if self.output_mode is None:
                    raise ValueError(f"'output_mode' not found in metadata for '{sentiment_context}'.")
                self.model = AlbertForSequenceClassification.from_pretrained(weights_dir)
                # Ensure num_labels matches the current dataset
                self.model.config.num_labels = self.num_labels
                config_message = f"Running with fine-tuned weights for the {self.model_size} model, sentiment context '{sentiment_context}', configured for num_labels {self.num_labels}."
            except Exception as e:
                print(f"Failed to load fine-tuned weights for sentiment context '{sentiment_context}'. Defaulting to pre-trained weights. Error: {e}")
                # Load pre-trained model
                config = AlbertConfig.from_pretrained(model_name, num_labels=self.num_labels)
                self.model = AlbertForSequenceClassification.from_pretrained(model_name, config=config)
                # Set default output_mode based on model's config
                self.output_mode = self.num_labels
                config_message = f"Model initialized with {model_name}, configured for num_labels {self.num_labels}."
        else:
            # Load pre-trained model
            config = AlbertConfig.from_pretrained(model_name, num_labels=self.num_labels)
            self.model = AlbertForSequenceClassification.from_pretrained(model_name, config=config)
            # Set default output_mode based on model's config
            self.output_mode = self.num_labels
            config_message = f"Model initialized with {model_name}, configured for num_labels {self.num_labels}."

        print(config_message)


class ALBERTSentimentFineTuner(ALBERTSentimentBaseProcessor):
    """
    Class for fine-tuning the ALBERT model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = None
        self.dataset_name = None

    def prepare_finetuning_dataset(self, dataset_name='sst', split='train', sample_size=None, shuffle_data=True):
        """
        Loads, samples, cleans, tokenizes, and prepares the dataset for fine-tuning.
        Returns a dataset with input_ids, attention_mask, and labels.
        """
        self.dataset_name = dataset_name
        self.load_dataset_config(dataset_name)  # Load dataset configurations
        config = self.dataset_config

        data_source = config.get('data_source', 'huggingface')

        if data_source == 'local_csv':
            # Load local CSV file from the same directory
            try:
                csv_file_path = os.path.join(os.path.dirname(__file__), 'test_output.csv')
                df = pd.read_csv(csv_file_path)
                dataset = Dataset.from_pandas(df)
            except Exception as e:
                raise ValueError(f"Failed to load local CSV file for dataset '{dataset_name}': {e}")
        else:
            # Load the dataset from Hugging Face Datasets
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

        # Handle sampling from large datasets efficiently
        if sample_size:
            total_examples = len(dataset)
            if sample_size > total_examples:
                sample_size = total_examples

            if shuffle_data:
                indices = random.sample(range(total_examples), sample_size)
            else:
                indices = list(range(sample_size))

            dataset = dataset.select(indices)
        else:
            if shuffle_data:
                dataset = dataset.shuffle(seed=42)

        text_field = config.get('text_field')
        label_field = config.get('label_field')

        # Clean the text
        texts = [self.clean_text(example[text_field]) for example in dataset]

        # Map labels if label_mapping is provided
        label_mapping = config.get('label_mapping')
        if label_mapping:
            # Convert keys and values to integers
            label_mapping = {int(k): int(v) for k, v in label_mapping.items()}
            labels = [label_mapping[int(example[label_field])] for example in dataset]
        else:
            labels = [example[label_field] for example in dataset]

        # Ensure labels are within the correct range
        max_label = max(labels)
        if self.num_labels is None:
            self.num_labels = max_label + 1
        elif max_label >= self.num_labels:
            raise ValueError(f"Label value {max_label} exceeds num_labels {self.num_labels - 1}.")

        # Tokenize the texts
        tokenized_texts = self.parallel_tokenization(texts)

        # Prepare the final dataset
        final_dataset = Dataset.from_dict({
            'input_ids': [tokenized['input_ids'].squeeze(0) for tokenized in tokenized_texts],
            'attention_mask': [tokenized['attention_mask'].squeeze(0) for tokenized in tokenized_texts],
            'labels': labels  # Mapped labels
        })

        return final_dataset

    def compute_cross_entropy(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_huber_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits").squeeze(-1)
        loss = F.smooth_l1_loss(logits, labels)  # Huber loss
        return (loss, outputs) if return_outputs else loss

    class CustomTrainer(Trainer):
        def __init__(self, *args, loss_function=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function

        def compute_loss(self, model, inputs, return_outputs=False):
            return self.loss_function(model, inputs, return_outputs)

    def compute_classification_metrics(self, model, eval_dataset):
        """
        Computes classification metrics such as F1 score for the model on the eval_dataset.
        """
        trainer = Trainer(model=model)
        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions
        preds = np.argmax(logits, axis=-1)
        labels = np.array(eval_dataset['labels'])

        f1 = f1_score(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)

        metrics = {
            'f1_score': f1,
            'accuracy': accuracy
        }
        return metrics

    def compute_regression_metrics(self, model, eval_dataset):
        """
        Computes regression metrics such as MSE, MAE for the model on the eval_dataset.
        """
        trainer = Trainer(model=model)
        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions.squeeze(-1)
        labels = np.array(eval_dataset['labels'])

        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)

        metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }
        return metrics

    def get_loss_and_hyperparams(self):
        """
        Returns the appropriate loss function, hyperparameters, and evaluation metric based on the loss function.
        """
        if self.dataset_config is None:
            raise ValueError("Dataset configuration not loaded. Ensure 'load_dataset_config' is called before 'get_loss_and_hyperparams'.")

        # Get the loss function name and get the function from self
        loss_function_name = self.dataset_config.get('loss_function')
        if loss_function_name:
            loss_function = getattr(self, loss_function_name)
        else:
            raise ValueError("Loss function not specified in the dataset configuration.")

        # Get the evaluation metric name and get the function from self
        evaluation_metric_name = self.dataset_config.get('evaluation_metric')
        if evaluation_metric_name:
            evaluation_metric = getattr(self, evaluation_metric_name)
        else:
            raise ValueError("Evaluation metric not specified in the dataset configuration.")

        # Set hyperparameters based on the loss function
        if loss_function_name == 'compute_huber_loss':
            hyperparams = {
                'learning_rate': 1e-5,
                'batch_size': 8,
                'epochs': 5,
                'weight_decay': 0.05
            }
        elif loss_function_name == 'compute_cross_entropy':
            hyperparams = {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'epochs': 3,
                'weight_decay': 0.01
            }
        else:
            # Default hyperparameters
            hyperparams = {
                'learning_rate': 1e-5,
                'batch_size': 8,
                'epochs': 3,
                'weight_decay': 0.01
            }

        return loss_function, hyperparams, evaluation_metric

    def fine_tune(self, dataset=None, start_from_sentiment_context=None, save_fine_tune='yes', fine_tune_version_name='', fine_tune_quality=True):
        """
        Fine-tunes the ALBERT model on the provided dataset. Optionally starts from a specific fine-tuned version.
        """
        if dataset is None or 'train' not in dataset or 'test' not in dataset:
            raise ValueError("A properly structured dataset with 'train' and 'test' splits must be provided for fine-tuning.")

        # Get loss function, hyperparameters, and evaluation metric
        loss_function, hyperparams, evaluation_metric = self.get_loss_and_hyperparams()
        self.loss_function = loss_function

        # Ensure self.num_labels is set before initializing the model
        if self.num_labels is None:
            raise ValueError("Number of labels (num_labels) not set. Please ensure that prepare_finetuning_dataset has been called and num_labels is set.")

        # Load the specified version's weights before fine-tuning
        if start_from_sentiment_context is not None:
            self.initialize_model(sentiment_context=start_from_sentiment_context)
        else:
            # Ensure model is initialized with default weights if no version is specified
            self.initialize_model()

        if self.model is None:
            raise ValueError("Model is not initialized. Ensure that `initialize_model` has been called successfully.")

        # Set up the training arguments using hyperparameters
        training_args = TrainingArguments(
            output_dir='./results',
            learning_rate=hyperparams['learning_rate'],
            per_device_train_batch_size=hyperparams['batch_size'],
            num_train_epochs=hyperparams['epochs'],
            weight_decay=hyperparams.get('weight_decay', 0.0),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch"
        )

        # Dynamic padding with DataCollator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Use the CustomTrainer
        trainer = self.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
            loss_function=self.loss_function  # Pass the loss function here
        )

        # Start training
        trainer.train()
        print("Completed fine-tuning on the dataset.")

        if save_fine_tune == 'yes':
            # Save the model with metadata after fine-tuning
            sample_size = len(dataset['train'])
            hyperparameters = {
                'learning_rate': hyperparams['learning_rate'],
                'batch_size': hyperparams['batch_size'],
                'epochs': hyperparams['epochs'],
            }
            if 'weight_decay' in hyperparams:
                hyperparameters['weight_decay'] = hyperparams['weight_decay']

            model_type = 'albert-xlarge-v2' if self.model_size == 'xlarge' else 'albert-base-v2'

            self.save_model_with_metadata(
                model=self.model,
                sample_size=sample_size,
                hyperparameters=hyperparameters,
                dataset_name=self.dataset_name,
                model_type=model_type,
                fine_tune_version_name=fine_tune_version_name,
                fine_tune_quality=fine_tune_quality,
                eval_dataset=dataset['test'],
                evaluation_metric=evaluation_metric
            )

    def save_model_with_metadata(self, model, sample_size, hyperparameters, dataset_name, model_type, fine_tune_version_name='', fine_tune_quality=True, eval_dataset=None, evaluation_metric=None):
        """
        Saves the fine-tuned model along with associated metadata.
        """
        # Determine the version name
        if fine_tune_version_name:
            version_name = fine_tune_version_name
        else:
            version_name = f"version_{int(time.time())}"  # Generate a unique version name based on the current timestamp
        print(f"Auto-saving model with version name: {version_name}")

        # Create the directory for this version
        save_directory = os.path.join('model_versions', version_name)
        os.makedirs(save_directory, exist_ok=True)

        # Save the model using the transformers library
        model.save_pretrained(save_directory)

        # Compute quality metrics if requested
        if fine_tune_quality and eval_dataset is not None and evaluation_metric is not None:
            metrics = evaluation_metric(model, eval_dataset)
        else:
            metrics = {}

        # Create metadata dictionary (flattened)
        metadata = {
            'version_name': version_name,
            'sample_size': sample_size,
            'dataset': dataset_name,
            'model_type': model_type,
            'weights_filepath': os.path.join(save_directory, 'pytorch_model.bin'),
            'output_mode': self.num_labels  # Use self.num_labels
        }

        # Add hyperparameters directly to metadata
        metadata.update(hyperparameters)
        # Add metrics directly to metadata
        metadata.update(metrics)

        # Load existing metadata, if any, and append the new metadata
        metadata_file_path = 'metadata.json'
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = []

        existing_metadata.append(metadata)

        # Save all metadata to the centralized JSON file
        with open(metadata_file_path, 'w') as f:
            json.dump(existing_metadata, f, indent=4)

        print(f"Model and metadata for version '{version_name}' saved successfully.")


class ALBERTSentimentInferencer(ALBERTSentimentBaseProcessor):
    """
    Class for running inference using the ALBERT model.
    """
    def run_inference(self, texts, use_fine_tuned_weights=True, sentiment_context=None, dataset_name='sst'):
        """
        Runs inference on a list of texts using either the pre-trained or fine-tuned ALBERT model.
        Optionally uses a specific version of fine-tuned weights.
        """
        # Load dataset config to get num_labels
        self.load_dataset_config(dataset_name)

        if use_fine_tuned_weights and sentiment_context is not None:
            try:
                self.initialize_model(sentiment_context=sentiment_context)
            except Exception as e:
                print(f"Failed to load fine-tuned weights for sentiment context '{sentiment_context}'. Defaulting to pre-trained weights. Error: {e}")
                self.initialize_model()
        else:
            # Ensure the model is initialized with default weights
            self.initialize_model()

        if self.model is None:
            raise ValueError("Model is not initialized. Ensure that `initialize_model` has been called successfully.")

        # Clean the texts
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Tokenize texts
        tokenized_texts = self.parallel_tokenization(cleaned_texts)

        # Prepare the dataset
        inference_dataset = Dataset.from_dict({
            'input_ids': [tokenized['input_ids'].squeeze(0) for tokenized in tokenized_texts],
            'attention_mask': [tokenized['attention_mask'].squeeze(0) for tokenized in tokenized_texts],
        })

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(inference_dataset)

        if self.output_mode == 1:
            # Continuous output, no softmax applied
            results = torch.tensor(predictions.predictions).squeeze(-1)
        else:
            # Categorical output, apply softmax
            results = F.softmax(torch.tensor(predictions.predictions), dim=-1)

        # Summary print statement
        model_type = self.model_size
        fine_tuning_info = sentiment_context if sentiment_context else "pre-trained"
        print(f"Inference completed using {model_type} model on the {dataset_name} dataset with {fine_tuning_info} weights.")

        return results


class SentimentCSVDataSaver:
    def __init__(self, dataset_handler, sentiment_context=None):
        """
        Initializes the SentimentCSVDataSaver with a reference to DatasetHandler and sentiment context.
        """
        self.dataset_handler = dataset_handler
        self.sentiment_context = sentiment_context

    def save_results(self, predictions):
        """
        Saves sentiment analysis results to the CSV file based on the output_mode from metadata.
        """
        # Read the existing CSV file
        df = self.dataset_handler.read_csv()

        # Get the output_mode from metadata
        output_mode = self.get_output_mode()

        # Add analysis results
        if output_mode == 1:
            df['Sentiment_Score'] = predictions.tolist()
        else:
            # For categorical output, add columns for each class
            num_labels = output_mode
            class_labels = {
                2: ['Negative', 'Positive'],
                3: ['Negative', 'Neutral', 'Positive'],
                5: ['Very_Negative', 'Negative', 'Neutral', 'Positive', 'Very_Positive']
            }
            labels = class_labels.get(num_labels, [f'Class_{i}' for i in range(num_labels)])
            probabilities = predictions.tolist()
            for i, label in enumerate(labels):
                df[label] = [prob[i] for prob in probabilities]

        # Write back to CSV
        self.dataset_handler.write_csv(df)

    def get_output_mode(self):
        """
        Retrieves the output_mode from the metadata JSON based on the sentiment_context.
        """
        if self.sentiment_context is None:
            # Default to num_labels if no sentiment_context is provided
            return self.dataset_handler.num_labels

        # Load metadata
        metadata_file_path = 'metadata.json'
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)
        # Find the entry with the matching sentiment_context
        metadata = next((item for item in metadata_list if item['version_name'] == self.sentiment_context), None)
        if metadata is None:
            raise FileNotFoundError(f"No metadata found for sentiment context '{self.sentiment_context}'.")
        output_mode = metadata.get('output_mode')
        if output_mode is None:
            raise ValueError(f"'output_mode' not found in metadata for '{self.sentiment_context}'.")
        return output_mode
