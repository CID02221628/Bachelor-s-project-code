from dataset_handling import DatasetHandler
from alb_s1 import ALBERTSentimentFineTuner, ALBERTSentimentInferencer, SentimentCSVDataSaver
from lda_module import LDATextProcessor, LDAProcessor, LDACSVDataSaver
from bertopic_module import BERTopicProcessor, BERTopicCSVDataSaver
from kcluster_module import TextProcessingAndDatabase, KclusterAnalysis, KclusterCSVDataSaver

if __name__ == "__main__":

    # Top Dog Toggles
    sentiment_context = 'S140test18'  # Example sentiment context
    our_datasets = 'sst'  # Example dataset
    your_dataset_toggle = True  # Toggle for including your_dataset

    # Dataset-related toggles
    sample_size = 50  # Number of random samples to load
    shuffle_data = True  # Now controls random sampling
    split = 'train'

    # Sentiment Toggles
    sentiment_analysis = True
    max_seq_length = 128
    model_size = 'base'

    # Fine-tune parameters
    fine_tune_or_inference = 'inference'  # Change to 'fine_tune' when running fine-tuning
    save_fine_tune = 'yes'  # 'yes' or 'no', controls whether to save fine-tuned model
    fine_tune_version_name = 'S140test18'  # Name to use when saving fine-tuned model
    fine_tune_quality = True  # If True, compute quality metrics during fine-tuning

    # LDA toggles
    LDA_analysis = True  # Set to True to enable LDA analysis

    # BERTopic Toggles
    BERTopic_analysis = True  # Toggle for BERTopic analysis
    BERTopic_num_topics = 'auto'  # Number of topics for BERTopic

    # Kcluster Toggle
    Kcluster_analysis = True  # Set to True to enable K-cluster analysis

    # Initialize DatasetHandler
    dataset_handler = DatasetHandler(csv_file='output.csv')

    # Initialize or read the CSV file
    dataset_handler.initialize_csv(
        our_datasets=our_datasets,
        your_dataset_toggle=your_dataset_toggle,
        sample_size=sample_size,
        shuffle_data=shuffle_data,
        split=split
    )

    if fine_tune_or_inference == 'fine_tune' and sentiment_analysis:
        # Initialize the fine-tuner
        fine_tuner = ALBERTSentimentFineTuner(
            max_seq_length=max_seq_length,
            model_size=model_size
        )

        # Prepare the dataset for fine-tuning
        fine_tune_dataset = {
            'train': fine_tuner.prepare_finetuning_dataset(
                dataset_name=our_datasets,
                split='train',
                sample_size=sample_size,
                shuffle_data=shuffle_data
            ),
            'test': fine_tuner.prepare_finetuning_dataset(
                dataset_name=our_datasets,
                split='test',
                sample_size=sample_size,
                shuffle_data=shuffle_data
            )
        }

        # Ensure num_labels is set based on the dataset
        if fine_tuner.num_labels is None:
            raise ValueError("num_labels not set. Ensure that prepare_finetuning_dataset sets num_labels before fine-tuning.")

        # Initialize the model with the correct num_labels and sentiment context
        fine_tuner.initialize_model(sentiment_context=sentiment_context)

        # Run fine-tuning with the prepared dataset
        fine_tuner.fine_tune(
            dataset=fine_tune_dataset,
            start_from_sentiment_context=sentiment_context,
            save_fine_tune=save_fine_tune,
            fine_tune_version_name=fine_tune_version_name,
            fine_tune_quality=fine_tune_quality
        )

    else:
        # For each analysis
        if sentiment_analysis:
            # Initialize the inferencer
            inferencer = ALBERTSentimentInferencer(
                max_seq_length=max_seq_length,
                model_size=model_size
            )

            # Get texts for analysis
            texts_for_analysis = dataset_handler.get_texts_for_analysis()

            # Run inference with the texts
            predictions = inferencer.run_inference(
                texts=texts_for_analysis,
                use_fine_tuned_weights=sentiment_context is not None,
                sentiment_context=sentiment_context,
                dataset_name=our_datasets
            )

            # Save sentiment results to CSV
            sentiment_csv_saver = SentimentCSVDataSaver(
                dataset_handler=dataset_handler,
                sentiment_context=sentiment_context
            )
            sentiment_csv_saver.save_results(
                predictions=predictions
            )

        if LDA_analysis:
            # Initialize the LDA processor
            lda_text_processor = LDATextProcessor()
            lda_processor = LDAProcessor(num_topics=5)  # Adjust num_topics as needed

            # Get texts for analysis
            texts_for_analysis = dataset_handler.get_texts_for_analysis()

            # Preprocess the texts
            lda_texts = lda_text_processor.preprocess_texts(texts_for_analysis)

            # Perform LDA
            lda_model, corpus = lda_processor.perform_lda(lda_texts)

            # Get topic matrix
            topic_matrix = lda_processor.get_topic_matrix(lda_model, corpus)

            # Perform PCA
            pca_result = lda_processor.perform_pca(topic_matrix)

            # Perform t-SNE
            tsne_result = lda_processor.perform_tsne(topic_matrix)

            # Convert results to lists
            pca_coordinates = pca_result.tolist()
            tsne_coordinates = tsne_result.tolist()

            # Save LDA results to CSV
            lda_csv_saver = LDACSVDataSaver(dataset_handler)
            lda_csv_saver.save_results(
                pca_coordinates=pca_coordinates,
                tsne_coordinates=tsne_coordinates
            )

        if BERTopic_analysis:
            
            bertopic_processor = BERTopicProcessor(num_topics=BERTopic_num_topics)

            # Get texts for BERTopic analysis
            texts_for_bertopic = dataset_handler.get_texts_for_analysis()

            # Initialize the BERTopic model based on the size of the dataset
            bertopic_processor.initialize_model(len(texts_for_bertopic))

            # Run BERTopic analysis
            bertopic_results = bertopic_processor.perform_bertopic(texts=texts_for_bertopic)

            # Save visualizations as objects for later use
            BERT_distance = bertopic_processor.topic_model.visualize_topics()
            BERT_similarity_matrix = bertopic_processor.topic_model.visualize_heatmap()
            
            # Pass bertopic_processor to the CSV saver (optional)
            bertopic_csv_saver = BERTopicCSVDataSaver(dataset_handler=dataset_handler, bertopic_processor=bertopic_processor)
            bertopic_csv_saver.save_results(predictions=bertopic_results)



        if Kcluster_analysis:
            # Initialize the KCluster processor and TextProcessor
            text_processor = TextProcessingAndDatabase(dataset_handler)
            kcluster_processor = KclusterAnalysis(n_components=3)

            # Get texts for K-cluster analysis
            df, texts_for_kcluster = text_processor.process_texts()

            # Perform KMeans clustering and PCA
            if texts_for_kcluster:
                cluster_labels = kcluster_processor.perform_analysis(texts_for_kcluster)
                pca_results, labels = kcluster_processor.perform_analysis(texts_for_kcluster)

                # Save KCluster results to CSV
                kcluster_csv_saver = KclusterCSVDataSaver()
                kcluster_csv_saver.save_analysis_to_csv(pca_results, labels, dataset_handler)

    print("All analyses completed. The results have been updated in 'output.csv'.")


