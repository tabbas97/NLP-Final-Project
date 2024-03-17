# NLP-Final-Project
Text Summarization

**Done in the Spring of 2023**

## Goal

We plan on implementing text summarization from scratch. However the scope of the project was condensed to focus more on the nuances between the two types of summarization, Extractive and Abstractive.

We use ROUGE metrics to find the optimal model to summarize text.

## Results

|                	| Rouge - 1 	| Rouge - 2 	| Rouge - L 	|
|----------------	|-----------	|-----------	|-----------	|
| Extractive - 1 	| 24.6      	| 7.2       	| 22.7      	|
| Extractive - 2 	| 11.6      	| 11.8      	| 12.1      	|
| T5 - Small     	| 40.2      	| 17.3      	| 37.3      	|
| Mini BERT      	| 41.0      	| 18.2      	| 38        	|
| Pegasus        	| 43.9      	| 21.2      	| 40.76     	|

## Challenges

1. The primary challenges with performing said evaluations was the lack of compute. We were unable to run long-running instances of the code to train/evaluate on bigger subset of data.
