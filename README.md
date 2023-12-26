# STAR - Sparse Text Approach for Recommendation
STAR is an approach adapting Learned Sparse Retrieval for Recommendation. We provide code and data to run the model on three categories in the Amazon dataset: Books, Movies and TV, Video Games. 
The models that you can run:
- STAR - BERT-based bi-encoder with MLM head and sparse regularizer
- [CUP](https://arxiv.org/pdf/2311.01314.pdf) - BERT-based bi-encoder with MLP, producind 200-dim representations
- BERT_cross - cross encoder

## 1. Train the models
```
python main.py --op train --config_file <path_to_config>
```
Before training star_distil model you need to train and run inference on the bert_cross model. The distilled scores will automatically be generated during bert_cross inference.

## 2. Run inference
```
python main.py --op test --result_folder <path_to_output_folder>
```

## 3. Run evalutaion
```
python launch_in_bulk.py
```
This code with generate the text files with resutls and significance wrt star_distil model.
