# Sense-aware Lexical Sophistication

This project releases the automatic analysis tool and the resources in the paper:

<em>Xiaofei Lu and Renfen Hu. Sense-aware lexical sophistication indices and their relationship to second language writing quality. Under review</em>

## Prerequisites

1. Install Python and packages
*   **Python3**
*   **[`NLTK`](http://www.nltk.org/install.html)**
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`bert_serving`](https://pypi.org/project/bert-serving-server/)**

2. Download the pre-trained language model
In this study, we used the [`uncased BERT-Base`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model to generate deep contextualized word embeddings. More options can be found at https://github.com/google-research/bert.

3. Download the sense embeddings
The sense embeddings constructed in this study (about 107M) can be download at [Google Drive] or [BNU cloud storage](https://pan.bnu.edu.cn/l/yo7MZF).
Please place the file in the `dict` folder before running the codes.

## Automatic analysis 

Step 1. Start the BERT service
```python
bert-serving-start \
    -pooling_strategy NONE \
    -max_seq_len 128 \
    -pooling_layer -1 \
    -device_map 0 \           # please specify the GPU device you plan to use
    -model_dir bert_base \    # please specify the directory of the pre-trained BERT model
    -show_tokens_to_client \
    -priority_batch_size 32   # batch_size is set based on GPU memory, in our study we use the Nvidia 1080ti with 11G memory
```

Step 2. Tag the senses for polysemous words

```python
python tag_text_server.py
```
In this step, we firstly conduct sentence tokenization for each essay, which can be seen in the folder of **`samples`**. After that, we label the sense for each polysemous word sentence by sentence. The sense information is from the online version of [Oxford dictionary](https://www.lexico.com/).
The sense tagging results can be seen in the folder of **`output`**.

Step 3. Terminate the bert service
```python
bert-serving-terminate -port 5555
```

Step 4. Compute the sense-aware lexical sophistication indices

```python
python sense_aware_indices.py
```
The result can be seen in **`indices.csv`**.
