# Code for "Comparison of Diverse Decoding Methods from Conditional Language Models"

Please cite the following [paper](https://arxiv.org/pdf/1904.02767.pdf):
```
@inproceedings{ippolito-etal-2019-comparison,
    title = "Comparison of Diverse Decoding Methods from Conditional Language Models",
    author = "Ippolito, Daphne  and
      Kriz, Reno  and
      Sedoc, Joao  and
      Kustikova, Maria  and
      Callison-Burch, Chris",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1365",
}
```

# Outputs Used in Paper Analyses
The paper evaluates on two domains, open-ended dialog and image captioning. The candidate outputs used for running human and automatic evaluation can be found in JSON format [here](https://github.com/rekriz11/DeDiv/tree/master/all_experiments).

Follow the instructions below to rerun inference and reproduce these JSON files.

# Rerunning Open-ended Dialog Experiments
The chitchat model used in the paper was based on an older fork of OpenNMT-py. Please clone the following repo [https://github.com/rekriz11/OpenNMT-py](https://github.com/rekriz11/OpenNMT-py). Our model weights can be downloaded [here](https://www.seas.upenn.edu/~rekriz/opensutitles_model.pt), although any OpenNMT Seq2Seq saved model should work. 

## Instructions for generating canadidates.

First, install [bert-as-service](https://github.com/hanxiao/bert-as-service), download the [uncased pre-trained model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip), and launch a server using:
```
bert-serving-start -model_dir /data1/embeddings/BERT/uncased_L-12_H-768_A-12/ -pooling_strategy REDUCE_MEAN_MAX -num_worker=2 -pooling_layer "-4" -max_seq_len 50
```

Next, set the vars at the top of `run_experiments.sh` appropriately to refer to where OpenNMT-py and your model weights are stored.
Run 
```
./run_experiments.sh 100
python filter_top_scores.py \
--input_dir all_experiments/dialog/100decodes/ \
--output_dir all_experiments/dialog/100to10decodes_withTopScore \
--num_cands 10
```
to first produce 100 diverse decodings from each method, and then narrow these down to 10 by taking the most likely.

Similarly, run
```
./run_experiments.sh 100
python filter_cluster_postprocessing.py \
--input_dir all_experiments/dialog/100decodes/ \
--output_dir all_experiments/dialog/100to10decodes_withClustering \
--method kmeans_mod \
--num_cands 10
```
to first produce 100 diverse decodings from each method, and then narrow these down to 10 each using clustering post-processing.

Lastly, run
```
./run_experiments.sh 10
```
to produce exactly 10 decodings for each method.

# Rerunning Image Captioning Experiments
This section is still in progress. Contact us if you need it imminently. 

# Instructions for evaluating diversity.
Make sure a bert-as-service instance is running (see above section), and then run 
```
python analyze_diversity.py -dir all_experiments/dialog/100to10decodes_withClustering
python analyze_diversity.py -dir experiments/100to10decodes_withTopScores
```
Each of these commands will create a results.csv file in the specified directory with distinct-1, distinct-2, entropy-2, entropy-4, and other statstics about the outputs.
