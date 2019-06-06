# the_beamers
All experiments and evaluation code for decoding diversity project!

# Instructions for running diverse decoding experiments.
Set the vars at the top of `run_experiments.sh` appropriately to refer to where OpenNMT-py and your model checkpoint are stored.
Run
```
./run_experiments.sh 100
python filter_top_scores.py \
--input_dir all_experiments/dialog/100decodes/ \
--output_dir all_experiments/dialog/100to10decodes_withTopScore \
--num_cands 10
```
to first produce 100 diverse decodings from each method, and then narrow these down to 10 by taking the most likely.

Run
TODO: Explain how to run bert-as-service
```
./run_experiments.sh 100
python filter_cluster_postprocessing.py \
--input_dir all_experiments/dialog/100decodes/ \
--output_dir all_experiments/dialog/100to10decodes_withClustering \
--method kmeans_mod \
--num_cands 10
```
to first produce 100 diverse decodings from each method, and then narrow these down to 10 each using clustering post-processing.

# Instructions for running automatic evaluation.
Run 
```
python analyze_diversity.py -dir all_experiments/dialog/100to10decodes_withClustering
python analyze_diversity.py -dir experiments/100to10decodes_withTopScores
```
Each of these commands will create a results.csv file in the specified directory with distinct-1, distinct-2, entropy-2, entropy-4, and other statstics on the outputs.
