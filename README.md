# the_beamers
All experiments and evaluation code for decoding diversity project!

# Instructions for running diverse decoding experiments.
Set the vars at the top of `run_experiments.sh` appropriately to refer to where OpenNMT-py and your model checkpoint are stored.
Run
```
./run_experiments.sh 100
python convert100to10.py \
--input_dir experiments/100decodes/ \
--output_dir experiments/100to10decodesNoPDC \
--num_cands 10
```
to produce 10 diverse decodings from each method.

Run
```
./run_experiments.sh 100
python cluster_postprocessing.py \
--input_dir experiments/100decodes/ \
--output_dir experiments/100to10decodesPDC \
--method kmeans_mod \
--num_cands 10
```
to first produce 100 diverse decodings from each method, and then narrow these down to 10 each using clustering post-processing.

# Instructions for running automatic evaluation.
Run 
```
python analyze_diversity.py -dir experiments/100to10decodesNoPDC

python analyze_diversity.py -dir experiments/100to10decodes
```

