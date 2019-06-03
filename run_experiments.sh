#!/bin/bash
# All decoding experiments for the paper should go here.

set -e 
set -o pipefail
   
NUM_DECODES=$1
BATCH_SIZE="10"
ROOT_DIR=".."
TRANSLATE="${ROOT_DIR}/OpenNMT-daphne/translate.py" 
SOURCE_FILE="eval_data/CMDB_prompt_subset.txt"
OUTPUT_DIR="experiments/${NUM_DECODES}decodes"
# MODEL="${ROOT_DIR}/models/opensubtitles_2_6_t_given_s_acc_31.62_ppl_43.79_e10.pt" 
MODEL="${ROOT_DIR}/models/opensubtitles_2_6_t_given_s_acc_32.66_ppl_38.81_e10.pt"
SEED="666"
GPU=2

mkdir -p $OUTPUT_DIR

echo "Standard beam search, beam size ${NUM_DECODES}"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/standard_beam_search_bs"${NUM_DECODES}".json" \
-beam_size "${NUM_DECODES}" \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size "${BATCH_SIZE}" \
-seed "$SEED" \
-fast \
-n_best "${NUM_DECODES}" \
-gpu "${GPU}"

echo "Standard beam search, beam size ${NUM_DECODES}, npad 0.3"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/standard_beam_search_bs"${NUM_DECODES}"_npad0.3.json" \
-beam_size "${NUM_DECODES}" \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size "${BATCH_SIZE}" \
-seed "$SEED" \
-fast \
-n_best "${NUM_DECODES}" \
-hidden_state_noise 0.3 \
-gpu "${GPU}"

echo "Random sampling, temperature=1.0"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/random_sampling_temp1.0.json" \
-beam_size 1 \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size "${BATCH_SIZE}" \
-seed "$SEED" \
-fast \
-num_random_samples "${NUM_DECODES}" \
-random_sampling_temp 1.0 \
-random_sampling_topk -1 \
-gpu "${GPU}"

echo "Random sampling, temperature=0.7"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/random_sampling_temp0.7.json" \
-beam_size 1 \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size "${BATCH_SIZE}" \
-seed "$SEED" \
-num_random_samples "${NUM_DECODES}" \
-random_sampling_temp 0.7 \
-random_sampling_topk -1 \
-gpu "${GPU}"

echo "Random sampling, temperature=0.5"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/random_sampling_temp0.5.json" \
-beam_size 1 \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size "${BATCH_SIZE}" \
-seed "$SEED" \
-num_random_samples "${NUM_DECODES}" \
-random_sampling_temp 0.5 \
-random_sampling_topk -1 \
-gpu "${GPU}"

echo "Random sampling, temperature=1.0, sample from top 10."
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/random_sampling_temp1.0_top"${NUM_DECODES}".json" \
-beam_size 1 \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size "${BATCH_SIZE}" \
-seed "$SEED" \
-num_random_samples "${NUM_DECODES}" \
-random_sampling_temp 1.0 \
-random_sampling_topk 10 \
-gpu "${GPU}"

echo "K Per Candidate beam search"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/beam_search_bs"${NUM_DECODES}"_kpercand3.json" \
-beam_size "${NUM_DECODES}" \
-n_best "${NUM_DECODES}" \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size 1 \
-k_per_cand 3 \
-gpu "${GPU}"

echo "Diverse beam search"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/diverse_beam_search_bs"${NUM_DECODES}"_dbs0.8.json" \
-beam_size "${NUM_DECODES}" \
-n_best "${NUM_DECODES}" \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size 1 \
-hamming_penalty 0.8 \
-gpu "${GPU}"

echo "Iterative beam search"
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/iterative_beam_search_bs5_ibs"${NUM_DECODES}".json" \
-beam_size 5 \
-n_best 5 \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size 1 \
-beam_iters "${NUM_DECODES}" \
-gpu "${GPU}"

echo "Clustering beam search"
# (NOTE: This takes longer because it loads in GloVe embeddings)
python3 "$TRANSLATE" \
-model "$MODEL" \
-src "$SOURCE_FILE" \
-output "${OUTPUT_DIR}/clustering_beam_search_bs"${NUM_DECODES}"_cbs5.json" \
-beam_size "${NUM_DECODES}" \
-n_best "${NUM_DECODES}" \
-max_length 50 \
-block_ngram_repeat 0 \
-replace_unk \
-batch_size 1 \
-num_clusters 5 \
-cluster_embeddings_file /data1/embeddings/eng/glove.42B.300d.txt \
-gpu "${GPU}"
