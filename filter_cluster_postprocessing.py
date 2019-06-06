"""Filter the candidates by clustering them and then sampling from the clusters."""

import glob
import json
import csv
import editdistance
import nltk
from bert_serving.client import BertClient
from mosestokenizer import MosesDetokenizer
from sklearn.cluster import KMeans
import collections

import configargparse
import numpy as np
import os


def get_embs(candidates, bc, detokenize, normalize=False):
  """Returns the sequence embedding for each candidate."""

  detoked_cands = []
  for i, cand in enumerate(candidates):
    detoked_cands.append(detokenize(cand))
    if len(detokenize(cand)) == 0:
      print(i)
      print(cand)
      print(detokenize(cand))

  embs = bc.encode(detoked_cands)
  if normalize:
    embs = [e / np.linalg.norm(e) for e in embs]

  # This line is necessary depending on how the BERT server is setup.
  # embs = [np.mean(emb, 0) for emb in embs]

  return embs

def remove_duplicates(candidates, scores):
  new_candidates = []
  new_scores = []
  for cand, score in zip(candidates, scores):
    if cand not in new_candidates and len(cand) > 0:
      new_candidates.append(cand)
      new_scores.append(score)
  return new_candidates, new_scores


def distance_filtering(
    candidates, scores, new_count, normalize_embs, bc, detokenize):
  """Greedily take the furthest candidate from the ones taken so far."""

  embs = get_embs(candidates, bc, detokenize, normalize_embs)

  # Take the most likely candidate a sthe first to keep.
  most_likely_cand_idx = np.argmin(scores)
  cand_ids_to_keep = [most_likely_cand_idx]

  # At every step, choose the next candidate to be the one that is most
  # different from the ones that have been chosen so far.
  for _ in range(new_count - 1):
    best_idx_so_far = -1
    best_dist_so_far = 0.0
    for cdx, cand in enumerate(candidates):
      if cdx not in cand_ids_to_keep:
        d = sum(np.linalg.norm(embs[cdx] - embs[mdx]) for mdx in cand_ids_to_keep)
        if d > best_dist_so_far:
          best_dist_so_far = d
          best_idx_so_far = cdx
    cand_ids_to_keep.append(best_idx_so_far)
   
    filtered_cands = [candidates[cdx] for cdx in cand_ids_to_keep]
    filtered_scores = [scores[cdx] for cdx in cand_ids_to_keep]
    return filtered_cands, filtered_scores


def kmeans_filtering(candidates, scores, new_count, normalize_embs, bc, detokenize):
  """Take the most likely candidate from each cluster returned by kmeans."""
 
  embs = get_embs(candidates, bc, detokenize, normalize_embs)
  kmeans = KMeans(n_clusters=new_count).fit(embs)

  filtered_cands = []
  filtered_scores = []

  print('\n===EXAMPLE===')
  for cluster_idx in range(new_count):
    labels = kmeans.labels_
    r_in_cluster = [x for x in zip(candidates, scores, labels) if x[-1] == cluster_idx]

    # Output all of the responses in the cluster, sorted by likelihood.
    r_in_cluster = sorted(r_in_cluster, key=lambda r: r[1], reverse=True)
    print('%d in cluster %d' % (len(r_in_cluster), cluster_idx))

    filtered_cands.append(r_in_cluster[0][0])
    filtered_scores.append(r_in_cluster[0][1])

  return filtered_cands, filtered_scores


def kmeans_mod_filtering(
    candidates, scores, num_clusters, normalize_embs, bc, detokenize):
  """
  After initial k-means clustering, ignore clusters of size <= 2, and
  take top 2 candidates from largest clusters.
  """

  embs = get_embs(candidates, bc, detokenize, normalize_embs)
  kmeans = KMeans(n_clusters=num_clusters).fit(embs)

  r_clusters = []

  for cluster_idx in range(num_clusters):
    labels = kmeans.labels_

    r_in_cluster = [x for x in zip(candidates, scores, labels) if x[-1] == cluster_idx]

    # Output all of the responses in the cluster, sorted by likelihood.
    r_in_cluster = sorted(r_in_cluster, key=lambda r: r[1], reverse=True)
    #print('%d in cluster %d' % (len(r_in_cluster), cluster_idx))

    # Do not consider the responses from clusters of size <= 2
    if len(r_in_cluster) > 2:
      r_clusters.append(r_in_cluster)

  # Sort clusters by size
  r_clusters = sorted(r_clusters, key=len, reverse=True)

  # Get top remaining element from each cluster, prioritizing larger clusters first,
  # until we have the required number of items
  filtered_cands = []
  filtered_scores = []
  
  candidate_index = 0
  while len(filtered_cands) < num_clusters:
    for c in r_clusters:
      try:
        filtered_cands.append(c[candidate_index][0])
        filtered_scores.append(c[candidate_index][1])

        if len(filtered_cands) == num_clusters:
          break
      except IndexError:
        continue
    candidate_index += 1

  return filtered_cands, filtered_scores



def main(opt):
  if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

  bc = BertClient()
  detokenize = MosesDetokenizer('en')

  all_results = {}
  for json_file in glob.glob(os.path.join(opt.input_dir, '*.json')):
    out_json_file = os.path.join(opt.output_dir, os.path.basename(json_file))

    ## Check to make sure file doesn't already exist
    if not os.path.isfile(out_json_file):   
      with open(json_file, 'r') as f:
        try:
          experiment = json.load(f)
          print('Processing ' + json_file)
        except:
          print('Error processing ' + json_file)
          print('Skipping it.')
          continue

        for ex_num, example in enumerate(experiment):
          if ex_num % 10 == 0:
            print("Clustering output: " + str(ex_num))
          
          candidates = example['pred']
          scores = example['scores']
          candidates, scores = remove_duplicates(candidates, scores)

          if opt.method == 'kmeans':
            candidates, scores = kmeans_filtering(
                candidates, scores, opt.num_cands, True, bc, detokenize)
          elif opt.method == 'distance':
            candidates, scores = distance_filtering(
                candidates, scores, opt.num_cands, False, bc, detokenize)
          elif opt.method == 'kmeans_mod':
            candidates, scores = kmeans_mod_filtering(
                candidates, scores, opt.num_cands, True, bc, detokenize)
          else:
            raise ValueError('Not a valid filtering method')

          example['pred'] = candidates
          example['scores'] = scores

      out_json_file = os.path.join(opt.output_dir, os.path.basename(json_file))
      with open(out_json_file, 'w') as f:
        json.dump(experiment, f)
    else:
      print("SKIPPING: " + json_file)


if __name__ == '__main__':
  parser = configargparse.ArgumentParser(
      description='analyze_diversity.py',
      config_file_parser_class=configargparse.YAMLConfigFileParser,
      formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  group = parser.add_argument_group('Arguments')
  group.add('--input_dir', type=str, required=True,
            help='Directory containing json files.')
  group.add('--output_dir', type=str, required=True,
            help='Directory to write out files.')
  group.add('--num_cands', type=int, default=10,
            help='The target number of candidates.')
  group.add('--method', type=str, default='kmeans_mod',
            help='One of [distance, kmeans, kmeans_mod].')
  opt = parser.parse_args()

  main(opt)
