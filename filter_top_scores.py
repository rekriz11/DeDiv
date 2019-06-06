"""Filter the candidate outputs to the ones with highest likelihood score."""

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

def filter_candidates(candidates, scores, num_candidates):
  """Returns the num_candidates canadidates with the lowest scores."""
  new_candidates, new_scores = [], []

  sorted_by_score = sorted(zip(candidates, scores),
                           key=lambda x: x[1], reverse=True)
  
  for cand, score in sorted_by_score:
    if cand not in new_candidates and len(cand) > 0 and len(new_candidates) < num_candidates:
      new_candidates.append(cand)
      new_scores.append(score)

  assert len(new_candidates) == num_candidates
  return new_candidates, new_scores


def main(opt):
  if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

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

        for ex_num, example in enumerate(experiment['results']):
          candidates = example['pred']
          scores = example['scores']
          candidates, scores = filter_candidates(candidates, scores, opt.num_cands)

          if ex_num < 3:
            print(candidates)
            print(scores)

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
  opt = parser.parse_args()

  main(opt)
