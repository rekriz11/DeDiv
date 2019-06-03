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

def remove_duplicates(candidates, scores, num_candidates):
  # counter = collections.Counter(' '.join(c) for c in candidates)
  new_candidates, new_scores = [], []

  while len(new_candidates) < num_candidates:
    max_ind = scores.index(max(scores))
    if candidates[max_ind] not in new_candidates:
      new_candidates.append(candidates[max_ind])
      new_scores.append(scores[max_ind])
    scores[max_ind] = -1e20
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
          candidates, scores = remove_duplicates(candidates, scores, opt.num_cands)

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
