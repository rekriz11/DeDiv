import os
import glob
import json
import csv

if __name__ == '__main__':
  names = ['experiment', 'SPICE']

  fout = open('results.csv', 'w')
  writer = csv.DictWriter(fout, names)
  writer.writeheader()
   
  for fname in sorted(glob.glob('caption_eval_tmp/*_test.json')):
    print(fname)
    with open(fname) as fin:
      results = json.load(fin)
      to_write = results['overall']
      to_write['experiment'] = os.path.basename(fname.replace('_eval_test.json', ''))
      writer.writerow(to_write)
  fout.close()
