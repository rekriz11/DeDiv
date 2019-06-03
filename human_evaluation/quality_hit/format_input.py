import sys
import os
from os import listdir
from os.path import isfile, join
import json
import random
import csv
from mosestokenizer import MosesDetokenizer

num_fixes = 0


## Gets name of files in a list from a directory
def get_all_files(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))
             and ".json" in f]
    return files

## Flattens a two-dimensional list   
def flatten(listoflists):
    list = [item for sublist in listoflists for item in sublist]
    return list

## Detokenize and fix weird contractions
def fix(listy, detokenize):
    detok = detokenize(listy)
    fixed = str(detok)

    num_fixes = 0

    starts = ["i", "you", "he", "they", "we"]
    punctuation = ["!", "?", "."]

    for s in starts:
        for p in punctuation:
            bad1 = s + p + " l"
            fixed = fixed.replace(bad1, s + "'ll")
            
            bad2 = s + p + " e"
            fixed = fixed.replace(bad2, s + "'ve")
        
    fixed = fixed.replace("'r e", "'re")

    

    if fixed != detok:
        return fixed, 1
    else:
        return fixed, 0
    

# Load all json files
def load_directory(dir1, dir2, detokenize):    
    files1 = get_all_files(dir1)
    paths1 = [dir1 + "/" + file for file in files1]

    files2 = get_all_files(dir2)
    paths2 = [dir2 + "/" + file for file in files2]

    files1 = ["original/" + f for f in files1]
    files2 = ["clustered/" + f for f in files2]

    filepaths = paths1 + paths2
    files = files1 + files2

    num_fixes = 0
    inputs, preds, scores, systems = [], [], [], []
    
    for i, path in enumerate(filepaths):
        inps, prds, scrs, systms = [], [], [], []
        with open(path) as f:
            outputs = json.load(f)

            for result_dict in outputs["results"]:
                fixed_input, fix_bool = fix(result_dict["input"], detokenize)
                inps.append(fixed_input)
                num_fixes += fix_bool

                fixed_preds = []
                for p in result_dict["pred"]:
                    fixed_pred, fix_bool = fix(p, detokenize)
                    fixed_preds.append(fixed_pred)
                    num_fixes += fix_bool

                sorted_scores = sorted(result_dict["scores"], reverse=True)
                sorted_indices = [result_dict["scores"].index(s) for s in sorted_scores]
                sorted_preds = [fixed_preds[ind] for ind in sorted_indices]
                prds.append(sorted_preds)
                scrs.append(sorted_scores)
                
                systms.append([files[i] + "_" + str(k) for k in range(len(sorted_scores))])
            
        inputs.append(inps)
        preds.append(prds)
        scores.append(scrs)
        systems.append(systms)

    print("NUMBER OF FIXES: " + str(num_fixes))

    return inputs, preds, scores, systems


def make_rows(inputs, preds, scores, systems):
    mturk_input = [[] for i in range(len(inputs[0]))]
    
    for j in range(len(inputs[0])):
        input_current = inputs[0][j]
        preds_current = []
        systems_current = []

        for i in range(len(preds)):
            preds_current += preds[i][j]
            systems_current += systems[i][j]

        random_inds = [k for k in range(len(preds_current))]
        random.shuffle(random_inds)

        cur_start = 0
        while cur_start < len(random_inds) - 1:
            hit = random_inds[cur_start:cur_start+5]
            inputy = [input_current] + [preds_current[k] for k in hit] + [systems_current[k] for k in hit]
            mturk_input[j].append(inputy)
            cur_start += 5

    '''
    print("NUMBER OF TASKS: ")
    print(len(mturk_input))
    print(sum([len(m) for m in mturk_input]))
    '''

    rows = []
    current_hit_id = [0 for i in range(len(mturk_input))]

    c = 0
    while min(current_hit_id) < len(mturk_input[0]):
        available_sents = [i for i in range(len(current_hit_id)) \
                             if current_hit_id[i] == min(current_hit_id)]
        '''
        if c < 3:
            print(available_sents)
        '''
        
        random.shuffle(available_sents)
        current_sent_ids = available_sents[:5]

        '''
        ## Debug to make sure it's working
        if c < 3:
            print(current_sent_ids)
        '''

        row = []
        for i in current_sent_ids:
            current_hit = mturk_input[i][current_hit_id[i]] + [i]
            row += current_hit
            current_hit_id[i] += 1

            '''
            if c < 3:
                print(current_hit)
            '''

        rows.append(row)
        c += 1
    
    #print(len(rows))
    return rows


def output_csv(rows, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        firstrow = ['input1', 'sys11', 'sys12', 'sys13', 'sys14', 'sys15', 'sysid11', 'sysid12', 'sysid13', 'sysid14', 'sysid15', 'sentid1', \
                    'input2', 'sys21', 'sys22', 'sys23', 'sys24', 'sys25', 'sysid21', 'sysid22', 'sysid23', 'sysid24', 'sysid25', 'sentid2', \
                    'input3', 'sys31', 'sys32', 'sys33', 'sys34', 'sys35', 'sysid31', 'sysid32', 'sysid33', 'sysid34', 'sysid35', 'sentid3', \
                    'input4', 'sys41', 'sys42', 'sys43', 'sys44', 'sys45', 'sysid41', 'sysid42', 'sysid43', 'sysid44', 'sysid45', 'sentid4', \
                    'input5', 'sys51', 'sys52', 'sys53', 'sys54', 'sys55', 'sysid51', 'sysid52', 'sysid53', 'sysid54', 'sysid55', 'sentid5']
        csvwriter.writerow(firstrow)
        for row in rows:
            row_fixed = []
            for r in row:
                try:
                    row_fixed.append(r.encode('ascii', 'ignore').decode('ascii'))
                except AttributeError:
                    row_fixed.append(r)
            csvwriter.writerow(row_fixed)

def main(system_outputs_folder, clustered_outputs_folder, output_file):
    random.seed(37)
    detokenize = MosesDetokenizer('en')
    inputs, preds, scores, systems = load_directory(system_outputs_folder, clustered_outputs_folder, detokenize)
    rows = make_rows(inputs, preds, scores, systems)

    output_csv(rows, output_file)
    
    

if __name__ == '__main__':
    system_outputs_folder = sys.argv[1]
    clustered_outputs_folder = sys.argv[2]
    output_file = sys.argv[3]
    main(system_outputs_folder, clustered_outputs_folder, output_file)


'''
python3 format_input.py \
/data2/the_beamers/the_beamers_reno/experiments/10decodes/ \
/data2/the_beamers/the_beamers_reno/experiments/100to10decodes/ \
input/input.csv

python3 format_input.py \
/data2/the_beamers/the_beamers_reno/experiments_joao_model2/10decodes/ \
/data2/the_beamers/the_beamers_reno/experiments_joao_model2/100to10decodes/ \
input/input_joao_model2.csv
'''
