import sys
import os
from os import listdir
from os.path import isfile, join
import json
import random
import csv


## Gets name of files in a list from a directory
def get_all_files(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))
             and ".json" in f]
    return files

## Flattens a two-dimensional list   
def flatten(listoflists):
    list = [item for sublist in listoflists for item in sublist]
    return list

# Load all json files
def load_directory(dir1, dir2):
    files1 = get_all_files(dir1)
    paths1 = [dir1 + "/" + file for file in files1]

    files2 = get_all_files(dir2)
    paths2 = [dir2 + "/" + file for file in files2]

    files1 = ["original/" + f for f in files1]
    files2 = ["clustered/" + f for f in files2]

    filepaths = paths1 + paths2
    files = files1 + files2

    inputs, preds, scores, systems = [], [], [], []
    for i, path in enumerate(filepaths):
        inps, prds, scrs = [], [], []
        with open(path) as f:
            print(path)
            outputs = json.load(f)

            for result_dict in outputs["results"]:
                inps.append(' '.join(result_dict["input"]))
                prds.append([" ".join(p) for p in result_dict["pred"]])
                scrs.append(result_dict["scores"])

        for j in range(len(inps)):
            inps[j] = inps[j].replace('&apos;', "'")
            inps[j] = inps[j].replace('&#124;', "|")
            for k in range(len(prds[j])):
                prds[j][k] = prds[j][k].replace('&apos;', "'")
                prds[j][k] = prds[j][k].replace('&#124;', "'")
            
        inputs.append(inps)
        preds.append(prds)
        scores.append(scrs)
        systems.append([files[i] for j in range(len(inps))]) 

    return inputs, preds, scores, systems


def make_rows(inputs, preds, scores, systems):
    mturk_input = [[] for i in range(len(inputs[0]))]

    c = 0
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if c == 0:
                print(inputs[i][j])
                print(preds[i][j])
                print(systems[i][j])
                
            random_inds = [k for k in range(len(preds[i][j]))]
            random.shuffle(random_inds)

            hit1 = random_inds[:5]
            input1 = [inputs[i][j]] + [preds[i][j][k] for k in hit1] + [systems[i][j]]
                                             
            hit2 = random_inds[5:]
            input2 = [inputs[i][j]] + [preds[i][j][k] for k in hit2] + [systems[i][j]]
            
            mturk_input[j].append(input1)
            mturk_input[j].append(input2)

            if c == 0:
                print(hit1)
                print(input1)
                print("\n")
                print(hit2)
                print(input2)

            c += 1

    print(len(mturk_input))
    print(sum([len(m) for m in mturk_input]))

    rows = []
    current_hit_id = [0 for i in range(len(mturk_input))]

    c = 0
    while min(current_hit_id) < len(mturk_input[0]):
        available_sents = [i for i in range(len(current_hit_id)) \
                             if current_hit_id[i] == min(current_hit_id)]
        if c < 5:
            print(available_sents)
        random.shuffle(available_sents)
        current_sent_ids = available_sents[:5]

        ## Debug to make sure it's working
        if c < 5:
            print(current_sent_ids)

        row = []
        for i in current_sent_ids:
            current_hit = mturk_input[i][current_hit_id[i]] + [i]
            row += current_hit
            current_hit_id[i] += 1

            if c < 5:
                print(current_hit)

        if c < 5:
            print("\n\n")
        rows.append(row)
        c += 1
    
    print(len(rows))
    return rows


def output_csv(rows, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        firstrow = ['input1', 'sys11', 'sys12', 'sys13', 'sys14', 'sys15', 'sysid1', 'sentid1', \
                    'input2', 'sys21', 'sys22', 'sys23', 'sys24', 'sys25', 'sysid2', 'sentid2', \
                    'input3', 'sys31', 'sys32', 'sys33', 'sys34', 'sys35', 'sysid3', 'sentid3', \
                    'input4', 'sys41', 'sys42', 'sys43', 'sys44', 'sys45', 'sysid4', 'sentid4', \
                    'input5', 'sys51', 'sys52', 'sys53', 'sys54', 'sys55', 'sysid5', 'sentid5']
        csvwriter.writerow(firstrow)
        for row in rows:
            csvwriter.writerow([r.encode('ascii', 'ignore') for r in row])

def main(system_outputs_folder, clustered_outputs_folder, output_file):
    random.seed(37)
    inputs, preds, scores, systems = load_directory(system_outputs_folder, clustered_outputs_folder)
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

NOTE: NOT DONE!!!!
'''
