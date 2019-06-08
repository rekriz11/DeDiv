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
    for i in range(len(listy)):
        if listy[i] == "i":
            listy[i] = "I"            
            
    detok = detokenize(listy)
    fixed = str(detok)

    num_fixes = 0

    starts = ["i", "you", "he", "they", "we", "I", "You", "He", "They", "We"]
    punctuation = ["!", "?", "."]

    for s in starts:
        for p in punctuation:
            bad1 = s + p + " l"
            fixed = fixed.replace(bad1, s + "'ll")
            
            bad2 = s + p + " e"
            fixed = fixed.replace(bad2, s + "'ve")

    fixed = fixed.replace("'r e", "'re")
    fixed = fixed.replace(" a? ", " ")

    if fixed[-1] not in [".", "!", "?"]:
        fixed = fixed + "."

    

    if fixed != detok:
        return fixed.capitalize(), 1
    else:
        return fixed.capitalize(), 0
    

# Load all json files
def load_directory(dir1, dir2, dir3, detokenize):    
    files1 = get_all_files(dir1)
    paths1 = [dir1 + "/" + file for file in files1]

    files2 = get_all_files(dir2)
    paths2 = [dir2 + "/" + file for file in files2]

    files3 = get_all_files(dir3)
    paths3 = [dir3 + "/" + file for file in files3]

    files1 = ["original10/" + f for f in files1]
    files2 = ["clustered/" + f for f in files2]
    files3 = ["original100/" + f for f in files3]

    filepaths = paths1 + paths2 + paths3
    files = files1 + files2 + files3

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


def make_rows(inputs, preds, scores, systems, gold_dict):
    mturk_input = [[] for i in range(len(inputs[0]))]
    
    for j in range(len(inputs[0])):
        print("PROMPT: " + str(j))
        
        input_current = inputs[0][j]
        preds_current = []
        systems_current = []

        for i in range(len(preds)):
            preds_current += preds[i][j]
            systems_current += systems[i][j]

        random_inds = [k for k in range(len(preds_current))]
        random.shuffle(random_inds)

        shuffled = False
        while not shuffled:
            task_temp = []
            cur_start = 0
            while cur_start < len(random_inds) - 1:
                hit = random_inds[cur_start:cur_start+5]

                ## Checks to make sure all prompts in a set of 5 are unique
                if len(list(set([preds_current[k] for k in hit]))) == 5: 
                    inputy = [[input_current], [preds_current[k] for k in hit], [systems_current[k] for k in hit]]
                    task_temp.append(inputy)
                    cur_start += 5
                ## Otherwise, shuffle and try again
                else:
                    '''
                    print([preds_current[k] for k in hit])
                    print()
                    '''
                    random.shuffle(random_inds)
                    break

            if cur_start >= len(random_inds) - 1:
                mturk_input[j] = task_temp
                shuffled = True

    print("NUMBER OF TASKS: ")
    print(len(mturk_input))
    print(sum([len(m) for m in mturk_input]))

    rows = []
    current_hit_id = [0 for i in range(len(mturk_input))]

    c = 0
    while min(current_hit_id) < len(mturk_input[0]):
        available_sents = [i for i in range(len(current_hit_id)) \
                             if current_hit_id[i] == min(current_hit_id)]
        
        random.shuffle(available_sents)
        current_sent_ids = available_sents[:2]

        row = []
        for index, i in enumerate(current_sent_ids):
            if index == 0:
                ## Do not include a control
                current_hit = flatten(mturk_input[i][current_hit_id[i]]) + [i]
            else:
                ## Gets controls
                hit = mturk_input[i][current_hit_id[i]]
                control = gold_dict[hit[0][0]]
                control_ind = random.randint(0, 5)

                ## Inserts control into hit
                new_preds, new_systems = [], []
                for j in range(len(hit[1])):
                    if control_ind == j:
                        new_preds.append(control)
                        new_systems.append("CONTROL")

                    new_preds.append(hit[1][j])
                    new_systems.append(hit[2][j])

                ## Adds control at the end if necessary
                if control_ind == len(hit[1]):
                    new_preds.append(control)
                    new_systems.append("CONTROL")
                    
                '''
                print(new_preds)
                print(control_ind)
                print(hit[0][0])
                print(gold_dict[hit[0][0]])
                print(row)
                a = b
                '''
                

                current_hit = hit[0] + new_preds + new_systems + [i]
                

            row += current_hit
            current_hit_id[i] += 1
                

        rows.append(row)
        c += 1

    print("Number of HITs:")
    print(len(rows))
    return rows

## Gets gold responses
def get_gold_responses(input_file, gold_output_file, detokenize):
    inputs, outputs = [], []
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            detok, b = fix(line.strip().split(" "), detokenize)
            inputs.append(detok)

    with open(gold_output_file, 'r', encoding='utf8') as f:
        for line in f:
            detok, b = fix(line.strip().split(" "), detokenize)
            outputs.append(detok)

    gold_dict = dict()
    for i, inp in enumerate(inputs):
        gold_dict[inp] = outputs[i]
        
    return gold_dict


def output_csv(rows, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        firstrow = ['input1', 'sys11', 'sys12', 'sys13', 'sys14', 'sys15', 'sysid11', 'sysid12', 'sysid13', 'sysid14', 'sysid15', 'sentid1', \
                    'input2', 'sys21', 'sys22', 'sys23', 'sys24', 'sys25', 'sys26', 'sysid21', 'sysid22', 'sysid23', 'sysid24', 'sysid25', 'sysid26', 'sentid2']
        csvwriter.writerow(firstrow)
        for row in rows:
            row_fixed = []
            for r in row:
                try:
                    row_fixed.append(r.encode('ascii', 'ignore').decode('ascii'))
                except AttributeError:
                    row_fixed.append(r)
            csvwriter.writerow(row_fixed)

def main(system_outputs_folder, clustered_outputs_folder, outputs_folder_100, input_file, gold_output_file, output_file):
    random.seed(37)
    detokenize = MosesDetokenizer('en')
    ## Gets predicted responses from all systems
    inputs, preds, scores, systems = load_directory(system_outputs_folder, clustered_outputs_folder, outputs_folder_100, detokenize)

    ## Gets gold responses
    gold_dict = get_gold_responses(input_file, gold_output_file, detokenize)

    ## Formats responses into rows for MTurk experiment
    rows = make_rows(inputs, preds, scores, systems, gold_dict)

    output_csv(rows, output_file)
    
    

if __name__ == '__main__':
    system_outputs_folder = sys.argv[1]
    clustered_outputs_folder = sys.argv[2]
    outputs_folder_100 = sys.argv[3]
    input_file = sys.argv[4]
    gold_output_file = sys.argv[5]
    output_file = sys.argv[6]
    
    main(system_outputs_folder, clustered_outputs_folder, outputs_folder_100, input_file, gold_output_file, output_file)


'''
python3 human_evaluation/quality_hit/format_input_camera_ready.py \
all_experiments/dialog/10decodes/ \
all_experiments/dialog/100to10decodes_withClustering/ \
all_experiments/dialog/100to10decodes_withTopScores/ \
eval_data/CMDB_prompt_subset.txt \
eval_data/CMDB_prompt_subset_responses.txt \
human_evaluation/quality_hit/input/input_camera_ready.csv
'''
