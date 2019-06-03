import sys
import os
from os import listdir
from os.path import isfile, join
import json

## Gets name of files in a list from a directory
def get_all_files(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))
             and ".json" in f]
    return files

## Loads all files and their spice scores for each caption
def load_directory(dir1, dir2):
    files1 = get_all_files(dir1)
    paths1 = [dir1 + "/" + f for f in files1]

    files2 = get_all_files(dir2)
    paths2 = [dir2 + "/" + f for f in files2]

    paths = paths1 + paths2
    files = files1 + files2

    spice_dict = dict()
    for i, path in enumerate(paths):
        goods, bads = 0, 0
        if i in range(len(files1)):
            name = files[i]
        else:
            name = "pdc_" + files[i]
        spice_dict[name] = dict()
            
        with open(path) as f:
            outputs = json.load(f)
            for k,v in outputs["imgToEval"].items():
                try:
                    spice_score = v["SPICE"]["All"]["f"]
                    goods += 1
                except TypeError:
                    spice_score = v["SPICE"]
                    bads += 1
                spice_dict[name][k] = spice_score
        print(name)
        print(goods)
        print(bads)
        print()
    return spice_dict

## Outputs numbers to file in correct order
def output_to_files(spice_dict, output_folder):
    name = list(spice_dict.keys())[0]
    image_list = list(spice_dict[name].keys())

    for k, v in spice_dict.items():
        with open(output_folder + k, 'w', encoding='utf8') as f:
            for image in image_list:
                f.write(str(v[image]) + "\n")
                
                
        

def main(decodes_folder, pdc_folder, output_folder):
    spice_dict = load_directory(decodes_folder, pdc_folder)
    output_to_files(spice_dict, output_folder)
    
if __name__ == '__main__':
    decodes_folder = sys.argv[1]
    pdc_folder = sys.argv[2]
    output_folder = sys.argv[3]
    
    main(decodes_folder, pdc_folder, output_folder)



'''
python3 convert_caption_significance.py \
all_experiments/captioning/10decodes/caption_eval \
all_experiments/captioning/100to10decodes/caption_eval_oracle1 \
all_experiments/captioning/spice/
'''
