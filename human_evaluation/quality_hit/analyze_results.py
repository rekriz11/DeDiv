import sys
import csv
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

def read_results(mturk_file):
    mturk_dict = dict()
    with open(mturk_file, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f)
        headers = next(csvreader)
        
        for i, row in enumerate(csvreader):
            mturk_dict[i] = {key: value for key, value in zip(headers, row)}
    return mturk_dict

def calculate_results(mturk_dict, stat_name):
    stat = dict()
    prompts_actual = ["hi" for i in range(100)]

    for rowid, row_dict in mturk_dict.items():
        for i in range(1, 6):
            prompt = row_dict["Input.input" + str(i)]
            prompt_id = int(row_dict["Input.sentid" + str(i)])

            prompts_actual[prompt_id] = prompt

            for j in range(1, 6):
                system_name = row_dict["Input.sysid" + str(i) + str(j)][:-2]
                response_id = int(row_dict["Input.sysid" + str(i) + str(j)][-1])
                
                judgment = row_dict["Answer." + stat_name + str(i) + str(j)]
                if judgment == "Yes":
                    judgment = 1
                else:
                    judgment = 0

                try:
                    stat[system_name][prompt_id][response_id].append(judgment)
                except KeyError:
                    try:
                        stat[system_name][prompt_id][response_id] = [judgment]
                    except KeyError:
                        try:
                            stat[system_name][prompt_id] = dict()
                            stat[system_name][prompt_id][response_id] = [judgment]
                        except KeyError:
                            stat[system_name] = dict()
                            stat[system_name][prompt_id] = dict()
                            stat[system_name][prompt_id][response_id] = [judgment]

    ## Calculates inner-annotator agreement
    annotators = [[], [], []]
    for sys_name, prompts in stat.items():
        for prompt_id, responses in prompts.items():
            for response_id, judgments in responses.items():
                if len(judgments) == 3:
                    for i in range(len(judgments)):
                        annotators[i].append(judgments[i])

    if len(annotators[0]) > 0:
        print(len(annotators[0]))
        kappa1 = cohen_kappa_score(annotators[0], annotators[1])
        kappa2 = cohen_kappa_score(annotators[0], annotators[2])
        kappa3 = cohen_kappa_score(annotators[1], annotators[2])
        avg_kappa = (kappa1 + kappa2 + kappa3) / 3
        print("Inner-annotator agreement: " + str(round(avg_kappa, 3)))
        print()
    

    ## Calculates average disagreement
    diffs = []
    diff_dist = [0 for i in range(2)]
    for sys_name, prompts in stat.items():
        for prompt_id, responses in prompts.items():
            for response_id, judgments in responses.items():
                if len(judgments) > 1:
                    combs = combinations(list(range(len(judgments))), 2)
                    for c in combs:
                        diffs.append(abs(judgments[c[0]] - judgments[c[1]]))
                        diff_dist[abs(judgments[c[0]] - judgments[c[1]])] += 1

    try:
        print("Average disagreement: " + str(round(sum(diffs)/len(diffs), 3)))
    except ZeroDivisionError:
        a = 1

    print("Pairwise Agreement: ")
    try:
        print(diff_dist)
        print([round(d/sum(diff_dist), 3) for d in diff_dist])
        print()
    except ZeroDivisionError:
        a = 1

    systems = sorted(list(stat.keys()))

    ## Calculates average per sentence id
    done = [0 for i in range(len(systems))]
    not_done = [0 for i in range(len(systems))]
    avg_stat = [[] for i in range(len(systems))]
    for sys_name, prompts in stat.items():
        for prompt_id, responses in prompts.items():
            for response_id, judgments in responses.items():
                ## Keeps track of how many sentences are done
                if len('judgments') == 3:
                    done[systems.index(sys_name)] += 1
                else:
                    not_done[systems.index(sys_name)] += 1
                avg_stat[systems.index(sys_name)].append(sum(judgments)/len(judgments))

    ## Calculates overall average per sentence id
    avg_by_sent_id = [[] for i in range(100)]
    for sys_name, prompts in stat.items():
        for prompt_id, responses in prompts.items():
            for response_id, judgments in responses.items():
                avg_by_sent_id[prompt_id].append(sum(judgments)/len(judgments))
    print([len(a) for a in avg_by_sent_id])


    avg_sent_id = [sum(a)/len(a) for a in avg_by_sent_id]
    print(len(avg_sent_id))
                

    ## Calculates overall averages
    avg = [sum(s)/len(s) for s in avg_stat]

    #print("DONE: " + str(done))
    #print("NOT DONE: " + str(not_done))

    #print("\nAVERAGES:")
    #for i in range(len(systems)):
    #    print(systems[i] + ": " + str(avg[i]))
    return avg, systems, avg_sent_id, prompts_actual
            
                
            



def main(mturk_file):
    mturk_dict = read_results(mturk_file)

    print("\n## GRAMMAR ##")
    grammar, systems, grammarsi, prompts = calculate_results(mturk_dict, 'grammar')
    print("\n## COHERENCE ##")
    coherence, systems, coherencesi, prompts = calculate_results(mturk_dict, 'coherence')
    print("\n## INTERESTING ##")
    interesting, systems, interestingsi, prompts = calculate_results(mturk_dict, 'interesting')

    print("\n#### OVERALL STATS ####")
    for i in range(len(grammar)):
        group = systems[i].split("/")[0]
        if group == "clustered":
            cluster = "Yes"
        else:
            cluster = "No"
            
        system = systems[i].split("/")[1][:-5]
        print(system + "\t" + cluster + "\t" + str(round(grammar[i], 3)) + "\t" + \
              str(round(coherence[i], 3)) + "\t" + \
              str(round(interesting[i], 3)) + "\t" + \
              str(round(sum([grammar[i], coherence[i], interesting[i]])/3, 3)))


    print("\n#### Prompts by average ####")

    avg_sent_id = [(grammarsi[i] + coherencesi[i] + interestingsi[i]) / 3 for i in range(len(grammarsi))]

    indices = [i[0] for i in sorted(enumerate(avg_sent_id), key=lambda x:x[1])]
    for i in indices:
        print(str(round(avg_sent_id[i], 4)) + "\t" + prompts[i])
              

    
        
if __name__ == '__main__':
    mturk_file = 'results/Batch_3540427_batch_results.csv'
    main(mturk_file)

'''
python analyze_results_v2.py \
results/Batch_3540427_batch_results.csv
'''
