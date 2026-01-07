import pickle
import pdb
align_dict = {}
with open('match.txt', 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        align_dict[line[1]] = line[0]

meta_prox_graph = 'examplebook_pred_prox_graph'
aligned_prox_graph = []

with open(f"labels_p_labelo.txt", 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        if line[0] in align_dict:
            line[0] = align_dict[line[0]]
        if line[2] in align_dict:
            line[2] = align_dict[line[2]]
        
        aligned_prox_graph.append(line)
# print(aligned_prox_graph)

pickle.dump(aligned_prox_graph, open(f"{meta_prox_graph}_matched.pickle", 'wb'))


file = open(f'{meta_prox_graph}_matched.pickle', 'rb')
info = pickle.load(file)
print(info)
