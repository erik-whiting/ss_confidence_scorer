import os



models = ['ContextFold', 'ContraFold', 'EternaFold',
          'IPKnot', 'MXFold', 'MXFold2', 'NeuralFold',
          'NUPACK', 'pKnots', 'RNAFold', 'RNAStructure',
          'Simfold', 'SPOT-RNA']


model_pred_acc_map = {}
for m in models:
    model_pred_acc_map[m] = {"pred_index": None, "score_index": None}


header = "dataset,datapoint,sequence,ContextFold_prediction,ContextFold_accuracy,ContraFold_prediction,ContraFold_accuracy,EternaFold_prediction,EternaFold_accuracy,IPKnot_prediction,IPKnot_accuracy,MXFold_prediction,MXFold_accuracy,MXFold2_prediction,MXFold2_accuracy,NeuralFold_prediction,NeuralFold_accuracy,NUPACK_prediction,NUPACK_accuracy,pKnots_prediction,pKnots_accuracy,RNAFold_prediction,RNAFold_accuracy,RNAStructure_prediction,RNAStructure_accuracy,Simfold_prediction,Simfold_accuracy,SPOT-RNA_prediction,SPOT-RNA_accuracy,sequence_length,gc_content,sequence_entropy,average_longest_gc_helix,mfe,ens_def,longest_consecutive_A,longest_consecutive_C,longest_consecutive_G,longest_consecutive_U,avg_gu_pairs,averge_bps_predicted,hairpin_count,helix_count,single_strand_count,junction_count,average_mway_count,longest_singlestrand,average_singlestrand_size,longest_helix,average_helix_size,average_hamming_distance_of_preds,average_au_pairs_of_helices,helices_with_reverse_compliment,rate_of_gt_4_unpaired_nt_in_hairpin"
header = header.split(",")
for m in models:
    pred_index = header.index(f"{m}_prediction")
    score_index = header.index(f"{m}_accuracy")
    model_pred_acc_map[m]["pred_index"] = pred_index
    model_pred_acc_map[m]["score_index"] = score_index


keepers = ["sequence_length", "gc_content", "sequence_entropy", "mfe", "ens_def"]
keeper_indices = []
for k in keepers:
    keeper_indices.append(header.index(k))

new_headers = ["dataset", "datapoint", "sequence", "model", "prediction", "score"] + keepers
def make_restructured_file(kind):
    original_file = f"chem_map_{kind}_master_v5.csv"
    with open(original_file) as fh:
        lines = fh.readlines()

    lines.pop(0) # Remove headers
    new_file = open(f"chem_map_all_{kind}_preds.csv", "w")

    new_file.write(",".join(new_headers) + "\n")
    for line in lines:
        line = line.split(",")
        ds = line[0]
        dp = line[1]
        seq = line[2]
        keeper_vals = ",".join([line[i] for i in keeper_indices]) + "\n"
        for m in models:
            pred = line[model_pred_acc_map[m]["pred_index"]]
            score = line[model_pred_acc_map[m]["score_index"]]
            new_line = f"{ds},{dp},{seq},{m},{pred},{score}," + keeper_vals
            new_file.write(new_line)
    new_file.close()

make_restructured_file("easy")
make_restructured_file("hard")
