import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--tasks", type=str)
parser.add_argument("--circuits", type=str)
parser.add_argument("--id-map", type=str)
parser.add_argument("--out", type=str)

args = parser.parse_args()

task_file = args.tasks
circuits_dir = args.circuits
id_map_file = args.id_map
output_file = args.out

task_list = []
id_map = {}

with open(task_file) as f:
    data = json.load(f)
    task_dict = data["tasks"]

with open(id_map_file) as f:
    data = json.load(f)
    id_map["id"] = data["entries"]["qasm_file"]

predictions = []

for task in task_list:
    id = task["id"]
    processor = task["processor"]
    precision = task["precision"]

    file_path = circuits_dir + "/" + id_map[id]

    # run shit here with model
    predicted_threshold_min = 0
    predicted_forward_wall_s = 0
    predictions += {"id":id, "predicted_threshold_min": predicted_threshold_min, "predicted_forward_wall_s": predicted_forward_wall_s}


# write stuff to output file 
with open(output_file, 'w') as f: 
    json.dump(predictions, f, indent=4)