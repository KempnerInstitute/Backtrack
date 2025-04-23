import json
import os

output_dir = '/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs/oft-strat'
file_names = [
    'results_sudoku_data_sample.json_100_0_ckpt160000_@32',
    'results_sudoku_data_sample.json_100_100_ckpt160000_@32'
]

# load two json files 
def load_results(files):
    acc = []
    model_output = []
    for f in files:
        print(f'loading results from {f}')
        f = os.path.join(output_dir, f)
        with open(f, 'r') as f:
            data = json.load(f)
        print(len(data['ratings']), len(data['model_output_gen']))
        acc.extend(data['ratings'])
        model_output.append(data['model_output_gen'])

    return acc, model_output


acc, model_output = load_results(file_names)
parta, partb = model_output
# stack two parts in the second dimension
model_output = []
for a, b in zip(parta, partb):
    model_output.append(a + b)
print(len(acc), len(model_output), len(model_output[0]))

# save as a new file
new_file = 'results_sudoku_data_sample.json_200_0_ckpt160000_@32'
with open(os.path.join(output_dir, new_file), 'w') as f:
    json.dump({
        "ratings": acc,
        "model_output_gen": model_output
    }, f)
print('saved as ', new_file)