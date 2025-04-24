import json 
import sys
sys.path.append("../src")
from countdown import CountDown
import numpy as np

# Load DFS data
dfs_data_path = "/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs/val1_b4_t100_n500000_dfs.json"
with open(dfs_data_path, "r") as f:
    dfs_data = json.load(f)

cd = CountDown(100, 4)
solution_num = []
for i in range(200):
    nums = dfs_data[i]['nums']
    target = dfs_data[i]['target']
    all_solutions = cd.generate_all(target, nums)
    print(f"Target: {target}, Numbers: {nums}")
    print(f"All Solutions: {all_solutions}")
    solution_num.append(len(all_solutions))

print(f"Average number of solutions: {sum(solution_num)/len(solution_num)}")
print(f"Max number of solutions: {max(solution_num)}")
print(f"Min number of solutions: {min(solution_num)}")

#  save as a numpy array
np.save("val1_b4_t100_n500000_dfs.json_solution_num.npy", np.array(solution_num))
print("Saved solution_num.npy")

