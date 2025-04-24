from countdown import CountDown
import random
import re
import ast


def extract_last_node(target, num, s):
    trace = cd.convert_to_path(target, num, s)
    trace_list = trace.strip().split("\n")
    if "Goal Reached" not in trace_list[-1]:
        return None
    last_node = trace_list[-3]
    try:
        last_node = re.search(r'\[(.*?)\]', last_node).group(0)
    except:
        print("ERROR: ")
        print(s)
        print(trace)
    
    extracted_list = ast.literal_eval(last_node)  # Safely evaluates "[54, 1]" to [54, 1]
    return tuple(sorted(extracted_list))


max_target = 100
start_size = 4

cd = CountDown(max_target, start_size)
target_nums = [i for i in range(10, max_target+1)]
for _ in range(500):
    target = random.choice(target_nums)
    nums, operations = cd.generate(target)
    print(target, nums, operations)
    solutions = []
    cd.search_all(target, nums, solutions)

    hints = []
    for s in solutions:
        # no_backtrack_trace = cd.convert_to_path(target, nums, s)
        last_node = extract_last_node(target, nums, s)
        hints.append(last_node)
    hints = list(set(hints))
    print(hints)

# for _ in range(10):
#     print("-----"*5)
#     target = random.choice(target_nums)
#     nums, solution = cd.generate(target)
#     print(nums, solution)
#     no_backtrack_trace = cd.convert_to_path(target, nums, solution)
#     print("trace:")
#     print(no_backtrack_trace)