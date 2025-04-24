import re



def simple_rating(search_path):
    # Simple rating function based on number of operations
    nodes_explored = search_path.count("Exploring Operation") + 1
    return nodes_explored

def parse_trajectory_deep(search_path, part):
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]
    if "<|endoftext|>" in search_path:
      search_path = search_path.replace("<|endoftext|>", "")

    target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
    if not target_nums_match:
        return "Invalid input: Cannot find the initial state in the first line."

    if part == "first":
        _, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
        target = nums[4]
        nums = nums[:4]
    elif part == "second":
        target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
        nums = nums[4:]
    else:
        raise ValueError(f"Invalid part: {part}")
    
    # Extract the operations from the line that claims the goal is reached.
    if part == "first":
        goal_lines = re.finditer(r"\d+,\d+ equal", search_path)
    else:
        goal_lines = re.finditer(r"\d+,\d+ equal: Goal Reached", search_path)
    goal_lines = list(goal_lines)

    if not goal_lines:
        return "No goal reached statement found."

    goal_line = goal_lines[0]
    # get the last operation line before the goal reached statement
    operations = re.findall(r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]",
                            search_path[:goal_line.start()])
    if not operations:
        return "No operations found leading to the goal."

    final_operation = operations[-1][0]
    try:
        predicted_result = int(final_operation.split('=')[1])
    except:
        print("couldnt parse last op", final_operation)
        return "Couldnt parse last op"
    if predicted_result != target:
        return "Invalid path: Final operation does not result in target."

    # get the last current state, operations before the goal reached statement, and extract the operations
    operation_list = re.findall(r"Current State: \d+:\[.*?\], Operations: \[(.*?)\]", search_path[:goal_line.start()])[
        -1].split(',')
    operation_list = [op.replace("'", "").strip() for op in operation_list]
    operation_list += [final_operation]

    # Verify each operation and keep track of the numbers involved
    available_numbers = nums
    for operation in operation_list:
        # Verify the operation
        try:
            left, right = operation.split('=')
        except:
            return f"Could not split operation into lhs, rhs"
        try:
            if eval(left) != int(right):
                return f"Invalid operation: {operation}"
        except Exception as e:
            return f"Error in evaluating operation {operation}: {e}"
        # get the numbers involved
        used_numbers = re.findall(r"\d+", left)
        for n in used_numbers:
            if int(n) not in available_numbers:
                return f"Invalid operation: {operation}, number {n} not available in {available_numbers}"

        available_numbers = [n for n in available_numbers if n not in used_numbers]
        available_numbers.append(int(right))

    return "Valid path."


def parse_trajectory_deep_complete(search_path):
    first_part = parse_trajectory_deep(search_path, "first")
    second_part = parse_trajectory_deep(search_path, "second")
    if first_part == "Valid path." and second_part == "Valid path.":
        return "Valid path."
    else:
        return f"Invalid path: \n\tfirst part : {first_part} \n\t second part :{second_part}"
    
def metric_fn_deep(search_path, mode="dt"):
    rating = parse_trajectory_deep_complete(search_path)
    if rating == "Valid path.":
        score = simple_rating(search_path)
        first_line = search_path.strip().split('\n')[0]
        if "->" in first_line:
            first_line = first_line.split("->")[1]
        target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
        target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
        if len(nums) == 7:
            max_nodes = 0
            # 3c2 x ops (4) x 2c2 x ops (4) = 48
            max_nodes += 48
            # 4c2 x ops (4) x 3c2 x ops (4) x 2c2 x ops (4) = 1152
            max_nodes += 1152
        elif len(nums) == 8:
            max_nodes = 1152*2
        return (max(1. - score / max_nodes, 0.0), rating)
    return (0., rating)