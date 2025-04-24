'''
CountDown class for generating questions and trees
'''
import random
import itertools

import tiktoken
import re
import ast

from countdown_utils import combine_nums, sum_heuristic, mult_heuristic, great_prune, mult_prune

class CountDown(object):
    def __init__(self, max_target=24, start_size=4, min_target=10):
        self.max_target = max_target
        self.min_target = min_target
        self.start_size = start_size
    
    def generate(self, target):
        if target > self.max_target:
            raise ValueError("Target cannot be greater than max target")
        if target < self.min_target:
            raise ValueError("Target cannot be less than min target")
        
        found = False
        while not found:
            # nums in question can go up to max target
            nums = [random.randint(1, self.max_target-1) for _ in range(self.start_size)]
            solution = self.search(target, nums)
            if solution is not None:
                found = True
        return nums, solution

    def search(self, target, nums, operations=[]):
        # Navigate the entire solution tree, implemented with DFS
        if len(nums) == 1:
            if nums[0] == target:
                return operations
            else:
                return None

        for i, j in itertools.combinations(range(len(nums)), 2):
            num1, num2 = nums[i], nums[j]
            remaining_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
            for result, operation in combine_nums(num1, num2):
                new_nums = remaining_nums + [result]
                new_operations = operations + [operation]
                solution = self.search(target, new_nums, new_operations)
                if solution is not None:
                    return solution
        return None
    
    def search_all(self, target, nums, solutions, operations=[]):
        # Navigate the entire solution tree, implemented with DFS
        if len(nums) == 1:
            if nums[0] == target:
                return operations
            else:
                return None

        for i, j in itertools.combinations(range(len(nums)), 2):
            num1, num2 = nums[i], nums[j]
            remaining_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
            for result, operation in combine_nums(num1, num2):
                new_nums = remaining_nums + [result]
                new_operations = operations + [operation]
                solution = self.search_all(target, new_nums, solutions, new_operations)
                if solution is not None:
                    # print(solution)
                    solutions.append(solution)
                    # return solution
        return None
    
    def convert_to_path(self, target, nums, operations):
        # convert solution to readable path

        operations_explored = []
        search_trace = ""
        search_trace += f"Current State: {target}:{nums}, Operations: {operations_explored}\n"
        node_index = 1
        for operation in operations:
            # split at operation +, -, *, /
            if "+" in operation:
                i, j = operation.split("=")[0].split("+")
                i, j = int(i), int(j)
                result = i + j
            elif "-" in operation:
                i, j = operation.split("=")[0].split("-")
                i, j = int(i), int(j)
                result = i - j
            elif "*" in operation:
                i, j = operation.split("=")[0].split("*")
                i, j = int(i), int(j)
                result = i * j
            elif "/" in operation:
                i, j = operation.split("=")[0].split("/")
                i, j = int(i), int(j)
                result = i / j

            result = int(result)
            # bugged
            # new_nums = [int(nums[k]) for k in range(len(nums)) if nums[k] != i and nums[k] != j] + [result]
            # fixed 
            new_nums = [int(nums[k]) for k in range(len(nums))]
            new_nums.remove(i)
            new_nums.remove(j)
            new_nums.append(result)
            
            nums = new_nums
            search_trace += f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
            if len(nums) == 1:
                search_trace += f"{nums[0]},{target} equal: Goal Reached\n"
            else:
                node_index += 1
                search_trace += f"Generated Node #{node_index}: {new_nums} from Operation: {operation}\n"
                operations_explored.append(operation)
                search_trace += f"Current State: {target}:{nums}, Operations: {operations_explored}\n"
        return search_trace
    
    def extract_last_node(self, target, num, s):
        trace = self.convert_to_path(target, num, s)
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
        
    def generate_all_hints(self, target, nums):
        solutions = []
        self.search_all(target, nums, solutions)
        hints = []
        for s in solutions:
            last_node = self.extract_last_node(target, nums, s)
            if last_node is not None:
                hints.append(last_node)
        hints = list(set(hints))
        return hints
    
    def generate_all(self, target, nums):
        solutions = []
        self.search_all(target, nums, solutions)
        return solutions
    

if __name__ == "__main__":
    # Example usage
    countdown = CountDown(50, 4)
    target = 39
    nums, solution = countdown.generate(target)
    print(f"Numbers: {nums}")
    print(f"Solution: {solution}")
    search_trace = countdown.convert_to_path(target, nums, solution)
    print(search_trace)

    # all solutions
    all_solutions = countdown.generate_all(target, nums)
    print(f"All Solutions: {all_solutions}")
    # search_path = a_star_search(target, nums, heuristic=sum_heuristic, should_prune=great_prune)
    # mult_path = a_star_search(target, nums, heuristic=mult_heuristic, should_prune=mult_prune)

    # # print(search_path)
    # # parsed_path = check_solution_path(search_path)
    # # add_score = metric_fn(search_path)
    # # print(parsed_path)
    # print(f"solution length: {len(search_path)}")
    # enc = tiktoken.get_encoding("cl100k_base")
    # tokens = enc.encode(search_path)
    # print(f"token length: {len(tokens)}")

    # search_path = mult_path
    # # print(search_path)
    # parsed_path = check_solution_path(search_path)
    # mult_score = metric_fn(search_path)
    # # print(parsed_path)
    # print(f"solution length: {len(search_path)}")
    # enc = tiktoken.get_encoding("cl100k_base")
    # tokens = enc.encode(search_path)
    # print(f"token length: {len(tokens)}")
    # print(f"add score: {add_score}")
    # print(f"mult score: {mult_score}")