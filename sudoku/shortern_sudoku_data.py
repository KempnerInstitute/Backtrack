from data import load_sudoku_data, convert_to_sudoku
from solver import s_eliminate_strategy, print_board
import numpy as np
import random
import re



def add_random_entries(partial_list, full_list, num_entries=4):
    # Convert lists to sets for fast lookup
    partial_set = set(partial_list)
    full_set = set(full_list)
    
    # Find missing entries
    missing_entries = list(full_set - partial_set)
    
    # Select random entries (up to num_entries)
    selected_entries = random.sample(missing_entries, min(num_entries, len(missing_entries)))
    
    # Append to the first list
    updated_list = partial_list + selected_entries
    
    return updated_list

def generate_grid_string(entries):
    # Initialize a 9x9 grid with zeros
    grid = [[0] * 9 for _ in range(9)]
    
    # Parse the list and fill in the grid
    for entry in entries:
        match = re.match(r"\((\d+), (\d+)\) = (\d+)", entry)
        if match:
            x, y, value = map(int, match.groups())
            grid[x][y] = value

    # Flatten the grid into a string
    grid_string = ''.join(str(num) for row in grid for num in row)
    
    return grid_string

def make_board_easier(puzzle, solution):
    sudoku_puzzle = convert_to_sudoku(puzzle)
    board = print_board(sudoku_puzzle)
    solution = print_board(convert_to_sudoku(solution))

    trace = []
    _, trace = s_eliminate_strategy(sudoku_puzzle, trace)

    if len(trace) > 10:
        new_puzzle = board[1: -1]  + trace[:10] 
    else: # fill more cells from solution 
        board_content = board[1: -1] + trace 
        solution_content = solution[1: -1]
        new_puzzle = add_random_entries(board_content, solution_content, 10 - len(trace))

    # convert back to string
    new_puzzle = generate_grid_string(new_puzzle)


    return new_puzzle

if __name__ == "__main__":
    data = load_sudoku_data()
    puzzle = data.puzzle[0]
    solution = data.solution[0]
    new_puzzle = make_board_easier(puzzle, solution)
    print(new_puzzle)
    print("recover: ", print_board(convert_to_sudoku(new_puzzle)))
