# to download the dataset from Kaggle
import pandas as pd
import numpy as np
from dokusan.boards import BoxSize, Sudoku
from solver import print_board, generate_full_trace_backtrack_solution, generate_strategy_trace_solution, convert_full_trace_shortcut_solution, check_sudoku_solution
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import json


def download_sudoki_3m():
    import kagglehub
    # Download latest version
    path = kagglehub.dataset_download("radcliffe/3-million-sudoku-puzzles-with-ratings")
    print("Path to dataset files:", path)

def load_sudoku_data(path='/n/home05/sqin/self-correct/sudoku/data/sudoku-3m.csv'):
    data = pd.read_csv(path)
    return data

def view_grid(puzzle_string):
    return pd.DataFrame(np.array(list(puzzle_string.replace('.', ' '))).reshape((9, 9)))

def convert_to_sudoku(puzzle_string):
    sudoku_array = np.array(list(puzzle_string.replace('.', '0')), dtype=int).reshape((9, 9))
    sudoku = Sudoku.from_list(sudoku_array, BoxSize(3, 3))
    return sudoku

def generate_data(data, index):
    puzzle = data.puzzle[index]
    sudoku_puzzle = convert_to_sudoku(puzzle)
    solved_sudoku, full_trace_solution = generate_full_trace_backtrack_solution(sudoku_puzzle)
    board = print_board(sudoku_puzzle)
    # convert the full trace solution to a shortcut solution
    shortcut_solution = convert_full_trace_shortcut_solution(full_trace_solution)
    check_solution = check_sudoku_solution(board, shortcut_solution)
    if not check_solution:
        print(f"Failed to solve sudoku {index}")
        return None
    else:
        return board, full_trace_solution, shortcut_solution
    
if __name__ == "__main__":
    from shortern_sudoku_data import make_board_easier
    data = load_sudoku_data()

    # take two arguments from comman to specify the range of sudoku to process
    import sys
    i_start = int(sys.argv[1])
    i_end = int(sys.argv[2])
    
    results = []
    strategy_results = []
    strategy_failure = 0
    for i in range(i_start, i_end):
        if i % 1000 == 0:
            print(f"Processing sudoku {i}")
        puzzle = data.puzzle[i]
        solution = data.solution[i]    
        difficulty = data.difficulty[i]
        ''' option - make the board easier '''
        puzzle = make_board_easier(puzzle, solution)

        
        sudoku_puzzle = convert_to_sudoku(puzzle)
        
        # # try solve the sudoku by simple backtracking strategy 
        # solved_sudoku, full_trace_solution = generate_full_trace_backtrack_solution(sudoku_puzzle)
        
        board = print_board(sudoku_puzzle)

        # # convert the full trace solution to a shortcut solution
        # shortcut_solution = convert_full_trace_shortcut_solution(full_trace_solution)
        # check_solution = check_sudoku_solution(board, shortcut_solution)

        # # try solve the sudoku by strategy solution
        # assert not sudoku_puzzle.is_solved()
        
        # old - without backtrack
        # strategy_solution = generate_strategy_trace_solution(sudoku_puzzle)
        # check_solution_strategy = check_sudoku_solution(board, strategy_solution)
        # new - with backtrack
       
        strategy_solved_sudoku, strategy_full_trace_solution = generate_strategy_trace_solution(sudoku_puzzle)

        if len(strategy_full_trace_solution) > 0:
            strategy_shortcut_solution = convert_full_trace_shortcut_solution(strategy_full_trace_solution)
            check_solution_strategy = check_sudoku_solution(board, strategy_shortcut_solution)
        else:
            check_solution_strategy = False
        
        if not check_solution_strategy: # if strategy solution failed, use naive solution
            try:
                strategy_solved_sudoku, strategy_full_trace_solution = generate_full_trace_backtrack_solution(sudoku_puzzle) # just use naive solution
                strategy_failure += 1
            except:
                print(f"Warning: failed to solve sudoku by naive solution {i}")
                continue # just ignore this sudoku
        
        strategy_shortcut_solution = convert_full_trace_shortcut_solution(strategy_full_trace_solution)
        check_final_solution = check_sudoku_solution(board, strategy_shortcut_solution)
        
        # if not check_solution:
        if not check_final_solution:
            print(f"Failed to solve sudoku {i}")
        else:
            # full_trace_solution = " \n".join([str(x) for x in full_trace_solution])
            # shortcut_solution = " \n".join([str(x) for x in shortcut_solution])
            
            strategy_full_trace_solution = " \n".join([str(x) for x in strategy_full_trace_solution])
            strategy_shortcut_solution = " \n".join([str(x) for x in strategy_shortcut_solution])
            board = " \n".join([str(x) for x in board])

            # results.append(
            #     {
            #         "id": i,
            #         "difficulty": difficulty,
            #         "puzzle": puzzle,
            #         "board": board,
            #         "full_trace_solution": full_trace_solution,
            #         "shortcut_solution": shortcut_solution,
            #         # "strategy_solution": strategy_solution,
            #         # "strategy_success": int(check_solution_strategy),
            #     }
            # )

            strategy_results.append(
                {
                    "id": i,
                    "difficulty": difficulty,
                    "puzzle": puzzle,
                    "board": board,
                    "strategy_full_trace_solution": strategy_full_trace_solution,
                    "strategy_shortcut_solution": strategy_shortcut_solution,
                    "strategy_success": int(check_solution_strategy),
                }
            )
    
    # report the number of strategy failure
    print(f"Number of strategy failure: {strategy_failure} out of {i_end - i_start}")

    ''' write results to json file '''
    # with open(f"/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/data/sudoku_data_{i_start}_{i_end}.json", "w") as f:
    #     json.dump(results, f, indent=4, separators=(",", ": "))

    ''' write strategy results to json file '''
    # with open(f"/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_fixed/sudoku_strategy_data_{i_start}_{i_end}.json", "w") as f:
    #     json.dump(strategy_results, f, indent=4, separators=(",", ": "))

    ''' write easy strategy results to json file '''
    with open(f"/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_easy/easy_sudoku_strategy_data_{i_start}_{i_end}.json", "w") as f:
        json.dump(strategy_results, f, indent=4, separators=(",", ": "))








    ''' not so successful attemps to parallelize the data generation and writing to json file '''
    # num_iterations = 5_000
    # generate_data_fn = partial(generate_data, data)
    # def generate_data_in_parallel(start, end):
    #     return [generate_data_fn(i) for i in range(start, end)]
    
    # def parallel_write_json():
    #     # num_processes = cpu_count()  # Use all available CPU cores
    #     num_processes = 4
    #     print(f"Using {num_processes} processes")
    #     chunk_size = num_iterations // num_processes
    #     print(f"Chunk size: {chunk_size}")

    #     # Use a multiprocessing pool to parallelize data generation
    #     with Pool(processes=num_processes) as pool:
    #         # Create tasks for each chunk
    #         ranges = [(i, min(i + chunk_size, num_iterations)) for i in range(0, num_iterations, chunk_size)]
    #         results = pool.starmap(generate_data_in_parallel, ranges)

    # # Run the parallelized generation and writing
    # # time the process
    # start_time = time.time()
    # parallel_write_json()
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time:.2f} seconds")
