import operator
from dokusan import generators, solvers, renderers
from dokusan import exceptions, techniques
from dokusan.boards import BoxSize, Sudoku
from dokusan.boards import Cell, Sudoku
from typing import List
import numpy as np
import re 
from collections import defaultdict

def print_cell_updates(cells: List[Cell], trace: List) -> List:
    for cell in cells:
        if cell.value is not None:
            # print(f"Update: {cell.position[:2]} -> {cell.value}")
            trace.append(f"{cell.position[:2]} = {cell.value}") # update trace with backtrack
            # shortcut.append(f"Update: {cell.position[:2]} -> {cell.value}") # update shortcut trace
    return trace

def print_event_updates(event: str, trace: List) -> List:
    # print(event)
    trace.append(event.lower()) # full trace records the whole thing
    # deal with shortcut trace
    # if "Guess" in event:
    #     # shortcut.append(event) -> no need to record the guess (there is a cell update)
    #     last_guess.append(len(shortcut)) # record the index of the last guess
    # elif "Revert" in event:
    #     last_guess_index = last_guess.pop()
    #     print("-> Revert last guess: ", shortcut[last_guess_index])
    #     shortcut = shortcut[:last_guess_index] # remove the last guess
    #     print("-> Revert to step:", shortcut[-1])
    # elif "Invalid" in event:
    #     pass # do nothing
    # elif "No Candidates" in event:
    #     pass # do nothing
    # elif "Solved" in event:
    #     shortcut.append(event)
    # else:
    #     raise ValueError(f"Unknown event: {event}")
    return trace

def s_eliminate(sudoku: Sudoku, trace: List) -> Sudoku:
    _sudoku = Sudoku(*sudoku.cells(), box_size=sudoku.box_size)

    all_techniques = (
        techniques.LoneSingle,
        # techniques.HiddenSingle,
    )

    for step in techniques.BulkPencilMarking(_sudoku):
        trace = print_cell_updates(step.changes, trace)
        _sudoku.update(step.changes)
        
    has_result = True
    while has_result:
        for technique in all_techniques:
            has_result = False
            for step in technique(_sudoku):
                trace = print_cell_updates(step.changes, trace)
                _sudoku.update(step.changes)
                has_result = True
    return _sudoku, trace

def s_eliminate_strategy(sudoku: Sudoku, trace: List) -> Sudoku:
    _sudoku = Sudoku(*sudoku.cells(), box_size=sudoku.box_size)

    all_techniques = (
        techniques.LoneSingle,
        techniques.HiddenSingle,
        techniques.NakedPair,
        techniques.NakedTriplet,
        techniques.LockedCandidate,
        techniques.XYWing,
        techniques.UniqueRectangle,
    )

    for step in techniques.BulkPencilMarking(_sudoku):
        trace = print_cell_updates(step.changes, trace)
        _sudoku.update(step.changes)
        
    has_result = True
    while has_result:
        for technique in all_techniques:
            has_result = False
            for step in technique(_sudoku):
                trace = print_cell_updates(step.changes, trace)
                _sudoku.update(step.changes)
                has_result = True
    return _sudoku, trace


def s_backtrack(sudoku: Sudoku, trace: List=[]) -> Sudoku:
    _sudoku, trace = s_eliminate(sudoku, trace)
    
    cells = sorted(
        (cell for cell in _sudoku.cells() if not cell.value),
        key=operator.attrgetter("candidates"),
    )

    for cell in cells:
        for candidate in sorted(cell.candidates):            
            trace = print_event_updates(f"guess: {cell.position[:2]} {sorted(cell.candidates)} = {candidate}", trace)
            trace = print_cell_updates([Cell(position=cell.position, value=candidate)], trace)
            _sudoku.update([Cell(position=cell.position, value=candidate)]) # try fill a value
            try:
                _sudoku, trace = s_backtrack(_sudoku, trace) # see if the board can be solved 
                return _sudoku, trace
            except exceptions.InvalidSudoku:
                trace = print_event_updates(f"invalid", trace)
                pass
            except exceptions.NoCandidates:
                trace = print_event_updates(f"nocand: {cell.position[:2]}", trace)
                pass

            trace = print_event_updates(f"revert: {cell.position[:2]} {sorted(cell.candidates)} = {cell.value}", trace)
            trace = print_cell_updates([cell], trace) 
            _sudoku.update([cell])
            
        else:
            if len(cell.candidates) == 0:
                trace = print_event_updates(f"nocand: {cell.position[:2]}", trace)

                # trace = print_event_updates(f"No value for cells: {failed_cells}", trace)
            raise exceptions.NoCandidates

    return _sudoku, trace


def s_backtrack_strategy(sudoku: Sudoku, trace: List=[]) -> Sudoku:
    _sudoku, trace = s_eliminate_strategy(sudoku, trace)
    
    cells = sorted(
        (cell for cell in _sudoku.cells() if not cell.value),
        key=operator.attrgetter("candidates"),
    )

    for cell in cells:
        for candidate in sorted(cell.candidates):            
            trace = print_event_updates(f"guess: {cell.position[:2]} {sorted(cell.candidates)} = {candidate}", trace)
            trace = print_cell_updates([Cell(position=cell.position, value=candidate)], trace)
            _sudoku.update([Cell(position=cell.position, value=candidate)]) # try fill a value
            try:
                _sudoku, trace = s_backtrack_strategy(_sudoku, trace) # see if the board can be solved 
                return _sudoku, trace
            except exceptions.InvalidSudoku:
                trace = print_event_updates(f"invalid", trace)
                pass
            except exceptions.NoCandidates:
                trace = print_event_updates(f"nocand: {cell.position[:2]}", trace)
                pass

            trace = print_event_updates(f"revert: {cell.position[:2]} {sorted(cell.candidates)} = {cell.value}", trace)
            trace = print_cell_updates([cell], trace) 
            _sudoku.update([cell])
            
        else:
            if len(cell.candidates) == 0:
                trace = print_event_updates(f"nocand: {cell.position[:2]}", trace)

                # trace = print_event_updates(f"No value for cells: {failed_cells}", trace)
            raise exceptions.NoCandidates
            

    return _sudoku, trace


def s_backtrack_stubborn(sudoku: Sudoku, trace: List=[]) -> Sudoku:
    '''' not working idk... '''
    _sudoku, trace = s_eliminate_strategy(sudoku, trace)
    
    cells = sorted(
        (cell for cell in _sudoku.cells() if not cell.value),
        key=operator.attrgetter("candidates"),
    )

    for cell in cells:
        for candidate in sorted(cell.candidates):            
            trace = print_event_updates(f"guess: {cell.position[:2]} {sorted(cell.candidates)} = {candidate}", trace)
            trace = print_cell_updates([Cell(position=cell.position, value=candidate)], trace)
            _sudoku.update([Cell(position=cell.position, value=candidate)]) # try fill a value
            try:
                _sudoku, trace = s_backtrack_stubborn(_sudoku, trace) # see if the board can be solved 
                return _sudoku, trace
            except exceptions.InvalidSudoku:
                trace = print_event_updates(f"invalid", trace)
                # no candidates, but we will keep trying anyways
                pass
            except exceptions.NoCandidates:
                trace = print_event_updates(f"nocand: {cell.position[:2]}", trace)
                # no candidates, but we will keep trying anyways
                pass
            
        else:
            if len(cell.candidates) == 0:
                trace = print_event_updates(f"nocand: {cell.position[:2]}", trace)

                # trace = print_event_updates(f"No value for cells: {failed_cells}", trace)
            # raise exceptions.NoCandidates
        # check if sudoku is solved
        if _sudoku.is_solved():
            return _sudoku, trace
        else:
            # revert and try again
            trace = print_event_updates(f"revert: {cell.position[:2]} {sorted(cell.candidates)} = {cell.value}", trace)
            trace = print_cell_updates([cell], trace) 
            _sudoku.update([cell])

            _sudoku, trace = s_backtrack_stubborn(_sudoku, trace) # see if the board can be solved 


    return _sudoku, trace

def generate_full_trace_backtrack_solution(sudoku: Sudoku) -> str:
    full_trace_solution = []
    sudoku, full_trace_solution = s_backtrack(sudoku, full_trace_solution)
    
    assert sudoku.is_valid()  # confirm that the solution is valid
    # print(renderers.colorful(sudoku))

    return sudoku, full_trace_solution

def generate_strategy_trace_solution(sudoku: Sudoku) -> str:
    # old version without backtrack
    # try:
    #     for step in solvers.steps(sudoku):
    #             for cell in step.changes:
    #                 if cell.value is not None:
    #                     strategy_trace_solution.append(f"{cell.position[:2]} = {cell.value}")
    # except exceptions.Unsolvable:
    #     []
                
    # return strategy_trace_solution

    try:
        strategy_full_trace_solution = []
        sudoku, strategy_full_trace_solution = s_backtrack_strategy(sudoku, strategy_full_trace_solution)
    except:
        try: # give another shot 
            strategy_full_trace_solution = []
            sudoku, strategy_full_trace_solution = s_backtrack_strategy(sudoku, strategy_full_trace_solution)
        except: # then revert to the original solution
            return sudoku, []
    
    assert sudoku.is_valid()  # confirm that the solution is valid
    # print(renderers.colorful(sudoku))

    return sudoku, strategy_full_trace_solution

def convert_full_trace_shortcut_solution(full_trace_solution: List) -> str:
    ''' take the full trace solution and remove the backtrack steps '''

    last_guess = []
    shortcut_solution = []
    # sudoku_array = defaultdict(lambda: (-1, -1)) # store the sudoku array with the index
    for i, step in enumerate(full_trace_solution):
        if "revert" in step:
            last_guess_index = last_guess.pop()
            shortcut_solution = shortcut_solution[:last_guess_index]
            # pass
        elif "guess" in step:
            last_guess.append(len(shortcut_solution))
            pass
        elif "invalid" in step:
            pass
        elif "nocand" in step:
            pass
        elif "=" in step:
            # row, col, val = map(int, re.match(r"Update: \((\d+), (\d+)\) -> (\d+)", step).groups())
            # sudoku_array[(row, col)] = (val, i)
            shortcut_solution.append(step)
        else:
            raise ValueError(f"Unknown event: {step}")
    
    ''' alternative simple implementation '''
    # sort dictionary by the index 
    # shortcut_solution = []
    # for key, value in sorted(sudoku_array.items(), key=lambda x: x[1][1]):
    #     if value[0] != -1:
    #         shortcut_solution.append(f"Update: {key} -> {value[0]}")
    
    return shortcut_solution


def convert_full_trace_shortcut_solution_fuzzy(full_trace_solution: List) -> str:
    ''' take the full trace solution and remove the backtrack steps 
    This implementation is more fuzzy, it does not rely on the backtrack steps '''
    sudoku_array = defaultdict(lambda: (-1, -1)) # store the sudoku array with the index
    for i, step in enumerate(full_trace_solution):
        if "revert" in step:
            pass
        elif "guess" in step:
            pass
        elif "invalid" in step:
            pass
        elif "nocand" in step:
            pass
        elif "=" in step:
            try:
                row, col, val = map(int, re.match(r"\((\d+), (\d+)\) = (\d+)", step).groups())
                sudoku_array[(row, col)] = (val, i)
            except:
                if i == len(full_trace_solution) -1:
                    pass
                else:
                    # print(full_trace_solution[i-10: max(i+10, len(full_trace_solution))])
                    print(f"Unknown event with =: {step}")
                    pass
                    # raise ValueError(f"Unknown event: {step}")
        else:
            if i == len(full_trace_solution) -1:
                pass
            else:
                # print(full_trace_solution[i-10: max(i+10, len(full_trace_solution))])
                print(f"Unknown event: {step}")
                pass
                # raise ValueError(f"Unknown event: {step}")
            # raise ValueError(f"Unknown event: {step}")
    # sort dictionary by the index 
    shortcut_solution = []
    for key, value in sorted(sudoku_array.items(), key=lambda x: x[1][1]):
        if value[0] != -1:
            shortcut_solution.append(f"{key} = {value[0]}")
    return shortcut_solution


def print_board(sudoku: Sudoku) -> None:
    board = ["original:"]
    for cell in sorted(sudoku.cells(), key=operator.attrgetter("position")):
        if cell.value is not None:
            board.append(f"{cell.position[:2]} = {cell.value} ")
    board.append("solving")
    return board

def check_sudoku_solution(board: str, trace: List) -> bool:
    # contruct a 9 by 9 emptry array
    sudoku_array = np.zeros((9, 9), dtype=int)
    
    # record the initial board
    for cell in board:
        if "original" in cell or "solving" in cell:
            continue # skip the original and solving messages

        match = re.match(r"\((\d+), (\d+)\) = (\d+)", cell)
        row, col, val = map(int, match.groups())  # Convert to integers
        sudoku_array[row, col] = val
    
    solved =  np.all(sudoku_array != 0)
    assert not solved # confirm that the sudoku is not solved
    
    # now we have the sudoku array, we can check the solution
    for step in trace:
        if "guess" in step or "revert" in step:
            raise NotImplementedError("Backtracked solution is not implemented")
        elif "=" in step:
            match = re.match(r"\((\d+), (\d+)\) = (\d+)", step)
            row, col, val = map(int, match.groups()) 
            sudoku_array[row, col] = val
        else:
            raise NotImplementedError("Backtracked solution is not implemented")

    # convert the array to a sudoku object
    # first confirm there is no zere-value in the array
    solved = np.all(sudoku_array != 0)
    solved_sudoku = Sudoku.from_list(sudoku_array, BoxSize(3, 3))
    is_valid = solved_sudoku.is_valid()
    # if not (solved and is_valid):
    #     print(renderers.colorful(solved_sudoku))

    return (solved and is_valid)


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(0)

    for i in range(100):
        sudoku = generators.random_sudoku(avg_rank=150, box_size=BoxSize(3, 3))

        ''' naive solution '''
        solved_sudoku, full_trace_solution = generate_full_trace_backtrack_solution(sudoku)

        shortcut_solution = convert_full_trace_shortcut_solution(full_trace_solution)

        board = print_board(sudoku)
        check_solution_backtrack = check_sudoku_solution(board, shortcut_solution)
        
        ''' strategy solution '''
        assert not sudoku.is_solved()

        strategy_solved_sudoku, strategy_full_trace_solution = generate_strategy_trace_solution(sudoku)
        if len(strategy_full_trace_solution) == 0:
            print("Failed to solve sudoku w/ strategy", i)
            strategy_solved_sudoku, strategy_full_trace_solution = generate_full_trace_backtrack_solution(sudoku)
        
        strategy_shortcut_solution = convert_full_trace_shortcut_solution(strategy_full_trace_solution)
        
        assert strategy_solved_sudoku.is_solved()
        board = print_board(sudoku)
        check_solution_strategy = check_sudoku_solution(board, strategy_shortcut_solution)
        
        assert len(shortcut_solution) == len(strategy_shortcut_solution)


        ''' stubborn solution --> not working '''
        # board = print_board(sudoku)
        # assert not sudoku.is_solved()
        # stubborn_solved_sudoku, stubborn_full_trace_solution = s_backtrack_stubborn(sudoku)
        # board = print_board(stubborn_solved_sudoku)

        
        ''' check solutions '''
        if not check_solution_backtrack:
            print("Failed to solve sudoku w/ backtrack", i)
        # if not check_solution_strategy:
            # print("Failed to solve sudoku w/ strategy", i)

            # for i in range(len(full_trace_solution)):
            #     print(full_trace_solution[i])
            # print("-------------------"*10)
            # for i in range(len(shortcut_solution)):
            #     print(shortcut_solution[i])
