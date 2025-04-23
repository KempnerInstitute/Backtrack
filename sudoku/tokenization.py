import os
import json
import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace    
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
    
def generate_mapping(vocab_shift):
    """
    Generates a mapping dictionary for all (x, y) pairs where x and y range from 0 to 8.
    Each pair is mapped to x * 9 + y.
    With a vocab_shift, the mapping is shifted by the size of the vocabulary.
    
    Returns:
        dict: A dictionary with keys as "(x, y)" strings and values as integers.
        and the reverse dict.
    """
    mapping = {f"({x}, {y})": x * 9 + y for x in range(9) for y in range(9)}
    shifted_mapping = {key: value + vocab_shift for key, value in mapping.items()}
    reverse_shifted_mapping = {value: key for key, value in shifted_mapping.items()}

    return shifted_mapping, reverse_shifted_mapping


def map_position_to_integter(input_string, mapping):
    """ Replace all (x, y) pairs in the sample string with integers. """

    pattern = r"\((\d), (\d)\)"
    # Replacement function
    def replace_function(match):
        pair = match.group(0)  # Extract the full match, e.g., "(1, 1)"
        return str(mapping.get(pair, pair))  # Replace with mapped value, or leave unchanged

    # Apply the replacement
    return re.sub(pattern, replace_function, input_string)

def map_integer_to_position(input_string, reverse_mapping):
    """ Perform reverse mapping of integers in the input string back to (x, y). """
    # Tokenize the input string to identify valid integers
    tokens = re.findall(r'\d+|\D+', input_string)  # Matches numbers and non-numbers separately

    # Replace integers with their reverse mapping if they exist
    result = []
    for token in tokens:
        if token.isdigit():  # Check if the token is a number
            number = int(token)
            if number in reverse_mapping:
                result.append(reverse_mapping[number])  # Replace with (x, y)
            else:
                result.append(token)  # Leave it unchanged if not in the mapping
        else:
            result.append(token)  # Non-numeric tokens remain unchanged

    return ''.join(result)

def data_preprocessing(data, mapping):
    # replace all (x, y) pairs with integers
    # replace new line with " n "
    for i in range(len(data)):
        data[i] = data[i].replace("\n", " n ")
        data[i] = map_position_to_integter(data[i], mapping)
    return data

def data_postprocessing(data, reverse_mapping):
    # replace integers with (x, y) pairs
    # replace " n " with new line
    for i in range(len(data)):
        data[i] = map_integer_to_position(data[i], reverse_mapping)
        data[i] = data[i].replace(" n ", "\n")
    return data

def load_tokenizer():
    # Load the tokenizer from the json file
    # tokenizer = Tokenizer.from_file("sudoku_tokenizer.json")

    # Load the tokenizer as PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/n/home05/sqin/self-correct/sudoku/sudoku_tokenizer.json")
    # special tokens
    tokenizer.sol_start_token = "SOL_S "
    tokenizer.sol_end_token = "SOL_E "
    tokenizer.bos_token_w_space = "START "
    tokenizer.eos_token_w_space = "END "

    # special_token_ids
    tokenizer.add_special_tokens({'bos_token': "START",
                                  'pad_token': "MASK",
                                  'eos_token': "END",
                                  "unk_token": "[UNK]"})

    # tokenizer.add_special_tokens({'pad_token': "PAD"})

    with open("/n/home05/sqin/self-correct/sudoku/sudoku_pos_mapping.json", "r") as f:
        mappings = json.load(f)
        mapping = mappings["mapping"]
        reverse_mapping = mappings["reverse_mapping"]

    return tokenizer, mapping, reverse_mapping

def test_tokenizer(test_tokenizer, mapping, reverse_mapping):
    # sample_data = json.load(open("/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/data/sudoku_data_0_20000.json"))
    sample_data = json.load(open("/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_fixed/sudoku_strategy_data_0_100000.json"))
    # sample_data = json.load(open("/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data/sudoku_strategy_data_0_100000.json"))
    # sample_data = json.load(open("/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_easy/easy_sudoku_strategy_data_sample.json"))
    # take a subset
    sample_data = sample_data[:500]
    data_len = []
    backtracked = 0
    for i in range(len(sample_data)):
        # sample_test = sample_data[i]["board"] + "\n" + sample_data[i]["full_trace_solution"]
        # sample_test = sample_data[i]["board"] + "\n" + sample_data[i]["shortcut_solution"]
        sample_test = sample_data[i]["board"] + "\n" + sample_data[i]["strategy_shortcut_solution"]

        sample_test = sample_test.replace("\n", " n ")
        if "revert" in sample_test:
            backtracked += 1
        sample_test = map_position_to_integter(sample_test, mapping)
        encoded = test_tokenizer.encode(sample_test, add_special_tokens=True)
        if i == 0: # print some samples
            decoded = tokenizer.decode(encoded.ids)
            decoded = map_integer_to_position(decoded, reverse_mapping)
            decoded = decoded.replace(" n ", "\n")
            # print(decoded)
        # confirm there is no unknown token
        if "<unk>" in encoded.tokens:
            print("Unknown token found")
            print(encoded.tokens)
            break
        else:
            data_len.append(len(encoded.ids))

    import numpy as np
    print(np.mean(data_len), np.std(data_len))
    quit()

    # plot a histogram of the tokenized data
    plt.figure(figsize=(10, 5))
    plt.hist(data_len, bins=50)
    plt.xlabel("Number of tokens")
    plt.ylabel("Counts")
    plt.savefig("tokenized_data.png")

    for max_ls in [512, 1024, 2048, 4096, 8192, 16384]:
        print(f"Number of samples shorter than {max_ls} tokens: {len([x for x in data_len if x < max_ls])}")

    print(f"Number of samples with backtracking: {backtracked}")

if __name__ == "__main__":
    # initialize vocab 
    vocab_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                "guess", "revert", "nocand", "invalid", "original", "solving", 
                "=", "[", "]", "none", "n", ":", ","]
    vocab = {}
    for i, v in enumerate(vocab_list):
        vocab[v] = i

    # generate the mapping, and add to vocab
    mapping, reverse_mapping = generate_mapping(len(vocab))
    vocab.update({str(val): val for val in mapping.values()})

    # add special tokens to the vocab_dict
    special_tokens = ["START", "END", "MASK", "SOL_S", "SOL_E"]
    special_tokens_dict = {token: idx + len(vocab) for idx, token in enumerate(special_tokens)}
    vocab.update(special_tokens_dict)

    # create tokenizer from vocab
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="UNK"))
    tokenizer.pre_tokenizer = Whitespace()

    # save the tokenizer
    tokenizer.save("sudoku_tokenizer.json")
    
    # save two mappings in one file
    with open("sudoku_pos_mapping.json", "w") as f:
        json.dump({"mapping": mapping, "reverse_mapping": reverse_mapping}, 
                  f,
                  indent=4, 
                  separators=(",", ": ")
                  )

    # test the tokenizer
    loaded_tokenizer = Tokenizer.from_file("sudoku_tokenizer.json")
    test_tokenizer(loaded_tokenizer, mapping, reverse_mapping)