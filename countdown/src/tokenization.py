import os
import json
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Digits, BertPreTokenizer, Sequence
import random

def sample_corpus():
    # load the dataset
    train_dir = '/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_random'
    train_path = 'train1_b4_t100_n500000_random.json'
    val_path = 'val1_b4_t100_n500000_random.json'
    val_target_path = 'val_target1_b4_t100_n50000_random.json'
    # files = [os.path.join(train_dir, train_path), 
    #          os.path.join(train_dir, val_path),
    #         os.path.join(train_dir, val_target_path)]
    files = [
            os.path.join(train_dir, train_path), 
            os.path.join(train_dir, val_path),
    #          os.path.join(train_dir, val_target_path)
    ]

    optimal_paths = []
    search_paths = []
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            for entry in data:
                optimal_paths.append(entry['optimal_path'])
                search_paths.append(entry['search_path'])
    # randomly sample 1000 entries from each 
    random_idx = list(range(len(optimal_paths)))
    random_idx = random.sample(random_idx, 100)
    optimal_paths = [optimal_paths[i] for i in random_idx]
    search_paths = [search_paths[i] for i in random_idx]
    # combine the two lists
    combined_data = optimal_paths + search_paths

    # Save the dataset to a text file
    dataset_file = "countdown_corpus.txt"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for line in combined_data:
            f.write(line + "\n")

    return

def find_unique_vocab():
    import re
    with open("countdown_corpus.txt", "r", encoding="utf-8") as file:
        text = file.read()

    words = re.findall(r'\b[a-zA-Z]+\b', text)

    # Get unique words
    unique_words = list(set(words))

    # Display unique words
    return unique_words


def pretrain_tokenizer():
    unique_words = find_unique_vocab()
    math_symbols = ["+", "-", "*", "/", "=", "[", "]", "#"]
    digits = [str(i) for i in range(10)]         
    punctuation = [",", ":", ".", "'"]   
    special_tokens = ["UNK", "PAD", " START ", " END "]                                    

    vocab_list = digits + unique_words + math_symbols + punctuation + special_tokens

    vocab = {}
    for i, v in enumerate(vocab_list):
        vocab[v] = i

    # create tokenizer from vocab
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="UNK"))
    tokenizer.pre_tokenizer = Sequence([BertPreTokenizer(), Digits(individual_digits=True)])
    
    with open("countdown_corpus.txt", "r", encoding="utf-8") as file:
        text = file.read()
    sample_text = text
   
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Original: {sample_text[0:100]}")
    print(f"Encoded: {encoded.ids[0:100]}")    
    print(f"Decoded: {decoded[0:100]}")
    # check whether there is any UNK in decoded
    if "UNK" in decoded:
        print("There is an unknown token in the decoded text")
    else:
        print("There is no unknown token in the decoded text")

    # save the tokenizer
    tokenizer.save("countdown_tokenizer.json")

    return 


if __name__ == "__main__":
    # sample_corpus()
    pretrain_tokenizer()
