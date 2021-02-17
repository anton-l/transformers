import argparse
import gzip
import os
import json
# TODO: relative import
from transformers.models.clip.tokenization_clip import bytes_to_unicode


def convert_tokenizer_merges(merges_path, output_path):
    with gzip.open(merges_path) as merges_file:
        merges = merges_file.read().decode("utf-8").split('\n')
    # get 49152 most common merges, minus 256 base byte tokens and 2 special tokens (start/endoftext)
    merges = merges[1:49152 - 256 - 2 + 1]
    merges = [tuple(merge.split()) for merge in merges]
    tokens = list(bytes_to_unicode().values())
    tokens = tokens + [t + '</w>' for t in tokens]
    for merge in merges:
        tokens.append(''.join(merge))
    tokens.extend(['<|startoftext|>', '<|endoftext|>'])
    vocab = dict(zip(tokens, range(len(tokens))))

    with open(os.path.join(output_path, "vocab.json"), "w") as vocab_out:
        vocab_out.write(json.dumps(vocab))
    with open(os.path.join(output_path, "merges.txt"), "w") as merges_out:
        merges_out.write("#version: 0.2\n")
        for merge in merges:
            merges_out.write(" ".join(merge) + "\n")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--clip_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the official PyTorch checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer_merges_path",
        default=None,
        type=str,
        required=True,
        help="Path to the official PyTorch checkpoint file.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    convert_tokenizer_merges(args.tokenizer_merges_path, args.pytorch_dump_folder_path)

if __name__ == "__main__":
    main()