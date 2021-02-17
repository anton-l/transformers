import argparse
from clip import clip
from transformers import ClipTokenizer

TEST_TXT = [
    "Hello, world!",
    " Hello,  \n\t  world!",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \nsed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "An especia11y TЯICKY text 💁 (╯°□°）╯︵ ┻━┻)"
]


def compare_tokenizer(model_path):
    official_input_ids = clip.tokenize(TEST_TXT)
    official_decoded_text = [clip._tokenizer.decode(ids.numpy()) for ids in official_input_ids]
    print(official_input_ids)
    print(official_decoded_text)

    tokenizer = ClipTokenizer.from_pretrained(
        model_path,
        pad_token="<|pad|>",
        max_len=77,
    )
    tokens = tokenizer(
        TEST_TXT,
        add_special_tokens=True,
        padding="max_length",
    )
    decoded_text = [tokenizer.decode(ids) for ids in tokens['input_ids']]
    print(tokens['input_ids'])
    print(decoded_text)


def compare_everything(model_path):
    compare_tokenizer(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to the converted huggingface model dump.",
    )
    args = parser.parse_args()

    compare_everything(
        model_path=args.model_path
    )