"""
CLI for prompting a base model as a plain continuation model.

Unlike scripts.chat_cli, this does not insert chat template tokens such as
<|user_start|> or <|assistant_start|>. It prepends only BOS by default, then
samples a continuation from the base checkpoint.
"""

import argparse

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine


def parse_args():
    parser = argparse.ArgumentParser(description="Continue text with a base model")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Base checkpoint tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("-p", "--prompt", type=str, default="", help="Prompt to continue once, then exit")
    parser.add_argument("--max-tokens", type=int, default=160, help="Maximum tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature; 0 = greedy")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling; <=0 disables top-k")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend the BOS token")
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps"],
        help="Device type: cuda|cpu|mps. Empty means autodetect.",
    )
    return parser.parse_args()


def get_special_id_to_text(tokenizer):
    return {tokenizer.encode_special(text): text for text in tokenizer.get_special_tokens()}


def decode_token(tokenizer, token, special_id_to_text):
    if token in special_id_to_text:
        return special_id_to_text[token]
    return tokenizer.decode([token])


def render_tokens(tokenizer, tokens, special_id_to_text):
    return "".join(decode_token(tokenizer, token, special_id_to_text) for token in tokens)


def render_continuation(engine, tokenizer, prompt, args):
    tokens = tokenizer.encode(prompt)
    if not args.no_bos:
        tokens = [tokenizer.get_bos_token_id()] + tokens

    # Keep room for generated tokens in the model context.
    max_context = engine.model.config.sequence_len
    max_prompt_tokens = max_context - args.max_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("--max-tokens must be smaller than the model context length")
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[-max_prompt_tokens:]

    special_id_to_text = get_special_id_to_text(tokenizer)
    print(f"Prompt: {prompt}")
    print("-" * 100)
    print(render_tokens(tokenizer, tokens, special_id_to_text), end="", flush=True)

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    generated = []
    for token_column, _token_masks in engine.generate(
        tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
    ):
        token = token_column[0]
        generated.append(token)
        print(decode_token(tokenizer, token, special_id_to_text), end="", flush=True)
    print()
    return generated


def main():
    args = parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _ddp_rank, _ddp_local_rank, _ddp_world_size, device = compute_init(device_type)
    model, tokenizer, _meta = load_model(
        "base",
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)

    if args.prompt:
        render_continuation(engine, tokenizer, args.prompt, args)
        return

    print("\nNanoChat Base Continuation Mode")
    print("-" * 50)
    print("Type a prompt and the base model will continue it.")
    print("Type 'quit' or 'exit' to end.")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nPrompt: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower().strip() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not prompt.strip():
            continue
        print()
        render_continuation(engine, tokenizer, prompt, args)


if __name__ == "__main__":
    main()
