"""Generate a base-model completion from a restored E8/W5 release package."""
import argparse
import json
from pathlib import Path

import torch

from .evaluation import generate_greedy
from .runtime import SentencePieceTokenizer, load_quantized_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--execution', choices=('prefill', 'tokenwise'), default='prefill')
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    if len(tokenizer.encode(args.prompt))+args.max_new_tokens > 4096:
        parser.error('Prompt plus generation exceeds the source model training length of 4096 tokens')
    model = load_quantized_model(args.raw_dir)
    if tokenizer.sha256 != model._package_receipt['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from the quantized model source')
    output, ids, prompt_tokens, state_bytes = generate_greedy(
        model, tokenizer, args.prompt, args.max_new_tokens, args.execution)
    print(json.dumps({'prompt': args.prompt, 'completion': output, 'generated_ids': ids,
                      'prompt_tokens': prompt_tokens, 'state_bytes': state_bytes,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'weight_execution': 'fully decoded FP16 reference',
                      'package': model._package_receipt}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
