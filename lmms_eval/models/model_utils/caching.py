import torch
from typing import List, Tuple


def longest_common_prefix_len(
    seqs: List[torch.Tensor],
) -> Tuple[int, List[int]]:
    """
    Args
    ----
    seqs : list of 1 × Lᵢ int64 tensors (as returned by an HF tokenizer)

    Returns
    -------
    prefix_len      : int
        Length of the shared prefix (0 if none).
    suffix_lengths  : list[int]
        For each sequence, the number of tokens *after* the shared prefix,
        i.e. len(seqᵢ) − prefix_len.
    """
    # remove the dummy batch dim → [Lᵢ]
    seqs = [s.squeeze(0) for s in seqs]

    # common prefix can’t be longer than the shortest sequence
    min_len = min(s.size(0) for s in seqs)
    if len(seqs) == 1:                       # only one sequence ⇒ whole thing
        prefix_len = min_len
    else:
        # stack the truncated sequences into an (N × min_len) tensor
        stacked = torch.stack([s[:min_len] for s in seqs])  # N × min_len

        # column-wise equality: True where *all* sequences share the same token
        equal_cols = (stacked == stacked[0]).all(dim=0)     # 1 × min_len

        # first position where they diverge
        diff_pos = (~equal_cols).nonzero(as_tuple=False)
        prefix_len = min_len if diff_pos.numel() == 0 else diff_pos[0, 0].item()

    # suffix length for each sequence
    suffix_lengths = [s.size(0) - prefix_len for s in seqs]

    return prefix_len, suffix_lengths
