# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.pruning.prune_utils import SimpleCDPruner


def test_simple_cd_pruner_shape_2d():
    embeddings = torch.randn(10, 16)
    pruner = SimpleCDPruner(keep_ratio=0.4, temperature=0.2)

    pruned = pruner(embeddings)

    assert pruned.shape == (4, 16)


def test_simple_cd_pruner_shape_3d():
    embeddings = torch.randn(3, 12, 8)
    pruner = SimpleCDPruner(keep_ratio=0.25, temperature=0.2)

    pruned = pruner(embeddings)

    assert pruned.shape == (3, 3, 8)


def test_simple_cd_pruner_keep_ratio_one_returns_original():
    embeddings = torch.randn(2, 7, 5)
    pruner = SimpleCDPruner(keep_ratio=1.0, temperature=0.2)

    pruned = pruner(embeddings)

    assert torch.equal(pruned, embeddings)


def test_simple_cd_pruner_relevance_conditioning_suppresses_irrelevant_token():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    text_embedding = torch.tensor([1.0, 0.0])
    pruner = SimpleCDPruner(keep_ratio=0.5, temperature=0.5)

    pruned = pruner(embeddings, text_embedding)

    irrelevant = embeddings[3]
    contains_irrelevant = torch.any(
        torch.all(torch.isclose(pruned, irrelevant.unsqueeze(0)), dim=-1)
    )
    assert not contains_irrelevant


def test_simple_cd_pruner_reproducible_for_same_input():
    torch.manual_seed(0)
    embeddings = torch.randn(20, 6)
    pruner = SimpleCDPruner(keep_ratio=0.5, temperature=0.2)

    pruned_first = pruner(embeddings)
    pruned_second = pruner(embeddings)

    assert torch.equal(pruned_first, pruned_second)


@pytest.mark.parametrize(
    ("keep_ratio", "temperature"),
    [
        (0.0, 0.1),
        (-0.1, 0.1),
        (1.1, 0.1),
        (0.5, 0.0),
        (0.5, -1.0),
    ],
)
def test_simple_cd_pruner_invalid_constructor_args(
    keep_ratio: float, temperature: float
):
    with pytest.raises(ValueError):
        SimpleCDPruner(keep_ratio=keep_ratio, temperature=temperature)


def test_simple_cd_pruner_invalid_text_batch_size_raises():
    embeddings = torch.randn(2, 10, 4)
    text_embeddings = torch.randn(3, 4)
    pruner = SimpleCDPruner(keep_ratio=0.5, temperature=0.2)

    with pytest.raises(ValueError):
        pruner(embeddings, text_embeddings)
