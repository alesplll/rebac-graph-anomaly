"""Device choice and seeding are explicit and reproducible."""

import torch

from rga.nn.runtime import seed_torch, select_device, torch_generator

CPU = torch.device("cpu")


def test_cpu_is_chosen_when_cuda_is_not_wanted() -> None:
    assert select_device(prefer_cuda=False) == CPU


def test_the_same_seed_initialises_the_same_parameters() -> None:
    seed_torch(7)
    first = torch.nn.Linear(4, 4).weight.detach().clone()
    seed_torch(7)
    second = torch.nn.Linear(4, 4).weight.detach().clone()

    assert torch.equal(first, second)


def test_a_seeded_generator_repeats_its_draws() -> None:
    first = torch.randint(0, 100, (16,), generator=torch_generator(3, CPU))
    second = torch.randint(0, 100, (16,), generator=torch_generator(3, CPU))

    assert torch.equal(first, second)


def test_different_seeds_diverge() -> None:
    first = torch.randint(0, 100, (16,), generator=torch_generator(3, CPU))
    second = torch.randint(0, 100, (16,), generator=torch_generator(4, CPU))

    assert not torch.equal(first, second)
