import torch

from control_optimization.trajectory import propagate_with_population_100_sum


def test_identity_has_zero_population_100_sum() -> None:
    grid = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]

    def get_step(_controls, _dt, _time):
        return torch.eye(8, dtype=torch.complex128)

    result = propagate_with_population_100_sum(
        get_step,
        grid,
        controls,
        basis_indices=list(range(8)),
        dimension=8,
        return_traces=True,
    )
    assert float(result.population_100_sum) == 0.0
    assert torch.equal(result.population_100_trace, torch.zeros(3, dtype=torch.float64))


def test_sum_uses_exact_000_to_100_population() -> None:
    grid = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]
    swap = torch.eye(8, dtype=torch.complex128)
    swap[0, 0] = 0.0
    swap[4, 4] = 0.0
    swap[0, 4] = 1.0
    swap[4, 0] = 1.0

    calls = 0
    def get_step(_controls, _dt, _time):
        nonlocal calls
        calls += 1
        return swap if calls == 1 else torch.eye(8, dtype=torch.complex128)

    result = propagate_with_population_100_sum(
        get_step,
        grid,
        controls,
        basis_indices=list(range(8)),
        dimension=8,
        return_traces=True,
    )
    assert torch.equal(
        result.population_100_trace,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
    )
    assert float(result.population_100_sum) == 2.0


def test_population_is_conditioned_on_logical_survival_like_plot() -> None:
    grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]
    step = torch.eye(10, dtype=torch.complex128)
    # |000> -> sqrt(1/4)|100> + sqrt(3/4)|leakage>
    step[:, 0] = 0.0
    step[4, 0] = 0.5
    step[8, 0] = (3.0 / 4.0) ** 0.5

    def get_step(_controls, _dt, _time):
        return step

    result = propagate_with_population_100_sum(
        get_step,
        grid,
        controls,
        basis_indices=list(range(8)),
        dimension=10,
        return_traces=True,
    )
    # Logical survival is 1/4, so the normalized plotted P100 is one.
    assert torch.allclose(result.population_100_trace[-1], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(result.logical_survival_trace[-1], torch.tensor(0.25, dtype=torch.float64))
