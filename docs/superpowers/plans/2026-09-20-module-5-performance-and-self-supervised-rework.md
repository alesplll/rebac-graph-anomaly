# Модуль 5. Производительность и вторая попытка самообучения

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Снять питоновское узкое место в обучении, проверить измерением три
гипотезы из `docs/module-3-findings.md` §8 о том, почему самообучаемая сеть
проиграла, и сделать проверку обобщающей способности содержательной.

**Architecture:** Всё добавляется рядом с работающим кодом, а не вместо него.
Выборка отрицательных примеров переписывается на массивы с сохранением всех пяти
стратегий. Две новые идеи — цель реконструкции и голова соответствия — включаются
полями `ModelConfig` и выключены по умолчанию, поэтому старые артефакты и старые
числа остаются воспроизводимыми, а сравнение «со и без» получается честным на тех
же пяти сидах. Два новых аномальных паттерна регистрируются в реестре и попадают
только в отдельный набор данных, так что опубликованные таблицы модулей 2–3 не
обесцениваются.

**Tech Stack:** Python 3.12 через `uv`, numpy, torch как тензорная библиотека с
autograd, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-12-rebac-graph-anomaly-design.md` §15
(определение модуля 5) и `docs/module-3-findings.md` §8 (перечень направлений с
уже измеренными исходными числами).

## Global Constraints

- Python строго 3.12 через `uv`. Команды — `uv run pytest -m "not integration and not gpu"`, `uv run ruff check .`.
- **PyTorch Geometric и DGL не подключать никогда.** Все графовые операции пишем сами.
- Любая случайность — через явно засеянный `numpy.random.Generator`. Никакого модульного `random`, никаких несеянных значений по умолчанию.
- Пути только через `pathlib`. Перенос строки в файлах данных принудительно `\n`. Каждая точка входа под `if __name__ == "__main__":`.
- Комментарии и docstring в коде на английском. Сообщения коммитов короткие, на английском, без эмодзи, **без строк атрибуции ИИ**. После каждого коммита `git push origin main`. Веток в этом репозитории не заводим.
- Система не выносит вердикт о компрометации. Формулировки в коде и документации это отражают.
- Порядок шагов внутри задачи не сокращать: сначала падающий тест, потом подтверждение, что он падает по ожидаемой причине, потом минимальная реализация, потом зелёный прогон, потом коммит.
- Основная метрика — PR-AUC, всегда на пяти сидах `[1, 2, 3, 4, 5]`, всегда со средним и стандартным отклонением. Вывод по одному сиду не является выводом: это ровно та ошибка, которую модуль 3 уже совершил и исправил.

## Что меняется в уже опубликованных числах

Задача 1 меняет порядок обращений к генератору случайных чисел, поэтому
`docs/thesis/gnn.md` после неё обязан быть пересчитан (задача 6). Это не ошибка и
не регресс: набор данных, кандидаты и сиды те же, меняется только то, какие
именно испорченные рёбра выпали. Разница между старой и новой таблицей должна
укладываться в уже измеренный разброс по сидам; если не укладывается — это
находка, и её место в `docs/module-5-findings.md`.

Задачи 2, 3 и 5 ничего опубликованного не меняют: новое поведение выключено по
умолчанию, новые паттерны живут в отдельном конфиге. `baselines.md`,
`ablation.md`, `dataset.md`, `features.md`, `gradcheck.md` пересчитывать не нужно.

## File Structure

| Файл | Ответственность |
|---|---|
| `src/rga/nn/negatives.py` | Выборка испорченных рёбер. Переписывается целиком на массивы; публичная сигнатура и `CorruptedEdges` не меняются |
| `src/rga/nn/node_inputs.py` | Плюс `neighbourhood_targets` и `reconstruction_target` — выбор цели реконструкции живёт в одном месте, которое читают и обучение, и скоринг |
| `src/rga/nn/heads.py` | Плюс `CorrespondenceHead` |
| `src/rga/nn/config.py` | Плюс `reconstruction_target` и `correspondence_weight` |
| `src/rga/nn/train.py` | Плюс необязательный аргумент `context` и слагаемое соответствия в функции потерь |
| `src/rga/nn/scorer.py` | Плюс четвёртое слагаемое оценки, когда голова соответствия включена |
| `src/rga/nn/scoring.py` | `combine` принимает произвольное число рангов |
| `src/rga/eval/experiment.py` | Два новых имени скорера |
| `src/rga/generator/anomalies/collusion.py` | Новый модуль: `mutual_grant_ring` и `orphaned_owner` |
| `scripts/two_hop_study.py` | Диагностика §4.3: на каком расстоянии в графе стояли концы нормальных и аномальных кандидатов до появления ребра |
| `scripts/profile_epoch.py` | Замер стоимости эпохи, до и после векторизации |
| `configs/generator/small-divergent.yaml` | Набор данных со структурно непохожими скрытыми паттернами |
| `configs/experiments/{module5,generalisation}.yaml` | Два новых сравнения |
| `docs/thesis/{two-hop,generalisation,module5}.md` | Таблицы для ВКР |
| `docs/module-5-findings.md` | Что измерено и что из этого следует |

---

### Task 1: Векторизованная выборка отрицательных примеров

Сейчас `sample_negatives` — питоновский цикл по всем кандидатам с `rng.choice`
на каждой итерации: 2.8 секунды из 3.0 на эпоху, то есть видеокарта простаивает
94% времени. Переписываем на массивы, сохраняя все пять стратегий и их
чередование `draw % 5`.

**Files:**
- Modify: `src/rga/nn/negatives.py` (целиком, ниже строки с `_MAX_LEVEL`)
- Create: `scripts/profile_epoch.py`
- Test: `tests/nn/test_negatives.py` (дополняется)

**Interfaces:**
- Consumes: `AccessGraph` с полями `edge_src`, `edge_dst`, `num_nodes`; `LEVEL_CARRYING`.
- Produces: `sample_negatives(graph, src, dst, relation, level, *, per_edge, rng) -> CorruptedEdges` — сигнатура и тип результата без изменений, поэтому `train.py` править не нужно.

- [ ] **Step 1: Write the failing test**

В `tests/nn/test_negatives.py` дописать:

```python
def test_two_hop_negatives_land_within_two_steps() -> None:
    """Strategy 5 must produce a node reachable in two undirected steps."""
    graph = _chain_graph()  # existing helper: a path of 6 nodes
    rng = np.random.default_rng(7)
    corrupted = sample_negatives(
        graph,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([int(RelationType.HAS_PERMISSION)], dtype=np.int64),
        np.array([int(PermissionLevel.READ)], dtype=np.int64),
        per_edge=5,
        rng=rng,
    )
    within = _undirected_distances(graph, source=0)
    assert within[corrupted.dst[4]] <= 2


def test_every_negative_differs_from_its_positive() -> None:
    graph = _chain_graph()
    rng = np.random.default_rng(11)
    src = np.array([0, 2, 3], dtype=np.int64)
    dst = np.array([1, 3, 4], dtype=np.int64)
    relation = np.full(3, int(RelationType.HAS_PERMISSION), dtype=np.int64)
    level = np.full(3, int(PermissionLevel.WRITE), dtype=np.int64)
    corrupted = sample_negatives(
        graph, src, dst, relation, level, per_edge=5, rng=rng
    )
    for position in range(corrupted.src.size):
        origin = int(corrupted.origin[position])
        assert (
            int(corrupted.src[position]),
            int(corrupted.dst[position]),
            int(corrupted.relation[position]),
            int(corrupted.level[position]),
        ) != (int(src[origin]), int(dst[origin]), int(relation[origin]), int(level[origin]))


def test_sampling_is_reproducible_under_one_seed() -> None:
    graph = _chain_graph()
    src = np.array([0, 2], dtype=np.int64)
    dst = np.array([1, 3], dtype=np.int64)
    relation = np.full(2, int(RelationType.HAS_PERMISSION), dtype=np.int64)
    level = np.full(2, int(PermissionLevel.WRITE), dtype=np.int64)
    first = sample_negatives(
        graph, src, dst, relation, level, per_edge=4, rng=np.random.default_rng(3)
    )
    second = sample_negatives(
        graph, src, dst, relation, level, per_edge=4, rng=np.random.default_rng(3)
    )
    assert np.array_equal(first.dst, second.dst)
    assert np.array_equal(first.level, second.level)
```

Вспомогательная функция для первого теста, туда же в файл теста:

```python
def _undirected_distances(graph: AccessGraph, *, source: int) -> np.ndarray:
    """Hop counts from `source`, ignoring direction. Unreached nodes get a big number."""
    distance = np.full(graph.num_nodes, 1_000, dtype=np.int64)
    distance[source] = 0
    frontier = [source]
    while frontier:
        nxt: list[int] = []
        for node in frontier:
            for a, b in zip(graph.edge_src, graph.edge_dst, strict=True):
                for here, there in ((int(a), int(b)), (int(b), int(a))):
                    if here == node and distance[there] > distance[node] + 1:
                        distance[there] = distance[node] + 1
                        nxt.append(there)
        frontier = nxt
    return distance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_negatives.py -v`
Expected: `test_two_hop_negatives_land_within_two_steps` FAILS. Старый код при
`per_edge=5` обращается к стратегии `draw % 5 == 4`, но в его ветвлении это
`else`, и на позиции 4 действительно двухшаговый узел — тест может пройти. **Если
он проходит, это тоже результат: значит он закрепляет уже верное поведение, и его
роль — не дать векторизации его сломать.** Два остальных теста должны пройти на
старом коде. Отметить в выводе, какие именно упали, и идти дальше.

- [ ] **Step 3: Write the vectorised implementation**

Заменить `_undirected_neighbours`, `_two_hop` и тело `sample_negatives` на:

```python
def _undirected_csr(graph: AccessGraph) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour lists of the undirected projection, as (indptr, indices).

    Parallel edges are kept rather than deduplicated: building a set per node was
    the second Python loop in this module, and keeping duplicates only reweights
    the draw towards heavily connected pairs, leaving the set of reachable nodes
    exactly as it was.
    """
    tail = np.concatenate([graph.edge_src, graph.edge_dst]).astype(np.int64)
    head = np.concatenate([graph.edge_dst, graph.edge_src]).astype(np.int64)
    order = np.argsort(tail, kind="stable")
    indices = head[order]
    indptr = np.zeros(graph.num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(tail, minlength=graph.num_nodes), out=indptr[1:])
    return indptr, indices


def _random_neighbour(
    indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One neighbour of every given node, or -1 where the node has none."""
    if indices.size == 0:
        return np.full(nodes.shape, -1, dtype=np.int64)
    degree = indptr[nodes + 1] - indptr[nodes]
    offset = np.floor(rng.random(nodes.size) * np.maximum(degree, 1)).astype(np.int64)
    picked = indices[np.clip(indptr[nodes] + offset, 0, indices.size - 1)]
    return np.where(degree > 0, picked, -1)


def _two_hop(
    indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A node two undirected steps away, or -1 where the walk died out."""
    middle = _random_neighbour(indptr, indices, nodes, rng)
    second = _random_neighbour(indptr, indices, np.where(middle >= 0, middle, 0), rng)
    return np.where(middle >= 0, second, -1)


def sample_negatives(
    graph: AccessGraph,
    src: np.ndarray,
    dst: np.ndarray,
    relation: np.ndarray,
    level: np.ndarray,
    *,
    per_edge: int,
    rng: np.random.Generator,
) -> CorruptedEdges:
    """Draw `per_edge` corrupted variants of every positive.

    Every draw is made for the whole batch at once. The strategy of a draw is still
    `draw % 5`, so a batch still carries all five kinds in the same proportion as
    the per-edge loop this replaced; what changed is the order in which the random
    numbers are consumed, and therefore the exact edges that come out.
    """
    count = int(len(src))
    origin = np.repeat(np.arange(count, dtype=np.int64), per_edge)
    strategy = np.tile(np.arange(per_edge, dtype=np.int64) % 5, count)

    base_src = np.asarray(src, dtype=np.int64)[origin]
    base_dst = np.asarray(dst, dtype=np.int64)[origin]
    base_relation = np.asarray(relation, dtype=np.int64)[origin]
    base_level = np.asarray(level, dtype=np.int64)[origin]

    out_src, out_dst = base_src.copy(), base_dst.copy()
    out_relation, out_level = base_relation.copy(), base_level.copy()

    in_degree = np.bincount(graph.edge_dst, minlength=graph.num_nodes).astype(np.float64)
    popularity = in_degree + 1.0
    popularity /= popularity.sum()

    uniform = strategy == 0
    out_dst[uniform] = rng.integers(graph.num_nodes, size=int(uniform.sum()))

    popular = strategy == 1
    out_dst[popular] = rng.choice(graph.num_nodes, size=int(popular.sum()), p=popularity)

    swapped = strategy == 2
    out_src[swapped] = rng.integers(graph.num_nodes, size=int(swapped.sum()))

    carries = np.isin(base_relation, [int(kind) for kind in LEVEL_CARRYING])
    shifted = (strategy == 3) & carries
    if shifted.any():
        here = out_level[shifted]
        step = np.where(rng.random(int(shifted.sum())) < 0.5, -1, 1)
        step = np.where(here >= _MAX_LEVEL, -1, step)
        step = np.where(here <= _MIN_LEVEL, 1, step)
        out_level[shifted] = np.clip(here + step, _MIN_LEVEL, _MAX_LEVEL)

    # A level-carrying relation is what strategy 4 needs; without one it falls
    # through to the two-hop draw, exactly as the per-edge loop did.
    walked = (strategy == 4) | ((strategy == 3) & ~carries)
    if walked.any():
        indptr, indices = _undirected_csr(graph)
        reached = _two_hop(indptr, indices, out_src[walked], rng)
        fallback = rng.integers(graph.num_nodes, size=int(walked.sum()))
        out_dst[walked] = np.where(reached >= 0, reached, fallback)

    unchanged = (
        (out_src == base_src)
        & (out_dst == base_dst)
        & (out_relation == base_relation)
        & (out_level == base_level)
    )
    if unchanged.any() and graph.num_nodes > 1:
        # A corruption that changed nothing is not a negative. Draw uniformly from
        # the nodes other than the original object: the index is taken among
        # `num_nodes - 1` and stepped over the one that must not come out, which is
        # the loop-free form of the rejection the per-edge version used.
        avoid = base_dst[unchanged]
        drawn = rng.integers(graph.num_nodes - 1, size=int(unchanged.sum()))
        out_dst[unchanged] = drawn + (drawn >= avoid)

    return CorruptedEdges(
        src=out_src, dst=out_dst, relation=out_relation, level=out_level, origin=origin
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/nn/ -v`
Expected: PASS, включая все ранее существовавшие тесты модуля.

Затем весь набор, потому что `train.py` вызывает эту функцию:
Run: `uv run pytest -m "not integration and not gpu" -q`
Expected: PASS. Если упадёт тест воспроизводимости скорера — это ожидаемо и
означает, что он зафиксировал конкретные числа старой выборки; такой тест надо
переписать на утверждение «два прогона с одним сидом совпадают», а не «прогон
равен вот этому числу».

- [ ] **Step 5: Write the profiling script**

Create `scripts/profile_epoch.py`:

```python
"""Where the time of one training epoch goes.

Prints the cost of negative sampling against the cost of a forward and backward
pass, which is the measurement that decides whether the GPU is worth using at all.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.candidates import candidate_arrays
from rga.nn.config import ModelConfig
from rga.nn.encoder import GraphEncoder
from rga.nn.graph_tensors import graph_tensors
from rga.nn.negatives import sample_negatives
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features


def main() -> None:
    config = load_dataset_config(Path("configs/generator/small-history.yaml"))
    started = time.perf_counter()
    dataset = build_dataset(config)
    print(f"build_dataset: {time.perf_counter() - started:.1f}s, events {len(dataset.events)}")

    started = time.perf_counter()
    train = build_candidates(dataset, span=Span.TRAIN)
    print(f"build_candidates(TRAIN): {time.perf_counter() - started:.1f}s")

    arrays, _, _ = candidate_arrays(train)
    graph = train.graph
    assert graph is not None
    rng = np.random.default_rng(20260912)

    started = time.perf_counter()
    sample_negatives(
        graph, arrays.src, arrays.dst, arrays.relation, arrays.level, per_edge=4, rng=rng
    )
    sampling = time.perf_counter() - started
    print(f"sample_negatives once: {sampling:.2f}s")

    device = torch.device("cpu")
    tensors = graph_tensors(graph, device=device)
    inputs = node_input_features(tensors)
    encoder = GraphEncoder(NODE_INPUT_DIM, ModelConfig()).to(device)
    started = time.perf_counter()
    encoder(inputs, tensors).sum().backward()
    forward = time.perf_counter() - started
    print(f"one encoder forward+backward: {forward:.2f}s")
    print(f"estimate: {sampling + forward:.2f}s per epoch")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Measure the speedup**

Run: `uv run python scripts/profile_epoch.py`
Expected: `sample_negatives once` заметно меньше прежних 2.8 s. Записать полученное
число — оно идёт в `docs/module-5-findings.md` задачи 7 как измеренный, а не
обещанный результат. Если ускорения нет, остановиться и разобраться: гипотеза
модуля 3 о причине медлительности была бы опровергнута, и это важнее, чем
продолжать план.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check .
git add src/rga/nn/negatives.py tests/nn/test_negatives.py scripts/profile_epoch.py
git commit -m "perf: draw negative samples for the whole batch at once"
git push origin main
```

---

### Task 2: Реконструкция окрестности вместо профиля узла

`docs/module-3-findings.md` §4.2: отклонение профиля субъекта даёт устойчиво
обратный порядок (ROC-AUC 0.011), потому что ошибка реконструкции почти совпадает
со степенью узла. Проверяем предложенное там средство: восстанавливать не
собственный профиль узла, а усреднённый профиль его окрестности.

**Files:**
- Modify: `src/rga/nn/config.py`
- Modify: `src/rga/nn/node_inputs.py`
- Modify: `src/rga/nn/train.py:81` (строка с `F.mse_loss(model.reconstruction(h), inputs)`)
- Modify: `src/rga/nn/scorer.py` (метод `_encode`)
- Modify: `src/rga/eval/experiment.py` (функция `build_scorer`)
- Test: `tests/nn/test_node_inputs.py`

**Interfaces:**
- Consumes: `GraphTensors` с полями `src[slot]`, `dst[slot]`, `num_nodes`, `device`; `node_input_features(tensors) -> Tensor[num_nodes, NODE_INPUT_DIM]`.
- Produces:
  - `ModelConfig.reconstruction_target: str = "profile"` — допустимые значения `"profile"` и `"neighbourhood"`.
  - `neighbourhood_targets(tensors: GraphTensors, inputs: torch.Tensor) -> torch.Tensor` формы `[num_nodes, NODE_INPUT_DIM]`.
  - `reconstruction_target(tensors: GraphTensors, inputs: torch.Tensor, config: ModelConfig) -> torch.Tensor` — то, что обе стороны, обучение и скоринг, должны использовать вместо `inputs`.
  - Имя скорера `"gnn_neighbourhood"` в `build_scorer`.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_node_inputs.py` (или дописать, если файл есть):

```python
def test_neighbourhood_target_averages_the_neighbours() -> None:
    """A node's target is the mean profile of the nodes it touches, not its own."""
    graph = _star_graph()  # centre 0 joined to leaves 1, 2, 3
    tensors = graph_tensors(graph, device=torch.device("cpu"))
    inputs = node_input_features(tensors)
    targets = neighbourhood_targets(tensors, inputs)

    expected = inputs[[1, 2, 3]].mean(dim=0)
    assert torch.allclose(targets[0], expected, atol=1e-6)


def test_isolated_node_reconstructs_itself() -> None:
    """With no neighbours there is nothing to average, so the profile stands in."""
    graph = _star_graph_with_isolate()  # node 4 has no edges
    tensors = graph_tensors(graph, device=torch.device("cpu"))
    inputs = node_input_features(tensors)
    targets = neighbourhood_targets(tensors, inputs)

    assert torch.allclose(targets[4], inputs[4], atol=1e-6)


def test_target_choice_is_driven_by_the_configuration() -> None:
    graph = _star_graph()
    tensors = graph_tensors(graph, device=torch.device("cpu"))
    inputs = node_input_features(tensors)

    profile = reconstruction_target(tensors, inputs, ModelConfig())
    neighbourhood = reconstruction_target(
        tensors, inputs, replace(ModelConfig(), reconstruction_target="neighbourhood")
    )
    assert torch.allclose(profile, inputs)
    assert not torch.allclose(neighbourhood, inputs)


def test_an_unknown_target_is_refused() -> None:
    graph = _star_graph()
    tensors = graph_tensors(graph, device=torch.device("cpu"))
    inputs = node_input_features(tensors)
    with pytest.raises(ValueError, match="reconstruction target"):
        reconstruction_target(
            tensors, inputs, replace(ModelConfig(), reconstruction_target="nonsense")
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_node_inputs.py -v`
Expected: FAIL with `ImportError: cannot import name 'neighbourhood_targets'`.

- [ ] **Step 3: Write the implementation**

В `src/rga/nn/config.py` добавить два поля (второе понадобится задаче 3, но
объявляется здесь, чтобы `ModelConfig` правился один раз):

```python
    #: What the reconstruction head rebuilds: "profile" (the node's own attributes)
    #: or "neighbourhood" (the mean attributes of the nodes it touches). The profile
    #: variant makes the deviation track node degree; see docs/module-3-findings.md.
    reconstruction_target: str = "profile"
    #: Weight of the correspondence term. Zero leaves the head out of the model.
    correspondence_weight: float = 0.0
```

В `src/rga/nn/node_inputs.py` добавить в конец:

```python
def neighbourhood_targets(tensors: GraphTensors, inputs: torch.Tensor) -> torch.Tensor:
    """The mean attribute vector of every node's undirected neighbourhood.

    A node with no neighbours gets its own vector: there is nothing to average, and
    leaving a row of zeros would make every isolate look equally strange.
    """
    total = torch.zeros_like(inputs)
    seen = torch.zeros(tensors.num_nodes, device=tensors.device)
    for slot in range(NUM_SLOTS):
        src, dst = tensors.src[slot], tensors.dst[slot]
        total.index_add_(0, dst, inputs[src])
        seen.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    lonely = seen == 0
    averaged = total / seen.clamp(min=1.0).unsqueeze(1)
    return torch.where(lonely.unsqueeze(1), inputs, averaged)


def reconstruction_target(
    tensors: GraphTensors, inputs: torch.Tensor, config: ModelConfig
) -> torch.Tensor:
    """What the reconstruction head is asked to rebuild."""
    if config.reconstruction_target == "profile":
        return inputs
    if config.reconstruction_target == "neighbourhood":
        return neighbourhood_targets(tensors, inputs)
    raise ValueError(f"unknown reconstruction target: {config.reconstruction_target!r}")
```

Добавить `from rga.nn.config import ModelConfig` в импорты файла.

В `src/rga/nn/train.py` заменить строку слагаемого реконструкции:

```python
        targets = reconstruction_target(tensors, inputs, config)
        loss = loss + config.reconstruction_weight * F.mse_loss(model.reconstruction(h), targets)
```

Вычислять `targets` один раз перед циклом эпох, а не внутри — граф в цикле не
меняется.

В `src/rga/nn/scorer.py`, метод `_encode`:

```python
            deviation = (
                self._model.reconstruction.deviation(
                    state, reconstruction_target(tensors, inputs, self._config)
                )
                .cpu()
                .numpy()
            )
```

В `src/rga/eval/experiment.py`, в `build_scorer`, после ветки `"gnn"`:

```python
    if name == "gnn_neighbourhood":
        from dataclasses import replace

        from rga.nn.scorer import GnnScorer

        return GnnScorer(
            seed=seed, config=replace(ModelConfig(), reconstruction_target="neighbourhood")
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/nn/ -v`
Expected: PASS.

Run: `uv run pytest -m "not integration and not gpu" -q`
Expected: PASS. Значение по умолчанию `"profile"` означает, что ни одно
существующее поведение не изменилось.

- [ ] **Step 5: Commit**

```bash
uv run ruff check .
git add src/rga/nn/config.py src/rga/nn/node_inputs.py src/rga/nn/train.py \
        src/rga/nn/scorer.py src/rga/eval/experiment.py tests/nn/test_node_inputs.py
git commit -m "feat: let the reconstruction head rebuild the neighbourhood"
git push origin main
```

---

### Task 3: Голова соответствия структуры и контекста

`docs/module-3-findings.md` §4.1: в контрастной постановке признаковая строка
кандидата одинакова у положительного примера и всех его испорченных вариантов,
поэтому 73 веса головы никогда не получают градиента. Сейчас строка просто
исключена. §8 предлагает проверить другое средство — отдельную задачу
соответствия: «структура кандидата i против контекста кандидата j». Подмешанный к
контрастной вариант уже измерен и дал 0.112 ± 0.091; отдельный — нет.

**Files:**
- Modify: `src/rga/nn/heads.py`
- Modify: `src/rga/nn/train.py`
- Modify: `src/rga/nn/scorer.py`
- Modify: `src/rga/nn/scoring.py`
- Modify: `src/rga/eval/experiment.py`
- Test: `tests/nn/test_heads.py`, `tests/nn/test_scoring.py`

**Interfaces:**
- Consumes: `ModelConfig.correspondence_weight` из задачи 2; `CandidateArrays` с полями `src`, `dst`, `relation`, `level`, `features`.
- Produces:
  - `CorrespondenceHead(node_dim: int, context_dim: int, config: ModelConfig)` с `forward(h_src, h_dst, context) -> Tensor[batch]` — логиты.
  - `GnnModel(edge_dim, config, context_dim=0)`; поле `self.correspondence` существует только когда `config.correspondence_weight > 0` и `context_dim > 0`.
  - `train_model(..., context: np.ndarray | None = None)`.
  - `combine(*ranks: np.ndarray) -> np.ndarray`.
  - Имя скорера `"gnn_correspondence"`.

- [ ] **Step 1: Write the failing test**

Дописать в `tests/nn/test_heads.py`:

```python
def test_correspondence_head_returns_one_logit_per_pair() -> None:
    head = CorrespondenceHead(node_dim=8, context_dim=5, config=ModelConfig())
    h_src = torch.randn(4, 8)
    h_dst = torch.randn(4, 8)
    context = torch.randn(4, 5)
    assert head(h_src, h_dst, context).shape == (4,)


def test_correspondence_head_reacts_to_the_context_row() -> None:
    """The point of the head: a different context must move the logit."""
    torch.manual_seed(0)
    head = CorrespondenceHead(node_dim=8, context_dim=5, config=ModelConfig())
    h_src, h_dst = torch.randn(1, 8), torch.randn(1, 8)
    mine = head(h_src, h_dst, torch.zeros(1, 5))
    theirs = head(h_src, h_dst, torch.ones(1, 5))
    assert not torch.allclose(mine, theirs)
```

И в `tests/nn/test_scoring.py`:

```python
def test_combine_accepts_a_fourth_term() -> None:
    ranks = [np.array([0.0, 1.0]) for _ in range(4)]
    assert combine(*ranks).tolist() == [0.0, 1.0]


def test_combine_still_averages_three_terms() -> None:
    result = combine(
        np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.5, 0.0])
    )
    assert result.tolist() == [0.5, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_heads.py tests/nn/test_scoring.py -v`
Expected: FAIL with `ImportError: cannot import name 'CorrespondenceHead'` и
`TypeError: combine() takes 3 positional arguments but 4 were given`.

- [ ] **Step 3: Write the implementation**

В `src/rga/nn/heads.py` добавить:

```python
class CorrespondenceHead(nn.Module):
    """Does this context row belong to this change?

    The contrastive task corrupts structure only, so the context row of a positive
    travels with all of its negatives unchanged and the likelihood head never gets
    a gradient that tells the two apart. This head asks a question the row does
    decide: given the two endpoint representations, is this the row that came with
    the change, or somebody else's? It is trained alongside, never mixed into the
    likelihood term, so what it learns can be measured on its own.
    """

    def __init__(self, node_dim: int, context_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + context_dim, config.edge_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_hidden, 1),
        )

    def forward(
        self, h_src: torch.Tensor, h_dst: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Logits: high means the row and the change belong together."""
        return self.mlp(torch.cat([h_src, h_dst, context], dim=1)).squeeze(1)
```

В `src/rga/nn/scoring.py` заменить `combine`:

```python
def combine(*ranks: np.ndarray) -> np.ndarray:
    """The anomaly score: equal weights over the ranked terms.

    Three terms by default — likelihood and the two profile deviations. A model
    with a correspondence head contributes a fourth.
    """
    if not ranks:
        raise ValueError("combine needs at least one ranked term")
    return np.vstack(ranks).mean(axis=0)
```

В `src/rga/nn/train.py`:

```python
class GnnModel(nn.Module):
    """The encoder and its heads, trained together."""

    def __init__(self, edge_dim: int, config: ModelConfig, context_dim: int = 0) -> None:
        super().__init__()
        self.encoder = GraphEncoder(NODE_INPUT_DIM, config)
        self.likelihood = EdgeLikelihoodHead(config.hidden_dim, edge_dim, config)
        self.reconstruction = NodeReconstructionHead(config.hidden_dim, NODE_INPUT_DIM, config)
        self.correspondence = (
            CorrespondenceHead(config.hidden_dim, context_dim, config)
            if config.correspondence_weight > 0.0 and context_dim > 0
            else None
        )
```

В `train_model` добавить параметр и слагаемое:

```python
def train_model(
    graph: AccessGraph,
    arrays: CandidateArrays,
    config: ModelConfig,
    *,
    seed: int,
    device: torch.device,
    context: np.ndarray | None = None,
) -> GnnModel:
```

Сразу после создания модели:

```python
    context_dim = 0 if context is None else int(context.shape[1])
    model = GnnModel(
        edge_dim=arrays.features.shape[1], config=config, context_dim=context_dim
    ).to(device)
    context_tensor = None if context is None else torch.as_tensor(context, device=device)
```

И внутри цикла эпох, после строки со слагаемым реконструкции:

```python
        if model.correspondence is not None and context_tensor is not None:
            # A derangement: every change is paired with somebody else's row, so a
            # negative is never accidentally a positive.
            shuffled = fit_index[torch.as_tensor(
                (np.arange(fit.size) + 1 + rng.integers(fit.size - 1)) % fit.size,
                device=device,
            )]
            belongs = model.correspondence(
                h[src[fit_index]], h[dst[fit_index]], context_tensor[fit_index]
            )
            stranger = model.correspondence(
                h[src[fit_index]], h[dst[fit_index]], context_tensor[shuffled]
            )
            loss = loss + config.correspondence_weight * (
                F.binary_cross_entropy_with_logits(belongs, torch.ones_like(belongs))
                + F.binary_cross_entropy_with_logits(stranger, torch.zeros_like(stranger))
            )
```

В `src/rga/nn/scorer.py`:

- `fit`: сохранить полный контекст до зануления и передать его в обучение.

```python
        arrays, mean, std = candidate_arrays(train)
        context = arrays.features
        arrays = without_features(arrays)
        self._mean, self._std = mean, std
        self._model = train_model(
            train.graph,
            arrays,
            self._config,
            seed=self._seed,
            device=self._device,
            context=context if self._config.correspondence_weight > 0.0 else None,
        )
```

После обучения, там же, где подгоняются остальные ранговые преобразования:

```python
        if self._model.correspondence is not None:
            self._correspondence_rank = RankTransform.fit(
                1.0 - self._correspondence(state, arrays, context)
            )
```

- `score`: четвёртое слагаемое, когда голова есть.

```python
        terms = [
            unlikeliness,
            self._node_rank(deviation, arrays.src),
            self._node_rank(deviation, arrays.dst),
        ]
        if self._model.correspondence is not None and self._correspondence_rank is not None:
            terms.append(
                self._correspondence_rank.apply(
                    1.0 - self._correspondence(state, without_features(arrays), arrays.features)
                )
            )
        return combine(*terms)
```

- Новый закрытый метод рядом с `_likelihood`:

```python
    def _correspondence(
        self, state: torch.Tensor, arrays: CandidateArrays, context: np.ndarray
    ) -> np.ndarray:
        """Probability the model gives to the row belonging to the change."""
        assert self._model is not None and self._model.correspondence is not None
        padded = torch.cat([state, torch.zeros(1, state.shape[1], device=self._device)])
        unknown = padded.shape[0] - 1

        def endpoints(index: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.where(index < 0, unknown, index), device=self._device)

        with torch.no_grad():
            logits = self._model.correspondence(
                padded[endpoints(arrays.src)],
                padded[endpoints(arrays.dst)],
                torch.as_tensor(context, device=self._device),
            )
        return torch.sigmoid(logits).cpu().numpy()
```

- В `__init__` добавить `self._correspondence_rank: RankTransform | None = None`.
- В `state_for_artifact` и `restore_from_artifact` добавить `context_dim` и
  `correspondence_reference`, иначе сохранённая модель не восстановится. При
  восстановлении `GnnModel(edge_dim=..., config=self._config, context_dim=int(state["context_dim"]))`.

В `src/rga/eval/experiment.py`, в `build_scorer`:

```python
    if name == "gnn_correspondence":
        from dataclasses import replace

        from rga.nn.scorer import GnnScorer

        return GnnScorer(
            seed=seed, config=replace(ModelConfig(), correspondence_weight=1.0)
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/nn/ -v`
Expected: PASS.

Run: `uv run pytest -m "not integration and not gpu" -q`
Expected: PASS. `correspondence_weight` по умолчанию 0.0, значит `gnn`,
`gnn_supervised` и `gnn_forest` остались ровно такими, какими были.

- [ ] **Step 5: Smoke-test the new scorer end to end**

Run:
```bash
uv run rga evaluate --config configs/experiments/smoke.yaml --out /tmp/claude-1000/-home-wexel-Data-Code-Projects-rebac-graph-anomaly/0fda9a6a-828a-48fc-bd94-32fcf6a689fa/scratchpad/corr
```
предварительно добавив `gnn_correspondence` в `scorers` файла `smoke.yaml`
временно, локально, **не коммитя** эту правку.
Expected: прогон завершается, в выводе есть строка `gnn_correspondence`.
Затем вернуть `smoke.yaml` в исходное состояние: `git checkout -- configs/experiments/smoke.yaml`.

- [ ] **Step 6: Commit**

```bash
uv run ruff check .
git add src/rga/nn/heads.py src/rga/nn/train.py src/rga/nn/scorer.py \
        src/rga/nn/scoring.py src/rga/eval/experiment.py \
        tests/nn/test_heads.py tests/nn/test_scoring.py
git commit -m "feat: add a correspondence head over the candidate context row"
git push origin main
```

---

### Task 4: Почему двухшаговые отрицательные примеры не помогают

`docs/module-3-findings.md` §4.3 выдвигает объяснение и прямо говорит, что оно не
проверено: узел в двух шагах — это форма законной будущей выдачи, поэтому обучение
отвергать такие пары учит модель занижать правдоподобие нормальным изменениям.
Объяснение проверяемо измерением, без обучения: достаточно посчитать, на каком
расстоянии стояли концы кандидата в графе **до** появления ребра, отдельно для
нормальных и для аномальных.

**Files:**
- Create: `scripts/two_hop_study.py`
- Create: `docs/thesis/two-hop.md` (порождается скриптом)
- Test: `tests/nn/test_two_hop_study.py`

**Interfaces:**
- Consumes: `_undirected_csr` из задачи 1; `build_candidates(dataset, span=Span.EVAL)`; `CandidateSet` с полями `labels`, `graph`, и `rga.nn.candidates.candidate_arrays` для индексов концов.
- Produces: `hop_distribution(graph, src, dst) -> np.ndarray` — расстояние на каждого кандидата, где 0 — тот же узел, 1 — сосед, 2 — два шага, 3 — дальше или недостижим.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_two_hop_study.py`:

```python
"""The diagnostic behind module 3, section 4.3."""

import numpy as np

from rga.nn.two_hop import hop_distribution


def test_distance_zero_one_two_and_beyond() -> None:
    graph = _path_graph(5)  # 0-1-2-3-4
    src = np.array([0, 0, 0, 0], dtype=np.int64)
    dst = np.array([0, 1, 2, 4], dtype=np.int64)
    assert hop_distribution(graph, src, dst).tolist() == [0, 1, 2, 3]


def test_unknown_endpoint_counts_as_beyond() -> None:
    graph = _path_graph(3)
    src = np.array([-1, 0], dtype=np.int64)
    dst = np.array([1, -1], dtype=np.int64)
    assert hop_distribution(graph, src, dst).tolist() == [3, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_two_hop_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.two_hop'`.

- [ ] **Step 3: Write the implementation**

Create `src/rga/nn/two_hop.py`:

```python
"""How far apart the ends of a change stood before it happened.

Module 3 found that drawing hard negatives from the two-hop neighbourhood did not
help, and guessed why: a node two steps away is the shape of a legitimate future
grant, so teaching the model to reject those pairs teaches it to suppress ordinary
changes. That guess is measurable without training anything.
"""

from __future__ import annotations

import numpy as np

from rga.domain.graph import AccessGraph
from rga.nn.negatives import _undirected_csr

#: Everything further than two steps, unreachable, or with an unknown endpoint.
BEYOND = 3


def hop_distribution(graph: AccessGraph, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Undirected distance between the ends of every candidate, capped at `BEYOND`."""
    indptr, indices = _undirected_csr(graph)
    result = np.full(src.shape, BEYOND, dtype=np.int64)
    for position, (a, b) in enumerate(zip(src, dst, strict=True)):
        if a < 0 or b < 0:
            continue
        if a == b:
            result[position] = 0
            continue
        first = indices[indptr[a] : indptr[a + 1]]
        if b in first:
            result[position] = 1
            continue
        for middle in first:
            if b in indices[indptr[middle] : indptr[middle + 1]]:
                result[position] = 2
                break
    return result
```

Цикл здесь питоновский и это осознанно: скрипт запускается один раз на пяти сидах,
а не в каждой эпохе обучения.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/nn/test_two_hop_study.py -v`
Expected: PASS.

- [ ] **Step 5: Write the study script**

Create `scripts/two_hop_study.py`, который на пяти сидах строит датасет
`configs/generator/small-history.yaml`, берёт кандидатов окна оценки, считает
`hop_distribution` по графу на момент разделения и пишет
`docs/thesis/two-hop.md` с таблицей:

| Расстояние до появления ребра | Доля нормальных | Доля аномальных |
|---|---|---|
| 0 (тот же узел) | … | … |
| 1 (сосед) | … | … |
| 2 (два шага) | … | … |
| 3 и дальше | … | … |

Значения — среднее ± отклонение по пяти сидам. Файл открывается абзацем, который
называет вопрос и то, чем ответ является: доля нормальных кандидатов на расстоянии
2 против доли аномальных там же.

Скрипт завершается печатью вывода в одну строку, например:
`normal at two hops: 0.63 +- 0.02, anomalous: 0.21 +- 0.05`.

- [ ] **Step 6: Run the study**

Run: `uv run python scripts/two_hop_study.py`
Expected: создан `docs/thesis/two-hop.md`, в выводе две доли.

**Как читать результат.** Если доля нормальных на расстоянии 2 заметно выше доли
аномальных — объяснение §4.3 подтверждено: двухшаговая стратегия учит модель
отвергать то, что в окне оценки является нормой. Если доли близки — объяснение
опровергнуто, и в `docs/module-5-findings.md` идёт именно это, а не подгонка под
ожидание. Оба исхода одинаково пригодны для работы.

- [ ] **Step 7: Commit**

```bash
uv run ruff check .
git add src/rga/nn/two_hop.py tests/nn/test_two_hop_study.py \
        scripts/two_hop_study.py docs/thesis/two-hop.md
git commit -m "docs: measure how far apart the ends of a change stood"
git push origin main
```

---

### Task 5: Набор данных, на котором проверка обобщения что-то проверяет

`docs/module-3-findings.md` §2: разрыв между известными и скрытыми паттернами
составил +0.02, то есть проверка обобщающей способности почти ничего не проверила.
Причина названа там же — паттерны 6–8 структурно похожи на 1–5.

**Решение принято такое: паттерны 1–8 не трогаем.** Изменение генератора
обесценило бы `baselines.md`, `ablation.md`, `dataset.md` и `gnn.md`, а числа из
них уже стоят в тексте ВКР. Вместо этого добавляем два новых паттерна, заведомо
непохожих по форме на всё существующее, и отдельный набор данных, где они играют
роль скрытых. Опубликованные таблицы остаются верными, а вопрос §2 получает
измеренный ответ вместо оговорки.

Новые паттерны:

1. `mutual_grant_ring` — замкнутое кольцо взаимных выдач между тремя–четырьмя
   пользователями разных отделов: каждый получает право на бакет команды
   следующего, последний замыкает цикл на первого. Все восемь существующих
   паттернов — это звезда, цепь или одиночное ребро; замкнутого цикла среди
   пользователей не порождает ни один, и увидеть его можно только глядя на форму
   графа, а не на строку признаков отдельного изменения.
2. `orphaned_owner` — владение бакетом, переданное пользователю, вышедшему из всех
   групп. Единственный паттерн, создающий ребро `OWNER_OF`: остальные работают с
   `HAS_PERMISSION` и `MEMBER_OF`, поэтому и тип ребра, и положение субъекта здесь
   новые.

**Files:**
- Create: `src/rga/generator/anomalies/collusion.py`
- Modify: `src/rga/generator/anomalies/__init__.py`
- Create: `configs/generator/small-divergent.yaml`
- Create: `configs/experiments/generalisation.yaml`
- Test: `tests/generator/test_collusion.py`

**Interfaces:**
- Consumes: `Injection`, `InjectionContext`, `NoCandidateError`, `label_for`, `pick`, `register`, `sample_maybe_night_ts` из `rga.generator.anomalies.base`; `EventOp`, `GraphEvent`; `PermissionLevel`, `RelationType`; `context.org` (`.users`, `.buckets`, `.team_of`), `context.graph` (`.node_index`, `.node_ids`, `.neighbors`), `context.window`, `context.rng`.
- Produces: два зарегистрированных паттерна с именами `"mutual_grant_ring"` и `"orphaned_owner"`; доступны через `get_pattern` и попадают в `available_patterns()`.

- [ ] **Step 1: Write the failing test**

Create `tests/generator/test_collusion.py`:

```python
"""Two patterns shaped unlike anything in the training catalogue."""

import pytest

from rga.domain.relations import RelationType
from rga.generator.anomalies.base import available_patterns, get_pattern


def test_both_patterns_are_registered() -> None:
    assert "mutual_grant_ring" in available_patterns()
    assert "orphaned_owner" in available_patterns()


def test_the_ring_closes() -> None:
    """Every member of the ring both receives a right and is granted past."""
    context = _context(seed=1)
    injection = get_pattern("mutual_grant_ring").inject(context)

    subjects = [event.subject for event in injection.events]
    assert len(set(subjects)) == len(subjects)  # each person appears once
    assert len(subjects) >= 3
    # Every granted bucket belongs to the team of the next person in the ring,
    # and the last one wraps back to the first.
    assert injection.events[-1].object == _bucket_of_team(context, subjects[0])


def test_the_orphan_receives_ownership() -> None:
    context = _context(seed=2)
    injection = get_pattern("orphaned_owner").inject(context)

    assert len(injection.events) == 1
    event = injection.events[0]
    assert event.relation is RelationType.OWNER_OF
    # The subject belongs to no group at the time of the grant.
    assert context.graph.neighbors(event.subject, RelationType.MEMBER_OF).size == 0


def test_every_created_edge_is_labelled() -> None:
    for name in ("mutual_grant_ring", "orphaned_owner"):
        injection = get_pattern(name).inject(_context(seed=3))
        assert len(injection.labels) == len(injection.events)
        assert {label.pattern for label in injection.labels} == {name}
```

Вспомогательная `_context` строит небольшой датасет тем же способом, каким это
делают существующие тесты паттернов — скопировать приём из
`tests/generator/test_persistence.py`, не изобретать свой.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generator/test_collusion.py -v`
Expected: FAIL with `KeyError: 'mutual_grant_ring'`.

- [ ] **Step 3: Write the implementation**

Create `src/rga/generator/anomalies/collusion.py`. Образцом брать
`src/rga/generator/anomalies/persistence.py`: тот же декоратор `@register`, тот же
возврат `Injection(events=..., labels=...)`, тот же `NoCandidateError`, когда на
графе не нашлось места. Докстринг модуля объясняет, зачем эти два паттерна
существуют: они нужны не для реалистичности, а чтобы у проверки обобщающей
способности появился предмет — форма, которой в обучающем каталоге нет.

Ключевые места:

```python
#: How many people a ring passes through.
_RING_SIZE = (3, 4)


@register
class MutualGrantRing:
    """A closed cycle of reciprocal grants across department boundaries.

    Every other pattern in the catalogue is a star, a chain or a single edge. A
    cycle is the one shape a per-change feature row cannot see at all: each edge
    of it looks like an ordinary cross-team grant, and only the closure makes the
    group of them strange.
    """

    name = "mutual_grant_ring"
```

и

```python
@register
class OrphanedOwner:
    """Ownership of a bucket handed to somebody who has left every group.

    The only pattern that creates an OWNER_OF edge, and the only one whose subject
    sits outside the membership structure entirely. Both facts are new to a model
    trained on the first five patterns.
    """

    name = "orphaned_owner"
```

Зарегистрировать модуль в `src/rga/generator/anomalies/__init__.py`, добавив
`collusion` в список импортов «ради побочного эффекта».

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/generator/ -v`
Expected: PASS.

Run: `uv run pytest -m "not integration and not gpu" -q`
Expected: PASS. Проверить отдельно, что тест, фиксирующий состав
`available_patterns()`, если он существует, обновлён, а не обойдён.

- [ ] **Step 5: Add the dataset and the experiment**

Create `configs/generator/small-divergent.yaml` — копия `small-history.yaml` с
заменённым блоком `anomalies`:

```yaml
# small-history with a hidden set that is genuinely unlike the training set.
#
# Module 3 measured a generalisation gap of +0.02 and concluded that patterns 6-8
# are too similar to 1-5 for the check to mean anything. This dataset keeps the
# five training patterns as they are and puts two structurally different ones in
# the evaluation window: a closed ring of reciprocal grants, and ownership handed
# to somebody outside the membership structure.
anomalies:
  patterns:
    - self_grant_admin
    - privileged_group_join
    - grant_burst
    - hierarchy_bypass
    - cross_department
    - mutual_grant_ring
    - orphaned_owner
  rate: 0.02
  train_contamination: 0.0
  train_patterns:
    - self_grant_admin
    - privileged_group_join
    - grant_burst
    - hierarchy_bypass
    - cross_department
  train_rate: 0.004
```

Остальные секции — `name: small-divergent`, `seed`, `eval_window_days`, `org`,
`timeline` — скопировать из `small-history.yaml` дословно.

Create `configs/experiments/generalisation.yaml`:

```yaml
# Does the supervised model survive a hidden set it has no template for?
#
# The companion to gnn.yaml: same scorers, same seeds, a dataset whose evaluation
# window carries two patterns shaped unlike anything in the training span.
name: generalisation
dataset: configs/generator/small-divergent.yaml
seeds: [1, 2, 3, 4, 5]
scorers:
  - rules
  - isolation_forest
  - gnn
  - gnn_supervised
ks: [20, 50, 100]
```

- [ ] **Step 6: Run the experiment**

Run:
```bash
uv run rga evaluate --config configs/experiments/generalisation.yaml \
    --out experiments/runs/generalisation
```
Expected: таблица записана, в разбивке по паттернам видны строки
`mutual_grant_ring` и `orphaned_owner`.

Скопировать таблицу в `docs/thesis/generalisation.md`, дополнив её абзацем,
который прямо называет измеренное: полнота@50 супервизорной сети на обученных
паттернах против полноты на этих двух. Если она заметно просела — гипотеза
дизайн-документа из §8, отвергнутая в модуле 3 на похожих паттернах, на
непохожих подтверждается, и это ровно то ограничение метода, которое в работе
надо назвать. Если не просела — вывод ещё сильнее: модель переносится и на
незнакомую форму.

- [ ] **Step 7: Commit**

```bash
uv run ruff check .
git add src/rga/generator/anomalies/collusion.py src/rga/generator/anomalies/__init__.py \
        tests/generator/test_collusion.py configs/generator/small-divergent.yaml \
        configs/experiments/generalisation.yaml docs/thesis/generalisation.md
git commit -m "feat: add two hidden patterns unlike the training catalogue"
git push origin main
```

---

### Task 6: Пересчёт таблиц и сравнение вариантов

Задача 1 изменила последовательность случайных чисел, поэтому `gnn.md` обязан быть
пересчитан. Задачи 2 и 3 добавили два варианта, которые надо сравнить с исходным на
тех же пяти сидах: одиночный сид здесь ничего не доказывает — модуль 3 уже сделал
этот вывод дважды и оба раза был опровергнут проверкой на пяти.

**Files:**
- Create: `configs/experiments/module5.yaml`
- Modify: `docs/thesis/gnn.md` (перезаписывается прогоном)
- Create: `docs/thesis/module5.md`
- Modify: `docs/thesis/README.md`
- Modify: `docs/run-on-gpu.md`

- [ ] **Step 1: Create the comparison config**

Create `configs/experiments/module5.yaml`:

```yaml
# The three self-supervised variants against each other and against the best
# classical method. Same dataset, same seeds, same candidates as gnn.yaml.
name: module5
dataset: configs/generator/small-history.yaml
seeds: [1, 2, 3, 4, 5]
scorers:
  - isolation_forest
  - gnn
  - gnn_neighbourhood
  - gnn_correspondence
ks: [20, 50, 100]
```

- [ ] **Step 2: Re-run the module 3 comparison**

Run:
```bash
uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn
```
Expected: прогон завершается. Записать новые числа в `docs/thesis/gnn.md`.

**Сравнить с прежними** (`gnn_supervised` 0.888 ± 0.115, `isolation_forest`
0.719 ± 0.118, `gnn_forest` 0.682 ± 0.087, `gnn` 0.198 ± 0.152). Если расхождение
укладывается в отклонение — так и записать. Если нет — остановиться, это находка, и
разбирать её надо по `superpowers:systematic-debugging`, а не списывать на шум.

- [ ] **Step 3: Run the variant comparison**

Run:
```bash
uv run rga evaluate --config configs/experiments/module5.yaml --out experiments/runs/module5
```
Expected: четыре строки в таблице.

- [ ] **Step 4: Write the table**

Создать `docs/thesis/module5.md` с полученной таблицей и разбивкой по паттернам.
Открывающий абзац называет исходную точку — `gnn` 0.198 ± 0.152 — и говорит, что
сравнивается: цель реконструкции и отдельная задача соответствия, по одной
за раз, на одних и тех же сидах.

- [ ] **Step 5: Update the artefact index**

В `docs/thesis/README.md` дописать три строки таблицы:

| `two-hop.md` | На каком расстоянии стояли концы изменения до его появления | `uv run python scripts/two_hop_study.py` |
| `generalisation.md` | Скрытые паттерны, непохожие на обучающие | `uv run rga evaluate --config configs/experiments/generalisation.yaml --out experiments/runs/generalisation` |
| `module5.md` | Три самообучаемых варианта против лучшего классического метода | `uv run rga evaluate --config configs/experiments/module5.yaml --out experiments/runs/module5` |

- [ ] **Step 6: Update the GPU instructions**

В `docs/run-on-gpu.md` заменить оценку времени полного прогона: она опиралась на
2.8 секунды выборки отрицательных примеров, которых больше нет. Подставить число,
измеренное в задаче 1 шаг 6, и пересчитать оценку для
`configs/experiments/gnn-full.yaml`. Добавить, что после прогона на ПК обратно
переносится только `docs/thesis/gnn-full.md` и содержимое
`experiments/runs/gnn-full`.

- [ ] **Step 7: Commit**

```bash
git add configs/experiments/module5.yaml docs/thesis/gnn.md docs/thesis/module5.md \
        docs/thesis/README.md docs/run-on-gpu.md
git commit -m "docs: recompute the comparison and add the module 5 variants"
git push origin main
```

---

### Task 7: Закрытие модуля

**Files:**
- Create: `docs/module-5-findings.md`
- Modify: `docs/module-3-findings.md` (раздел 8 — отметить, что проверено)
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Write the findings**

Create `docs/module-5-findings.md` по образцу `docs/module-3-findings.md`: те же
разделы, тот же принцип — каждое утверждение подкреплено измерением, а
отрицательный результат записывается как отрицательный.

Обязательные разделы:

1. **Что сделано** — векторизация, два варианта модели, два новых паттерна, две диагностики.
2. **Стоимость эпохи до и после** — числа из `scripts/profile_epoch.py`, и что из этого следует для видеокарты.
3. **Реконструкция окрестности** — PR-AUC `gnn_neighbourhood` против `gnn` на пяти сидах; отдельно, перестало ли отклонение профиля субъекта давать обратный порядок (было ROC-AUC 0.011).
4. **Голова соответствия** — PR-AUC `gnn_correspondence` против `gnn` и против уже измеренного подмешанного варианта 0.112 ± 0.091.
5. **Двухшаговые отрицательные примеры** — подтвердилось ли объяснение §4.3.
6. **Проверка обобщения на непохожих паттернах** — просадка супервизорной модели или её отсутствие.
7. **Что из этого следует** — защищаемые утверждения, каждое с числом.

Если вариант не выиграл, так и писать. Три отрицательных результата модуля 3 —
самая сильная часть работы именно потому, что они разобраны, а не спрятаны.

- [ ] **Step 2: Mark what section 8 asked for as done**

В `docs/module-3-findings.md` раздел 8 превратить из списка намерений в список со
ссылками: против каждого пункта — где теперь лежит ответ.

- [ ] **Step 3: Update CLAUDE.md**

Раздел «Текущее состояние»: закрыть модуль 5, перечислить, что готово, добавить
строки про новые документы в таблицу. Снять фразу «Следующий шаг — модуль 4»,
которая уже неверна. Оставить честную формулировку про то, выиграли варианты или
нет.

- [ ] **Step 4: Update README.md**

README на английском и намеренно короткий. Добавить не более трёх строк: что
делает `scripts/profile_epoch.py`, и что `configs/experiments/` содержит
`module5.yaml` и `generalisation.yaml`. Ничего не расписывать.

- [ ] **Step 5: Final verification**

```bash
uv run pytest -m "not integration and not gpu" -q
uv run ruff check .
git status --short
```
Expected: тесты зелёные, ruff молчит, дерево чистое.

- [ ] **Step 6: Commit**

```bash
git add docs/module-5-findings.md docs/module-3-findings.md CLAUDE.md README.md
git commit -m "docs: close module 5"
git push origin main
```

---

## Что остаётся пользователю

Полный прогон `configs/experiments/gnn-full.yaml` на ПК с видеокартой — по
инструкции из `docs/run-on-gpu.md`, обновлённой в задаче 6. Это единственный шаг
плана, который нельзя выполнить на ноутбуке. Результат — `docs/thesis/gnn-full.md`.
