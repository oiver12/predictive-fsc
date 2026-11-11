"""A parameter list, one solver call per job, and completed-result skipping."""

from dataclasses import asdict
from collections.abc import Callable, Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any
from zipfile import BadZipFile

from .io import load_problem, load_result, save_result
from .baselines import _docker, fivi, saynt
from .io.results import provenance
from .search import _check_config, search, successive_halving
from .types import EMConfig, LoadedProblem, Problem, SearchConfig, SearchResult


PREDICTIVE_ONLY = ("trajectories", "fits_per_round", "reactive_starts", "exploration", "em")


def configuration(
    settings: Mapping[str, Any], *, option: Callable[[str], str] = str,
) -> tuple[SearchConfig, list[int]]:
    """Use SearchConfig field names, with a scalar or list for memory.

    Settings that the chosen search would ignore are rejected. Stopping rule: the
    search ends at whichever of cpu_seconds and rounds comes first; with neither,
    the SearchConfig CPU default applies. `option` names a setting in error messages.
    """
    settings = dict(settings)
    memory = settings.pop("memory", SearchConfig.memory)
    memories = [memory] if isinstance(memory, int) else memory
    if (not isinstance(memories, list) or not memories
            or any(type(memory) is not int or memory < 1 for memory in memories)):
        raise ValueError(f"{option('memory')} must be a positive integer or a nonempty positive integer list")
    if len(memories) != len(set(memories)):
        raise ValueError(f"{option('memory')} sizes must be unique")
    if settings.get("initialization") == "random":
        unused = [option(name) for name in PREDICTIVE_ONLY if name in settings]
        if unused:
            raise ValueError(f"random initialization does not use {', '.join(unused)} "
                             f"(it uses only memory, seed, CPU budget, rounds and the sweep limit)")
    if len(memories) > 1 and "rounds" in settings:
        raise ValueError(f"successive halving takes a CPU budget per arm, not {option('rounds')}")
    if settings.get("rounds") is not None and "cpu_seconds" not in settings:
        settings["cpu_seconds"] = None
    try:
        settings["em"] = EMConfig(**settings.pop("em", {}))
        config = SearchConfig(memory=memories[0], **settings)
    except TypeError as error:
        raise ValueError(f"unknown run setting: {error}") from None
    try:
        _check_config(config)
    except TypeError as error:
        raise ValueError(f'invalid run setting: {error}') from None
    return config, memories


def _summary(result: SearchResult) -> dict[str, Any]:
    return dict(value=result.value, memory=result.config.memory, cpu_seconds=result.cpu_seconds,
                candidates=len(result.progress),
                eligible_candidates=len(result.eligible),
                failures=len(result.failures))


def run_job(
    problem: Problem, config: SearchConfig, memories: Sequence[int], output: str | Path,
) -> dict[str, Any]:
    """Save one result; halving additionally saves every arm and its allocation."""
    output = Path(output)
    if len(memories) == 1:
        result = search(problem, config)
        summary = _summary(result)
    else:
        halving = successive_halving(problem, config, memory_sizes=memories)
        output.unlink(missing_ok=True)
        for arm in halving.arms:
            save_result(output.with_name(f"{output.stem}-m{arm.config.memory}.npz"), arm)
        result = halving.best if halving.best is not None else halving.arms[0]
        summary = dict(value=result.value, memory=halving.selected_memory,
                       cpu_seconds=halving.cpu_seconds,
                       half_values=halving.half_values, survivors=halving.survivors,
                       memory_sizes=[arm.config.memory for arm in halving.arms],
                       arm_budget_cpu_seconds=config.cpu_seconds,
                       arm_results={str(arm.config.memory): f"{output.stem}-m{arm.config.memory}.npz"
                                    for arm in halving.arms},
                       candidates=sum(len(arm.progress) for arm in halving.arms),
                       eligible_candidates=sum(len(arm.eligible) for arm in halving.arms),
                       failures=sum(len(arm.failures) for arm in halving.arms))
        allocation = output.with_suffix(".halving.json")
        allocation.parent.mkdir(parents=True, exist_ok=True)
        allocation.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    save_result(output, result)
    return {**summary, 'problem': problem.name, 'result': str(output)}


def _baseline_job(
    problem: Problem, config: fivi.FiviConfig | saynt.SayntConfig, output: Path, force: bool,
) -> dict[str, Any]:
    solver = 'fivi' if isinstance(config, fivi.FiviConfig) else 'saynt'
    adapter = fivi if solver == 'fivi' else saynt
    image = _docker.image_id(f'predictive-fsc-{solver}')
    if solver == 'fivi':
        data = (json.dumps(fivi.export_model(problem), allow_nan=False, separators=(',', ':')) + '\n').encode()
        model_hash = hashlib.sha256(data).hexdigest()
    else:
        model_hash = provenance(problem, config.seed).problem_sha256
    key = dict(model=model_hash, settings=asdict(config), image=image)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    directory = output / f'{problem.name}-{solver}-{digest}'
    skipped = (directory / 'result.json').exists() and not force
    try:
        if skipped:
            result = adapter.load_result(directory)
            if (result['solver'] != solver or result['config'] != asdict(config)
                    or result['image_id'] != image):
                raise ValueError('saved solver configuration or image does not match this job')
            if result['solver'] == 'fivi' and result['model_sha256'] != model_hash:
                raise ValueError('saved model does not match this job')
        else:
            result = adapter.run(problem, config, directory, image)  # type: ignore[arg-type]
        summary = dict(status='skipped' if skipped else 'done', solver=result['solver'], problem=problem.name,
                       result=str(directory / 'result.json'), value=result['value'], solver_status=result['status'])
        if result['solver'] == 'fivi':
            counts = result['vectors_per_stage']
            summary.update(upper_bound=result['upper_bound'], iteration=result['iteration'],
                           max_vectors_per_stage=max(counts) if counts else None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if skipped:
            raise ValueError(f'invalid cached result: {directory}; rerun with --force') from error
        raise
    return summary


def reproduce(
    plan: str | Path, output: str | Path, *, force: bool = False, dry_run: bool = False,
    run_index: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Run all entries, or one 1-based entry; scheduling belongs to the launcher."""
    plan, output = Path(plan), Path(output)
    with plan.open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) != {"runs"} or not isinstance(document["runs"], list):
        raise ValueError("reproduction plan must contain a [[runs]] list")
    if run_index is not None and not 1 <= run_index <= len(document['runs']):
        raise ValueError(f"--run-index must be between 1 and {len(document['runs'])}")
    # Repeated entries load each model just once, including during enumeration.
    problems: dict[Path, LoadedProblem] = {}
    for index, entry in enumerate(document["runs"], 1):
        if run_index is not None and index != run_index:
            continue
        if not isinstance(entry, dict) or 'problem' not in entry:
            raise ValueError('each [[runs]] entry must specify a problem file')
        settings = entry.copy()
        solver = settings.pop('solver', 'mapo')
        problem_path = plan.parent / settings.pop("problem")
        if problem_path not in problems:
            problems[problem_path] = load_problem(problem_path)
        problem = problems[problem_path].problem
        if solver in ('fivi', 'saynt'):
            # Reject invalid settings before contacting Docker.
            try:
                config = (fivi.FiviConfig if solver == 'fivi' else saynt.SayntConfig)(**settings)
            except TypeError as error:
                raise ValueError(f'unknown or missing {"FiVI" if solver == "fivi" else solver.upper()} setting: {error}') from None
            if dry_run:
                yield dict(status='planned', solver=solver, problem=problem.name, config=asdict(config))
            else:
                yield _baseline_job(problem, config, output, force)
            continue
        if solver != 'mapo':
            raise ValueError('solver must be mapo, fivi or saynt')
        config, memories = configuration(
            settings, option=lambda name: "[runs.em]" if name == "em" else name)
        if dry_run:
            yield dict(status='planned', solver=solver, problem=problem.name,
                       config=asdict(config), memories=memories)
            continue
        identity = provenance(problem, config.seed)
        # The key covers what was asked for; the build is recorded inside each result.
        key = dict(problem=identity.problem_sha256, settings=asdict(config), memories=memories)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        path = output / f"{problem.name}-{config.initialization}-{digest}.npz"
        if path.exists() and not force:
            try:
                cached = load_result(path)
                if len(memories) > 1:
                    allocation = json.loads(path.with_suffix(".halving.json").read_text())
                    for arm_path in allocation['arm_results'].values():
                        load_result(path.parent / arm_path)
            except (OSError, ValueError, KeyError, BadZipFile, EOFError) as error:
                raise ValueError(f'incomplete cached result: {path}; rerun with --force') from error
            yield dict(status="skipped", problem=problem.name, result=str(path), value=cached.value,
                       same_build=cached.provenance.build_identity == identity.build_identity)
        else:
            # A forced halving run overwrites arms before saving its final result.
            # Remove the old completion marker before any new work can fail.
            path.unlink(missing_ok=True)
            yield {'status': 'done', **run_job(problem, config, memories, path)}
