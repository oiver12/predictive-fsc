"""Small command-line interface to the solver and portable result files."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import math
import signal
import subprocess
import sys
from zipfile import BadZipFile

from .io import load_problem, load_result
from .io.results import provenance
from .policy import evaluate_controller
from .reproduction import configuration, run_job, reproduce
from .types import SearchConfig


DEFAULTS = SearchConfig()
STOPPING = (f"The search stops at whichever of --cpu-seconds and --rounds comes first; "
            f"with neither, it uses {DEFAULTS.cpu_seconds:g} CPU seconds.")
EM_FLAGS = {"em_iterations": "max_iterations", "em_tolerance": "tolerance",
            "pseudocount": "pseudocount"}
OPTIONS = {"em": "--em-iterations/--em-tolerance/--pseudocount"}


def _search_options(parser, output_help, memory_help, cpu_help, *, halving=False):
    """Add the options shared by solve and halving, grouped by when they apply."""
    common = parser.add_argument_group("common options")
    common.add_argument("--memory", type=int, nargs="+", required=halving, help=memory_help)
    common.add_argument("--output", type=Path, help=output_help)
    common.add_argument("--seed", type=int, help=f"root seed (default {DEFAULTS.seed})")
    common.add_argument("--cpu-seconds", type=float, help=cpu_help)
    if not halving:
        common.add_argument("--rounds", type=int,
                            help="round limit; predictive: batches of fits after bootstrap, "
                                 "random: random starts (default: none)")
    common.add_argument("--initialization", choices=("predictive", "random"),
                        help=f"default {DEFAULTS.initialization}")
    predictive = parser.add_argument_group(
        "predictive search only", "Rejected with --initialization random.")
    predictive.add_argument("--trajectories", type=int,
                            help="trajectories per batch (default: the paper's size rule)")
    predictive.add_argument("--fits-per-round", type=int,
                            help=f"EM fits per batch (default {DEFAULTS.fits_per_round})")
    predictive.add_argument("--reactive-starts", type=int,
                            help=f"reactive bootstrap controllers (default {DEFAULTS.reactive_starts})")
    predictive.add_argument("--exploration", type=float,
                            help=f"uniform-action probability when sampling (default {DEFAULTS.exploration:g})")
    predictive.add_argument("--em-iterations", type=int,
                            help=f"maximum EM updates per fit (default {DEFAULTS.em.max_iterations})")
    predictive.add_argument("--em-tolerance", type=float,
                            help=f"EM convergence tolerance (default {DEFAULTS.em.tolerance:g})")
    predictive.add_argument("--pseudocount", type=float,
                            help=f"EM smoothing pseudocount (default {DEFAULTS.em.pseudocount:g})")
    safety = parser.add_argument_group("safety limit")
    safety.add_argument("--max-sweeps", type=int,
                        help=f"policy-iteration sweep cap per refinement (default {DEFAULTS.max_sweeps})")


def _search_settings(args):
    """Only the options the user typed; SearchConfig supplies every default."""
    settings = {key: value for key, value in vars(args).items()
                if key not in ("command", "problem", "output")}
    em = {EM_FLAGS[key]: settings.pop(key) for key in list(settings) if key in EM_FLAGS}
    if em:
        settings["em"] = em
    return settings


def _option(name: str) -> str:
    return OPTIONS.get(name, "--" + name.replace("_", "-"))


def _parser():
    parser = argparse.ArgumentParser(prog="predictive-fsc")
    parser.add_argument("--version", action="version", version=version("predictive-fsc"))
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate paired TOML/.pomdp problems")
    validate.add_argument("paths", type=Path, nargs="+")

    solve = commands.add_parser(
        "solve", argument_default=argparse.SUPPRESS, help="run one predictive or random search",
        description="Run one search and save one result. " + STOPPING)
    solve.add_argument("problem", type=Path)
    _search_options(solve, "result NPZ path (default: outputs/<problem>.npz)",
                    f"memory size (default {DEFAULTS.memory}); use halving for several",
                    f"CPU budget in seconds (default {DEFAULTS.cpu_seconds:g} unless --rounds is given)")

    halving = commands.add_parser(
        "halving", argument_default=argparse.SUPPRESS,
        help="successive halving over several memory sizes",
        description="Give every memory size half the budget, then continue the better half "
                    "to the full per-arm budget. Saves the selected result, every arm "
                    "(<output>-mN.npz) and the allocation (<output>.halving.json).")
    halving.add_argument("problem", type=Path)
    _search_options(halving, "result NPZ path (default: outputs/<problem>-halving.npz)",
                    "at least two memory sizes",
                    f"full CPU budget per arm (default {DEFAULTS.cpu_seconds:g})", halving=True)

    evaluate = commands.add_parser("evaluate", help="verify and reevaluate a saved controller")
    evaluate.add_argument("problem", type=Path)
    evaluate.add_argument("result", type=Path)
    reproduce_parser = commands.add_parser(
        "reproduce", help="run a TOML parameter list; skip completed jobs",
        description="Run every [[runs]] entry. Default solver=mapo uses SearchConfig fields; "
                    "a memory list runs successive halving. solver=fivi uses budget_seconds, "
                    "seed and optional budget_clock, parallelism, heap and iterations. "
                    "solver=saynt uses budget_seconds, seed and PAYNT/Storm phase seconds. " + STOPPING)
    reproduce_parser.add_argument("plan", type=Path)
    reproduce_parser.add_argument("--output", type=Path,
                                  help="result directory (default: outputs/<plan>/)")
    reproduce_parser.add_argument("--force", action="store_true", help="rerun completed jobs")
    reproduce_parser.add_argument("--dry-run", action="store_true", help="validate and list jobs without running solvers or Docker")
    reproduce_parser.add_argument("--run-index", type=int,
                                  help="execute only this 1-based [[runs]] entry")
    figures = commands.add_parser("figures", help="plot saved traces or generate paper panels and CSV tables")
    figures.add_argument("results", type=Path, nargs="+", help="NPZ files or result directories")
    figures.add_argument("--output", type=Path,
                         help="figure path, .svg/.pdf/.png (default: anytime.svg in a single results directory)")
    figures.add_argument("--paper", choices=('main', 'navigation', 'tiger'),
                         help="generate paper panels and CSV tables from new saved results; --output is a directory")
    figures.add_argument("--models", type=Path, default=Path('models'),
                         help="problem directory for navigation occupancy and sampled paths (default: models)")
    return parser


def _evaluate(problem_path, result_path):
    problem = load_problem(problem_path).problem
    result = load_result(result_path)
    if result.provenance.problem_sha256 != provenance(problem, result.config.seed).problem_sha256:
        raise ValueError("saved result belongs to a different numerical problem")
    if result.controller is None or result.value is None:
        raise ValueError("result contains no controller completed within its budget")
    value = evaluate_controller(problem, result.controller)
    if not math.isclose(value, result.value, abs_tol=1e-7, rel_tol=1e-10):
        raise ValueError(f"saved value {result.value} differs from reevaluated value {value}")
    return dict(problem=problem.name, value=value, saved_value=result.value, matches=True)


def main() -> None:
    # External launchers send SIGTERM; unwind adapter cleanup before exiting.
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        _main()
    finally:
        signal.signal(signal.SIGTERM, previous)


def _main() -> None:
    # Limit already-loaded BLAS/OpenMP pools; numba is never run in parallel.
    import scipy.linalg  # noqa: F401  (load SciPy's bundled BLAS before limiting)
    from threadpoolctl import threadpool_limits
    import numba
    numba.set_num_threads(1)  # first: this loads numba's OpenMP layer
    threadpool_limits(1)
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "validate":
            for path in args.paths:
                problem = load_problem(path).problem
                model = problem.model
                print(f"{path}: valid ({model.num_states} states, {model.num_actions} actions, "
                      f"{model.num_observations} observations, H={problem.horizon})")
        elif args.command in ("solve", "halving"):
            settings = _search_settings(args)
            if args.command == "solve" and len(settings.get("memory", [0])) > 1:
                raise ValueError("solve takes one --memory size; use halving for several")
            if args.command == "solve" and "memory" in settings:
                settings["memory"] = settings["memory"][0]
            if args.command == "halving" and len(settings["memory"]) < 2:
                raise ValueError("halving needs at least two --memory sizes; use solve for one")
            config, memories = configuration(settings, option=_option)
            problem = load_problem(args.problem).problem
            suffix = "-halving" if args.command == "halving" else ""
            output = getattr(args, "output", None) or Path("outputs") / f"{problem.name}{suffix}.npz"
            print(json.dumps(run_job(problem, config, memories, output), allow_nan=False))
        elif args.command == "evaluate":
            print(json.dumps(_evaluate(args.problem, args.result), allow_nan=False))
        elif args.command == "reproduce":
            output = args.output or Path("outputs") / args.plan.stem
            for summary in reproduce(args.plan, output, force=args.force, dry_run=args.dry_run,
                                     run_index=args.run_index):
                print(json.dumps(summary, allow_nan=False), flush=True)
        elif args.command == "figures":
            if args.paper:
                from .analysis.paper import render
                output = args.output or Path('outputs') / f'figures-{args.paper}'
                files = render(args.paper, args.results, output, args.models)
                print(json.dumps(dict(files=[str(path) for path in files])))
                return
            from .analysis.plotting import plot_results
            paths = []
            for path in args.results:
                paths.extend(sorted(path.glob("*.npz")) if path.is_dir() else [path])
            output = args.output
            if output is None:
                if len(args.results) != 1 or not args.results[0].is_dir():
                    raise ValueError("--output is required unless one results directory is given")
                output = args.results[0] / "anytime.svg"
            skipped = plot_results(paths, output)
            print(json.dumps(dict(figure=str(output), results=len(paths),
                                  plotted=len(paths) - len(skipped),
                                  skipped=[str(path) for path in skipped])))
    except (ValueError, KeyError, OSError, BadZipFile, EOFError) as error:
        parser.exit(2, f"error: {error}\n")
    except subprocess.SubprocessError as error:
        parser.exit(2, f"error: {getattr(error, 'stderr', None) or error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "interrupted\n")


if __name__ == "__main__":
    main()
