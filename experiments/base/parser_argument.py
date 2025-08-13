import argparse
from functools import wraps
from typing import Callable, List


def output_added_arguments(add_algo_arguments: Callable) -> Callable:
    @wraps(add_algo_arguments)
    def decorated(parser: argparse.ArgumentParser) -> List[str]:
        unfiltered_old_arguments = list(parser._option_string_actions.keys())

        add_algo_arguments(parser)

        unfiltered_arguments = list(parser._option_string_actions.keys())
        unfiltered_added_arguments = [
            argument for argument in unfiltered_arguments if argument not in unfiltered_old_arguments
        ]

        return [
            argument.strip("-")
            for argument in unfiltered_added_arguments
            if argument.startswith("--") and argument not in ["--help"]
        ]

    return decorated


@output_added_arguments
def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-dw",
        "--disable_wandb",
        help="Disable wandb.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-ls",
        "--learning_starts",
        help="How many samples to collect before training starts.",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "-uh",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        help="Number of collected samples.",
        default=1_000_000,
    )
    parser.add_argument(
        "-fq",
        "--features_qf",
        type=int,
        nargs="*",
        help="List of features for the Q-networks.",
        default=[256, 256],
    )
    parser.add_argument(
        "-fpi",
        "--features_pi",
        type=int,
        nargs="*",
        help="List of features for the actor.",
        default=[256, 256],
    )


def add_tau(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-tau",
        "--tau",
        help="Tau in target update.",
        type=float,
        default=5e-3,
    )


@output_added_arguments
def add_sac_arguments(parser: argparse.ArgumentParser):
    add_tau(parser)
