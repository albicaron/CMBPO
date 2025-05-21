from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

from algs.mbpo_sac import MBPO_SAC, set_device  # type: ignore


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def make_env(env_name: str, max_episode_steps: int) -> gym.Env:
    """Create a Gymnasium environment."""
    try:
        return gym.make(env_name, max_episode_steps=max_episode_steps)  # type: ignore[return-value]
    except gym.error.Error as exc:
        raise ValueError(f'Unknown environment id {env_name!r}. It must be registered with Gymnasium.') from exc


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='main_sac.py',
        description='Train SAC / MBPO‑SAC agents from the command line.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment options
    parser.add_argument('--env', '-e',
                        default='HalfCheetah-v4',
                        help='Gymnasium environment ID.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed for NumPy and PyTorch.')

    # Training hyper‑parameters
    parser.add_argument('--episodes', '-n', type=int, default=50, help='Total number of training episodes.')
    parser.add_argument('--steps', '-t', type=int, default=1_000, help='Maximum env steps per episode.')

    # The next is a "model-based" flag, but it is not used in the code as it is always true
    parser.add_argument('--model_based',
                        type=str2bool,
                        default=False,
                        help='Enable model-based policy optimization (MBPO). Set to False to disable.')

    # Logging and output
    parser.add_argument('--log_wandb',
                        action='store_true',
                        default=True,
                        help='Enable Weights & Biases tracking.')
    parser.add_argument(
        '--save_dir',
        '-o',
        type=pathlib.Path,
        default=pathlib.Path('trained_agents'),
        help='Directory in which to save the trained agent.',
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # Reproducibility ----------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device ------------------------------------------------------------------
    device = set_device()

    # Environment -------------------------------------------------------------
    gym.register_envs(gymnasium_robotics)
    env = make_env(args.env, max_episode_steps=args.steps)

    # Agent -------------------------------------------------------------------
    agent = MBPO_SAC(env,
                     args.seed,
                     device,
                     log_wandb=args.log_wandb,
                     model_based=args.model_based)

    # Training ----------------------------------------------------------------
    agent.train(num_episodes=args.episodes, max_steps=args.steps)

    # Saving ------------------------------------------------------------------
    args.save_dir.mkdir(parents=True, exist_ok=True)
    agent.save_agent(base_dir=str(args.save_dir))

    print(f'Training done! Agent for {args.env!r} saved to {args.save_dir.resolve()!s}.')


if __name__ == '__main__':
    # Add the parent directory to the system path to import modules
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

    main()
