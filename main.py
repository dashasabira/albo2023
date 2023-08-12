import logging
import os
import sys
import yaml

from argparse import ArgumentParser

from albo.utils.argparse_utils import apply_updates, replace_env_vars, format_nested
from albo.runners import albo_noiseless as albo_noiseless_runner


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = ArgumentParser()
    parser.add_argument('--config-file', required=True)

    args = parser.parse_args()

    logger.debug("Config file %s", args.config_file)
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    if hasattr(args, "overrides") and args.overrides is not None:
        apply_updates(config, args.overrides)

    # apply env substitution
    config = replace_env_vars(config)

    # interpolate task_id
    task_id_template = config["task_id"]
    task_id_interpolated = format_nested(task_id_template, config)
    config["task_id"] = task_id_interpolated

    # identify the runner for this task
    runner = config["runner"]

    # configure runner logger
    runner_logger = logging.getLogger(runner)
    runner_logger.setLevel(os.environ.get("LOG_LEVEL") or logging.INFO)
    runner_logger.addHandler(logging.StreamHandler(sys.stdout))

    # delegate execution to runner
    if runner == albo_noiseless_runner.runner:
        albo_noiseless_runner.main(config)
    else:
        raise ValueError(f"Unknown runner: {runner}")