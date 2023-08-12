import logging
import yaml

from albo.utils.argparse_utils import add_override_arguments, apply_updates, format_nested, replace_env_vars

runner = "albo-noiseless"
logger = logging.getLogger(runner)


def main(config):
    logger.info(f"Runner: {runner}")
    logger.info(f"Config: {config}")

    # TODO: create an implementation of the runner
