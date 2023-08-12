import logging


runner = "albo-noiseless"
logger = logging.getLogger(runner)


def main(config):
    logger.info(f"Runner: {runner}")
    logger.info(f"Config: {config}")

    # TODO: create an implementation of the runner
