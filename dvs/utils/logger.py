import logging
import os
import sys
import typing
from datetime import datetime
from zoneinfo import ZoneInfo

import colorama

# dvs/utils/logger.py


def setup_logger(
    name: typing.Text | logging.Logger,
    *,
    formatter: typing.Optional[logging.Formatter] = None,
    level: typing.Optional[int] = None,
    fmt: typing.Optional[typing.Text] = None,
) -> logging.Logger:
    logger = logging.getLogger(name) if isinstance(name, typing.Text) else name
    logger.setLevel(level or logging.DEBUG)

    # Create formatter similar to uvicorn's default format
    formatter = formatter or ColoredIsoDatetimeFormatter(
        fmt=fmt or "%(asctime)s %(levelname)-8s %(name)s  - %(message)s"
    )

    # Add stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class IsoDatetimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        record_datetime = datetime.fromtimestamp(record.created).astimezone(
            ZoneInfo(os.getenv("TZ", "UTC"))
        )
        # Drop microseconds
        record_datetime = record_datetime.replace(microsecond=0)
        return record_datetime.isoformat()


class ColoredIsoDatetimeFormatter(IsoDatetimeFormatter):
    COLORS = {
        "WARNING": colorama.Fore.YELLOW,
        "INFO": colorama.Fore.GREEN,
        "DEBUG": colorama.Fore.BLUE,
        "CRITICAL": colorama.Fore.RED,
        "ERROR": colorama.Fore.RED,
    }
    MSG_COLORS = {
        "INFO": colorama.Fore.GREEN,
        "WARNING": colorama.Fore.YELLOW,
        "CRITICAL": colorama.Fore.RED,
        "ERROR": colorama.Fore.RED,
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                self.COLORS[levelname] + f"{levelname:8s}" + colorama.Style.RESET_ALL
            )
            record.name = (
                colorama.Fore.LIGHTBLUE_EX + record.name + colorama.Style.RESET_ALL
            )
            if not isinstance(record.msg, typing.Text):
                record.msg = str(record.msg)
            if levelname in self.MSG_COLORS:
                record.msg = (
                    self.COLORS[levelname] + record.msg + colorama.Style.RESET_ALL
                )
        return super(ColoredIsoDatetimeFormatter, self).format(record)
