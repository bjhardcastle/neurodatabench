"""Low-overhead logging helpers shared by the runner and implementations.

The runner installs an asynchronous DEBUG file sink for the package logger.  An
implementation should use :func:`get_logger` instead of attaching its own
handler or writing diagnostic messages to stdout.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
from pathlib import Path

_PACKAGE_LOGGER_NAME = "neurodatabench"


class LoggingSession:
    """Own the asynchronous file logging resources for one benchmark run."""

    def __init__(
        self,
        package_logger: logging.Logger,
        queue_handler: logging.handlers.QueueHandler,
        console_handler: logging.Handler,
        listener: logging.handlers.QueueListener,
        previous_propagate: bool,
    ) -> None:
        """Create a session that owns the queue handler and listener."""
        self._package_logger = package_logger
        self._queue_handler = queue_handler
        self._console_handler = console_handler
        self._listener = listener
        self._previous_propagate = previous_propagate
        self._closed = False

    def close(self) -> None:
        """Flush queued records and close the file handler."""
        if self._closed:
            return
        self._closed = True
        self._package_logger.removeHandler(self._queue_handler)
        self._package_logger.removeHandler(self._console_handler)
        self._package_logger.propagate = self._previous_propagate
        self._listener.stop()
        for handler in self._listener.handlers:
            handler.close()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an implementation logger captured by the runner's log file.

    Names outside the package are placed below ``neurodatabench.implementations``
    so scripts executed as ``__main__`` are captured as well.
    """
    if not name or name == "__main__":
        name = "implementation"
    if name == _PACKAGE_LOGGER_NAME or name.startswith(f"{_PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.implementations.{name}")


def configure(
    *,
    console_level: str,
    file_path: Path | None = None,
) -> LoggingSession | None:
    """Configure package logging and optionally start an asynchronous DEBUG sink.

    File records are queued in memory and written by a daemon listener thread,
    keeping file I/O out of the benchmark thread.  Queueing still has a small
    cost when DEBUG logging is enabled; no logging design can make emitted
    diagnostic calls literally free.
    """
    level = logging.getLevelNamesMapping()[console_level]
    logging.basicConfig(level=level)
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(logging.DEBUG if file_path is not None else level)
    if file_path is None:
        return None
    previous_propagate = package_logger.propagate
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    package_logger.addHandler(console_handler)
    package_logger.propagate = False

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    queue_handler = logging.handlers.QueueHandler(records)
    queue_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(queue_handler)
    listener = logging.handlers.QueueListener(records, file_handler)
    listener.start()
    return LoggingSession(
        package_logger,
        queue_handler,
        console_handler,
        listener,
        previous_propagate,
    )
