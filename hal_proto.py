"""Backward-compatible entry point; implementation lives in the ``hal`` package."""

from hal import *  # noqa: F401,F403
from hal.__main__ import main


if __name__ == "__main__":
    main()
