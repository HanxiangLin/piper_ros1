#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrist-mounted camera, stationary target. Never commands robot motion."""
import sys

from piper_eye_to_hand.collection import main as collect_main


def main(argv=None):
    return collect_main(argv, mode="eye_in_hand")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
