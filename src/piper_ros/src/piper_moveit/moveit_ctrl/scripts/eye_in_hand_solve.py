#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline wrist-camera calibration. Never publishes TF."""
import sys

from piper_eye_to_hand.calibration_cli import main as calibration_main


def main(argv=None):
    return calibration_main(argv, mode="eye_in_hand")


if __name__ == "__main__":
    sys.exit(main())
