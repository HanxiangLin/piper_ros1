#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline external-camera calibration. Never publishes TF."""
import sys

from piper_eye_to_hand.calibration_cli import main

if __name__ == "__main__":
    sys.exit(main(mode="eye_to_hand"))
