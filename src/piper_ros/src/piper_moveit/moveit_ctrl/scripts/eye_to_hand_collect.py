#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed external camera, moving target. Never commands robot motion."""
import sys

from piper_eye_to_hand.collection import main

if __name__ == "__main__":
    try:
        sys.exit(main(mode="eye_to_hand"))
    except KeyboardInterrupt:
        pass
