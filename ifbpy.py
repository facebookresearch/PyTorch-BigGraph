#!/usr/bin/env python3
import warnings

from libfb.py.ipython_par import launch_ipython


warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)


launch_ipython()
