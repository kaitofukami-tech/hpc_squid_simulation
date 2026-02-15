#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for backward compatibility.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("gmlp_project.gmlp_denoise_diff_model", run_name="__main__")
