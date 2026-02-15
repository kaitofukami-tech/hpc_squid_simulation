#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for backward compatibility.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("gmlp_project.build_denoise_datasets", run_name="__main__")
