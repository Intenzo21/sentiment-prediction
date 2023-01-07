#!/usr/bin/env bash
# Exit on error
set -o errexit

pip install -U pip
pip install wheel	# Solution to the deprecation warning when installing bs4 with setup.py install because the wheel package is missing
pip install -r requirements.txt