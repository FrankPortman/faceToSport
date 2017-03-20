#!/bin/bash

ls ../imgs/nfl-* | head -n 100 | xargs -P 8 -n 1 python detectFace.py

