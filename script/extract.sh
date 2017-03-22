#!/bin/bash

ls ../imgs/n* | xargs -P 8 -n 1 python detectFace.py
