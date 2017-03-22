#!/bin/bash

mkdir imgs && ls img | xargs -n8 -I% convert img/% -background white -alpha remove imgs/%.png
