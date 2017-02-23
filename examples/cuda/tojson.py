#!/usr/bin/env python

import json

import convolution_streams as kernel

results = kernel.tune()

with open("convolution_streams.json", 'w') as fp:
    json.dump(results, fp)
