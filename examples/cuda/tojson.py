#!/usr/bin/env python

import convolution_streams as kernel

results = kernel.tune()

import json

with open("convolution_streams.json", 'w') as fp:
    json.dump(results, fp)
