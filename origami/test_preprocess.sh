#!/bin/bash

python origami/preprocess.py \
    --inFile="/eos/user/d/dgillber/Omnifold/SlimFeb27/ZjetOmnifold_Feb20_PowhegPythia_mc16d_slim.root" \
    --inTree="OmniTree" \
    --outDir="out_origami/" \
    --label="test" \
    --startEvent 0 --stopEvent 2500
