#!/bin/bash

# this setup script works for MLB on lxplus
# but it won't work for you unless you have a functional copy
# of fastjet in your environment ...

# http://www.fastjet.fr/
# configure it like
# ./configure --prefix=$PWD/../fastjet-install --enable-pyext
# but after you run this script (it needs the PYTHON and PYTHON_CONFIG variables to be set)

# I also had to put a file called 'python config' in <venv>/bin/python-config, the contents are here:
# https://gist.github.com/mattleblanc/360d4ea510d63ab1e2270d91410f7307
# fastjet would not compile without this being in place

# good luck!!

lsetup "lcgenv -p LCG_98python3 x86_64-centos7-gcc8-opt ROOT"
source omnifold/bin/activate
export PYTHON=/afs/cern.ch/work/m/mleblanc/fractals/ATLASOmniFold/omnifold/bin/python
export PYTHON_CONFIG=/afs/cern.ch/work/m/mleblanc/fractals/ATLASOmniFold/omnifold/bin/python-config
export CPLUS_INCLUDE_PATH=/usr/include/python3.6m/:$CPLUS_INCLUDE_PATH
export PYTHONPATH=/afs/cern.ch/user/m/mleblanc/afs/fractals/ATLASOmniFold/fastjet-install/lib/python3.7/site-packages/:$PYTHON_PATH
