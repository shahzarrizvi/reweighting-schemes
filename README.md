# nTupleAnalysis

Python project for analysing ntuples for the H->aa->4g/4y Analysis.

After cloning this repo, we can use this python package in notebooks / code 

cd into the parent ntupleanalysis directory and ```pip install --user --editable .```

This will let us import this package into notebooks and also allow us to change the package and incorporate those changes into the notebooks dynamically

Be sure to add this to the top of the notebooks
```
%load_ext autoreload
%autoreload 2
```

Then we can just do ```from ntupleanalysis import *``` or import ```import ntupleanalysis```

Functionality:

```ntupleanalysis.plot_setup()``` sets up the plotting parameters to ATLAS style

```ntupleanalysis.prepare_dataset_table(filenames, treename, branch_list, col_names, entrysteps=10000, outputype=Event_table)``` Can be used to read in all the events from files into an ```ntupleanalysis.Event_table``` object. ```ntupleanalysis.Event_table.events``` is an Awkward Table with column names as specified. Jagged Array functionality is supported here. Please refer to ```ntupleanalysis/ntupleanalysis/table.py``` for details on all the member functions of the class Event_table.

