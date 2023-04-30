## Machine learning experiments for "Using mechanistic models and machine learning to design single-color multiplexed Nascent Chain Tracking experiments"

William S. Raymond, Sadaf Ghaffari, Luis U. Aguilera, Eric Ron, Tatsuya Morisaki,  
Zachary R. Fox , Michael P. May, Timothy J. Stasevich, and Brian Munsky 

---

### Python environment
The python environment used for data generation is stored at: 
```ML_env.txt```


#### rSNAPed version
The rSNAPed version provided is a custom version and is no up to date with the current package on PyPI and Github. The version used in the paper has been provided in ```/rsnaped/rsnaped.py```

### ML
To resimulate all data sets used in the paper call:
			```python master_run_all_ML.py```

**WARNING**:  This will take a while to run as it trains over 1000 classifiers

This will remake the following machine learning experiments

Figure 2:
- Training data size experiment

Figure 4:
- 24000 second classifiers with different, overlapping, and identical intensity

Figure 5:
- construct length parameter sweep
- ke vs ki 
- ke vs ke
- ki vs ki

Figure 7:
- tagging experiment

Figure 9
- multiplexing experiment

Figure S3
- Photobleaching experiments

Figure S4
- Supplemental sweeps of parameter sets
