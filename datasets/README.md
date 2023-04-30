## Data generation for "Using mechanistic models and machine learning to design single-color multiplexed Nascent Chain Tracking experiments"

William S. Raymond, Sadaf Ghaffari, Luis U. Aguilera, Eric Ron, Tatsuya Morisaki,  
Zachary R. Fox , Michael P. May, Timothy J. Stasevich, and Brian Munsky 

---

### Python environment
The python environment used for data generation is stored at: 
```data_making_env.yml```


#### rSNAPed version
The rSNAPed version provided is a custom version and is no up to date with the current package on PyPI and Github. The version used in the paper has been provided in ```/rsnaped/rsnaped.py```

### Data generation
To resimulate all data sets used in the paper call:
			```python master_make_all_data.py```

**WARNING**:  THIS WILL TAKE A LONG TIME TO COMPLETE ALL SIMULATIONS UPWARDS OF 2 WEEKS DEPENDING ON COMPUTER HARDWARE. IT WILL ALSO MAKE 120GB+ OF FILES!

This will remake the following data sets: 

Figure 4:
- P300_KDM5B_24000s_same_intensity_gaussian_14scale
- P300_KDM5B_24000s_similar_intensity_gaussian_14scale
- P300_KDM5B_24000s_different_intensity_gaussian_14scale

Figure 5: 
- par_sweep_5000
- par_sweep_kis
- construct_length_dataset_larger_range_14scale
- par_sweep_kes

Figure 7:
- par_sweep_different_tags_gaussian