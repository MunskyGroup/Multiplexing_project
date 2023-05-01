Using mechanistic models and machine learning to design single-color multiplexed Nascent Chain Tracking experiments
=======
----

William S. Raymond<sup>1</sup>, Sadaf Ghaffari<sup>2</sup>, Luis U. Aguilera<sup>3</sup>, Eric M. Ron<sup>1</sup>, Tatsuya Morisaki<sup>4</sup>, Zachary R. Fox<sup>1,5</sup>, Michael P. May<sup>1</sup>, and Timothy J. Stasevich<sup>4,6</sup>, Brian Munsky<sup>1,3,*</sup>

<sub><sup>
1. School of Biomedical Engineering, Colorado State University, Fort Collins, Colorado, USA 
2. Department of Computer Science, Colorado State University, Fort Collins, Colorado, USA 
3. Department of Chemical and Biological Engineering, Colorado State University, Fort Collins, Colorado, USA  
4. Department of Biochemistry and Molecular Biology, Colorado State University, Fort Collins, Colorado, USA 
5. Computational Sciences and Engineering Division, Oak Ridge National Laboratory, Oak Ridge, Tennessee, USA  
6. Cell Biology Unit, Institute of Innovative Research, Tokyo Institute of Technology, Nagatsuta-cho 4259, Midori-ku, Yokohama, Japan
</sup></sub>

For questions about the codes, please contact:  wsraymon@rams.colostate.edu, Luis.aguilera@colostate.edu and brian.munsky@colostate.edu


---

### Python environment
The python environment used for data generation is stored at: 
```ML_env.txt```


#### rSNAPed version
The rSNAPed version provided is a custom version and is no up to date with the current package on PyPI and Github. The version used in the paper has been provided in ```/rsnaped/rsnaped.py```

### ML
To resimulate all data sets used in the paper call:
			```python run_all_ML.py```

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
