
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
```data_making_env.yml```


#### rSNAPed version
The rSNAPed version provided is a custom version and is no up to date with the current package on PyPI and Github. The version used in the paper has been provided in ```/rsnaped/rsnaped.py```



### Data generation
To resimulate all data sets used in the paper call:
			```python make_all_data.py```

**WARNING**:  THIS WILL TAKE A LONG TIME TO COMPLETE ALL SIMULATIONS UPWARDS OF 2 WEEKS DEPENDING ON COMPUTER HARDWARE. IT WILL ALSO MAKE 120GB+ OF FILES!

This will remake the following data sets: 

Figure 4:
- P300_KDM5B_24000s_same_intensity_gaussian_14scale
- P300_KDM5B_24000s_similar_intensity_gaussian_14scale
- P300_KDM5B_24000s_different_intensity_gaussian_14scale

Figure 5/ Figure 6/Figure 8: 
- par_sweep_5000
- par_sweep_kis
- construct_length_dataset_larger_range_14scale
- par_sweep_kes

Figure 7:
- par_sweep_different_tags

Figure 9:
- multiplexing_7
- 
Figure S3:
- P300_KDM5B_350s_base_pb

----

Genes used for simulation are stored in ```./variable_length_genes_larger_range``` and ```./rsnaped/DataBases/gene_files```

* ```pUB_SM_KDM5B_PP7_coding_sequence.txt```
* ```pUB_SM_p300_MS2_coding_sequence.txt```
* ```COL3A1.fa```
* ```DOCK8.fa```
* ```EDEM3.fa```
* ```KDM6B.fa```
* ```LONRF2.fa```
* ```MAP3K6.fa```
* ```ORC2.fa```
* ```PHIP.fa```
* ```RRAGC.fa```
* ```TRIM33.fa```
----
