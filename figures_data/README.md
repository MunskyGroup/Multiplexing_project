
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
This subfolder contains all the figure generating codes and data needed to recreate the figure parts.

```./data_pipeline``` - Figure 1, explanation of the simulated NCT video generation process.

```./cell_bgs``` - Real max projection cell backgrounds used for rSNAPed in this experiment

```./data_size_sweep``` -  Figure 2, training data size vs accuracy for given classifier architecture 

```./simulated_concept``` - Figure 3, example of the classification and labelling process of two simulated cells

```./I_IF_F``` - Figure 4 and Figure S2, Comparison of parts of the architecture applied to similar, same, or different intensity conditions all with varying frequency information.

```./parsweep_heatmpas``` - Figure 5, classification under 4 varying parameter conditions, mRNA1 length vs mRNA2 length, ke vs ki, ke vs ke, ki vs ki for 320s of video at a 5s frame rate of simualted P300 vs KDM5B NCT experiments.

```./comparison_sweeps``` - Figure 6, classification performance as a function of video length for all conditions in Figure 5.

```./tagging_changes``` - Figure 7, changing an experiment design to recover classification.

```./wrong_params``` - Figure 8, exploring the effect of incorrect parameter guesses on classification accuracy.

```./multiplexing``` - Figure 9, example multiplexed cell with 90% accuracy labelling 7 different mRNAs with 2 colors.

```./gaussian_generation``` - Figure S1, Generation of new frames of bg video from real cell bgs.

```./pb``` - Figure S3, preliminary exploration of effect of photobleaching and tracking on classification.

```./supfigure_1280s10s``` - Figure 5 but for twice the data.

```./supfigure_3000s2s``` - Figure 5 but for maximum possible amount of data from simulated data.

