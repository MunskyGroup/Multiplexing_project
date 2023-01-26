
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
This repository contains the codes necessary to reproduce figures from the above manuscript. All codes are implemented in Python


## Graphical Abstract<br/>
![graphical abstract](https://raw.githubusercontent.com/MunskyGroup/Multiplexing_project/master/figures/MP_graphical_abstract.png)
---

## Manuscripts  <br/>

---


## Organization  <br/>

```./datasets ```: [dropbox link]

Machine learning results

		* parsweep_cl_ML:   construct length vs construct lengths
		* parsweep_img_ML:  number of frames vs frame interval
		* parsweep_keki_ML: elongation rate vs initation rate for both mRNAs
		* parsweep_kes_ML:  elongation rate mRNA1 vs elongation rate mRNA2
		* parsweep_kis_ML:  initation rate mRNA1 vs initation rate mRNA2
		* plots:            plotted results of the parameter sweep
		* metadata.yaml:    generation parameters

within each sub folder is 

	* acc_mat_N_.npy: test accuracy matrix across the parameter sweep
	* cl_key.csv:     test accuracy matrix with row and column labels (pandas)

```./ML_F_kis_diff```:  Frequency only classifier applied to varying imaging conditions

```./ML_IF_kis_diff```: Intensity + Frequency classifier applied to varying imaging conditions

```./ML_I_kis_diff```: Intensity only classifier applied to varying imaging conditions

```./ML_run_1280_10s_wfreq```: Classifier applied to 1280s of NCT for 50 cells video with a 10s frame interval (128 frames, 10s fi) for all comparison analyses

```./ML_run_1280_5s_wfreq```: Classifier applied to 1280s of NCT video for 50 cells with a 10s frame interval (256  frames, 10s fi) for all comparison analyses

```./ML_run_3000_2s_wfreq```: Classifier applied to 3000s of NCT video for 50 cells with a 10s frame interval (1500  frames, 2s fi) for all comparison analyses

```./ML_run_320_5s_wfreq```: Classifier applied to 320s of NCT video for 50 cells with a 10s frame interval (64 frames, 5s fi) for all comparison analyses

```./ML_run_tag_3prime```:   Classifier applied to alternate tag design, 10xFLAG on the 3' end KDM5B vs 10xFLAG p300

```./ML_run_tag_base```: Classifier applied to original tag design, 10xFLAG on the 5' end KDM5B vs 10xFLAG p300

```./ML_run_tag_minus5```: Classifier applied to alternate tag design, 5xFLAG on the 5' end KDM5B vs 10xFLAG p300

```./ML_run_tag_plus5```: Classifier applied to alternate tag design, 15xFLAG on the 5' end KDM5B vs 10xFLAG p300

```./ML_run_tag_split```: Classifier applied to alternate tag design, 5xFLAG on the 5' end and 5xFLAG on the 3' end KDM5B vs 10xFLAG p300


Figures and figure generating codes


```./figures_data```: saved data and codes to recreate each figure
```./figures```:  constructed figures in various formats


---


## Example Notebooks <br/>


---

## Large datasets  <br/>

---

## Code organization <br/>

---

## Gene Sequences <br/>


---  

## Code implementation<br/>


 
 ---  

## Cluster implementation<br/>


 ---  
 
