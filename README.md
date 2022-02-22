# Bayesian-Approach-in-Learning-Based-HSI-Denoising
## Model Architecture
![This is my image](/figures/model1.png)
## Bayesian Motivated Training
<img src="https://latex.codecogs.com/svg.image?\bg_black&space;\begin{equation}\label{eq:5}&space;&space;\begin{cases}&space;&space;&space;\underset{\theta}{\text{min}}~\mathcal{L}(\hat{f},f)\\&space;&space;&space;\text{such&space;that&space;}&space;\hat{f}=\underset{f}{\text{arg&space;min}}&space;\lVert&space;h-f\rVert_p&plus;&space;\lambda\mathcal{R}(f;\theta)\\&space;&space;\end{cases}&space;&space;&space;\end{equation}" title="\bg_black \begin{equation}\label{eq:5} \begin{cases} \underset{\theta}{\text{min}}~\mathcal{L}(\hat{f},f)\\ \text{such that } \hat{f}=\underset{f}{\text{arg min}} \lVert h-f\rVert_p+ \lambda\mathcal{R}(f;\theta)\\ \end{cases} \end{equation}" />

## Training and Test Data
### Washington DC Dataset: https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html
### ICVL Dataset: http://icvl.cs.bgu.ac.il/hyperspectral/
### CAVE dataset: https://www.cs.columbia.edu/CAVE/databases/multispectral/

## Cite the paper with:
### Aetesam, Hazique, Suman Kumar Maji, and Hussein Yahia. "Bayesian Approach in a Learning-Based Hyperspectral Image Denoising Framework." IEEE Access 9 (2021): 169335-169347.
