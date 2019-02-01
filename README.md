# Machine Learning in Bioinformatics
> Máster en Bioinformática Aplicada a Medicina Personalizada y Salud (Curso 2018-2019)

# Scheduling
- Day 1 (4.02.019):
	- 1/2 Theory
	- 1/2 Practice: Hands-On
- Day 2 (5.02.019):
	- 1/2 Theory
	- 1/2 Practice: Group Project
	
# Theory slides

The theory slides are available [here](resources/slides.pdf).

# Day 1 Practice: Hands-On

We are going to use the `Breast Cancer Data` available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)). More information about this dataset can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names) and [here](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

Go to an empty folder and run the following commands to download the data: 
```bash
mkdir data

wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data -O data/wdbc.data

sed -i '1iid,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst' data/wdbc.data
```

Alternatively, the file is also available [here](data/wdbc.data).

Run the following command to use a Docker image with R and the required libraries already installed: 
```bash
docker run --rm -it -v $(pwd):$(pwd) -e WORK_DIR=$(pwd) singgroup/r-machine-learning R
```

And now, run the following R instructions to set the working directory and load the data file:
```R
setwd(Sys.getenv("WORK_DIR"))
data <- read.csv("data/wdbc.data")
```

The full script to develop during this session is available [here](resources/analysis.R).

# Day 2 Practice: Group Project

In small groups, develop a project working in one of [these datasets](DATASETS.md). Then, elaborate a report (5 pages maximum, including figures and tables), with the following contents (feel free to omit or include sections):

- Objectives
- Materials and methods 
Algorithms, dataset description etc.
- Results
Figures, performance tables, algorithms comparison, etc.

# Additional Resources
- *Ten quick tips for machine learning in computational biology* [[10.1186/s13040-017-0155-3](https://dx.doi.org/10.1186%2Fs13040-017-0155-3)]
- [*LIBSVM -- A Library for Support Vector Machines*](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)