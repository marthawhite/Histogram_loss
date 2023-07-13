# Replication

The goal of this problem was to replicate results obtained by Ehsan Imani in his [thesis](https://era.library.ualberta.ca/items/90c26ffa-6eff-4ac6-a011-9699d27d91d0/view/347e81b7-8f26-4acb-9960-044c8a2ee7db/Ehsan_Imani.pdf) and [paper](https://arxiv.org/abs/1806.04613) about the Histogram Loss. It contains old experiment code by Ehsan Imani that has been updated to Python 3. Additionally, it contains new code for running experiments on these datasets with our histogram loss framework to compare to the original results.

## Datasets
 - [CT Position](https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis) (53500, 386) - Predicting the relative location of a CT slice within the body ([Graf *et al.* 2011](https://doi.org/10.1007/978-3-642-23629-7_74))
 - [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (17389, 16) - Predicting the hourly count of rented bikes ([Fanaee-T and Gama 2014](https://doi.org/10.1007/s13748-013-0040-3))
 - [Song Year](http://archive.ics.uci.edu/dataset/203/yearpredictionmsd) (515345, 90) - Predicting the release year of songs from audio features ([Bertin-Mahieux *et al.* 2011](https://ismir2011.ismir.net/papers/OS6-1.pdf))
 - [Pole](https://github.com/EpistasisLab/pmlb) (15000, 49) - Describes a telecommunications problem ([Olson *et al.* 2017](https://doi.org/10.1186/s13040-017-0154-4))

 ## Base Models
 
 The original experiment used MLPs with 4 hidden layers and dropout of 0.05 on the input. The hidden layer sizes are half of the input size, except for bike sharing which uses 64.