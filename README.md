# Active Learning with Gaussian Processes for Crop Phenotyping
Plant phenotyping evaluates crops based on physical characteristics to support plant breeding and genetics activities. Since the current standard practice in collecting phenotype data needs human specialists to assess thousands of plants, phenotyping is currently the bottleneck in the plant breeding process. To increase efficiency of phenotyping, high throughput phenotyping (HTP) uses sensors and robotic platforms to gather plant phenotype data. However, the current practices involve exhaustive data collection and is time-consuming. In this project, we develop an active learning algorithm with Gaussian Processes to enable a robotic system to actively gather measurements from the field in order to learn the phenotype distribution. 

The paper can be found at https://arxiv.org/abs/1901.06803 and will appear at [AAMAS 2019](http://aamas2019.encs.concordia.ca/).

## Installation

### Requirements: 
* [GPytorch](https://github.com/cornellius-gp/gpytorch) (Beta Release)
* [Networkx](https://networkx.github.io/)
* [Seaborn](https://seaborn.pydata.org/)
* [Pandas](https://pandas.pydata.org/)

After installing the listed dependencies, simply clone this package to run experiments.

## Getting Started

See `run.py` script to setup a simulation environment and run an agent to adapively collect data and learn the target distribution. For example, 

```
python run.py --eval_only --render
```
simulates an agent moving in the field to gather samples in an informative manner. The simulation will use a randomly generated mixture of Gaussian dataset. Please contact us if you need the sorghum field data. 

## Results
Here is an example simulation video:
![Active Learing and Planning](imgs/ipp.gif)


## Contact

For any queries, feel free to raise an issue or contact me at sumit.sks4@gmail.com.

<!-- <img width="100" src=https://www.cmu.edu/marcom/brand-standards/images/logos-colors-type/full-color-seal-min.png /> -->
![](imgs/cmu_logo2.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This work was supported in part by U.S. Department of Agriculture
(DOA) award 2017-6700-726152.
