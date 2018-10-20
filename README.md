# Active Learning with Gaussian Processes for Crop Phenotyping
Plant phenotyping evaluates crops based on physical characteristics to support plant breeding and genetics activities. Since the current standard practice in collecting phenotype data needs human specialists to assess thousands of plants, phenotyping is currently the bottleneck in the plant breeding process. To increase efficiency of phenotyping, high throughput phenotyping (HTP) uses sensors and robotic platforms to gather plant phenotype data. Due to its limited on-board sample storage capacity, time and energy constraints, a robot can not exhaustively collect data from all the plants in a field, but instead has to select the most informative ones to learn the phenotype distribution via statistical models like Gaussian Processes. In this project, we develop an active learning algorithm with Gaussian Processes to enable a robotic system to actively gather measurements from the field without exhaustively covering the entire field. 

## Installation

### Requirements: 
* [GPytorch](https://github.com/cornellius-gp/gpytorch) (Beta Release)
* [Networkx](https://networkx.github.io/)
* [Seaborn](https://seaborn.pydata.org/)
* [Pandas](https://pandas.pydata.org/)

After installing the listed dependencies, simply clone this package to run experiments.

## Getting Started

See `run.py` script to setup a simulation environment and run an agent to collect data adaptively. For example, 

```
python run.py --eval_only --render
```
simulates an agent moving in the field to gather samples in an informative manner. The simulation will use a randomly generated mixture of Gaussian dataset. Please contact us if you need the sorghum field data. 

## Results



## Contact

For any queries, feel free to raise an issue or contact me at sumitsk@cmu.edu.

<!-- <img width="100" src=images/cmu_logo.png /> -->
![](images/cmu_logo2.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This work was supported in part by U.S. Department of Agriculture
(DOA) award 2017-6700-726152.
