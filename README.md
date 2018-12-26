# Vanderbilt Football Sports Writing Data Collection and Feature Extraction

This project is an example on how to implement computational linguistics via n-gram feature vector analysis programmatically. We computationally design kernels that allow machine learning algorithms to learn from string data for the purposes of eventual pattern recognition systems. This project currently contains instance-based/kernel-based algorithms which perform text analysis and natural language processing. In this PyDev project, we have two data sets (CASIS-25 & VANDERBILT SEC Sports Writers). The CASIS-25 data set has been extensively utilized in past and on-going research, so it will serve as the baseline performance measure by which we will test against the SEC Sports writer data set following our computations. Both data sets are contained in the 'DataSets' source folder and are named accordingly. 

The directory/PyDev package 'CharUnigram' contains an instance-based learner called 'unigramAnalyzer' which performs raw and normalized feature vector extraction on ASCII decimal values 32[inclusive] - 127[exclusive].

The directory/PyDev package 'KernelBasedLearners' contains 4 kernel-based machine learning modules which represent separate baselines which allows us to implement 4 different algorithms to ascertain the most efficient machine learning algorithm.

![GRNN CASIS PLOT](https://github.com/zedtran/PyMachineLearning/blob/master/KernelBasedLearners/GRNN_OUTPUT/CASIS-25/grnn_casis_plot.png)

![GRNN VANDERBILT PLOT](https://github.com/zedtran/PyMachineLearning/blob/master/KernelBasedLearners/GRNN_OUTPUT/VANDERBILT/grnn_vanderbilt_plot.png)  

## Getting Started

Minimally, users require a development environment and Python 2.7. We recommend the installation of Eclipse IDE with the PyDev plugin.

### Prerequisites

Users may download this code-base and file structure as-is, but any changes to the underlying subdirectory of dataset folders will cause I/O errors. It is assumed users know how to accurately specify file paths as variables. For example, the following line points to the "CharUnigramData" directory, "CASIS-25_Dataset" subdirectory, at all .txt files within:

```
'CharUnigramData/CASIS-25_Dataset/*.txt'
```

### Installing

Ensure the entire code base and dataset are downloaded as-is and placed in the appropriate workspace for your IDE.

## Contributing

Please see the [list](https://github.com/zedtran/PyMachineLearning/settings/collaboration) of collaborators.

## Authors

* **Donald Tran** - *Initial & Continuing work* - [zedtran](https://github.com/zedtran)
* **Josh Gaston** - *Initial work/Consult* - [jgaston93](https://github.com/jgaston93)
* **Thaddeus Hatcher** - *Reviewer* - [ThaddeusHatcher](https://github.com/ThaddeusHatcher)
* **Ben Petersen** - *Reviewer* - [BenPetersen37](https://github.com/BenPetersen37)
* **Cole McKiever** - *Reviewer* - [CBMcKiever](https://github.com/CBMcKiever)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

* **See [license.txt](https://github.com/zedtran/PyMachineLearning/blob/master/license.txt)