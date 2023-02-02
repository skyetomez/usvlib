# usvlib

This is the underlying library for the rat_filter project. It consists of functions to filter and prepare the spectrograms and scalograms used as inputs to the neural network.


## Motivation
The long-term goal of this project is to decipher animal communication to better understand human communication.

## Build Status
This library is currently being built and optimized. 

TODO:
* construction of spectorgram generator object
	* writing complete and no debugging for speed and running on cluster
* testing of scalogram generator objects
    * writting for parallel processing 
* general optimization of array using vectorization 
    * need to use snakeviz or another profiler; not comptaible with our jupyterlab notebook 

## Installation

```setup
pip install -r requirements.txt
```

## How to Use

1. Install using the installation instructions.
2. Add libary to path using the builtin `sys` library or add it to your project folder
3. Import like any other library. 


## Contirbute
This is an open source library; feel free to fork and edit.

## Credits
scipy
numpy
PyWavelets

## Thanks

Special thanks to Marco, general python community, and Moorman lab
