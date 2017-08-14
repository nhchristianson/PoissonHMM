# PoissonHMM
A Python library for working with and training HMMs with Poisson emissions.

There are two classes in this library:

`PHMM` creates a typical HMM with Poisson emissions, where every sequence is assumed to have been generated with the same Poisson parameters - i.e., if the HMM has three states with Poisson means of 1.0, 3.0, and 4.5, every sequence will be generated using those parameters.

`PHMM_d` creates a Poisson-emitting HMM where sequences can be generated with different Poisson parameters. Hence, the parameters are formatted as a nested array, where each subarray is the set of emission parameters for a single sequence, and the length of the overall array is the number of observation sequences you'd like to train. This allows for the training of a PHMM such that the state transition matrix is trained over all observation sequneces, but state magnitudes can differ from sequence to sequence.
