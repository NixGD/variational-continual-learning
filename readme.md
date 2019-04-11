# Advanced Machine Learning, Group 2
This is the code repository for the *reproducibility challenge* coursework
undertaken as part of the *Advanced Machine Learning* course, at the University of Oxford.
Group members contributing to this work are:

1. Adam Hillier
2. Marcello De Bernardi
3. Hadrien Pouget
4. Nicholas Goldowsky-Dill

We reproduce and analyze the results of the paper *Variational Continual Learning*,
by Nguyen et al (ICLR 2018), available freely at https://arxiv.org/abs/1710.10628.
## Variational Continual Learning

We have reproduced the three discriminative experiments presented in the paper.
The data from our experiments can be found in TensorBoard log files in the
sub-directory `final_logs/`.

We have two implementations of a Discriminative VCL model: in `models.vcl_nn`, and in
`models.contrib`. The former was our first implementation, has been more thoroughly
tested, and was used to obtain our experimental data; the latter is our attempt at a
cleaner implementation more similar to the PyTorch standard module style.
