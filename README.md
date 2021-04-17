# successor-features-for-transfer
A reusable framework and independent implementation for successor features (SF) for transfer in (deep) reinforcement learning using keras, based on [1].

Tabular setting:

![](https://github.com/mike-gimelfarb/successor-features-for-transfer/blob/main/source/figures/sfql_return.png)

Deep learning setting:

![](https://github.com/mike-gimelfarb/successor-features-for-transfer/blob/main/source/figures/sfdqn_return.png)

Currently supports:
- tabular SF representations for discrete environments, based on an efficient hash table representation
- deep neural network SF representations for large or continuous-state environments, based on keras; allows existing keras models or custom architectures (e.g. CNNs) as inputs for easy training and tuning
- tasks with pre-defined state features only, although support for training features on-the-fly may be added later
- tasks structured according to the OpenAI gym framework

# Requirements
- python 3.8 or later
- tensorflow 2.3 or later
- pybullet 3.0.8 and pybullet-gym 0.1 (for reacher domain)

# References
[1] Barreto, André, et al. "Successor features for transfer in reinforcement learning." Advances in neural information processing systems. 2017.
[2] Dayan, Peter. "Improving generalization for temporal difference learning: The successor representation." Neural Computation 5.4 (1993): 613-624.
