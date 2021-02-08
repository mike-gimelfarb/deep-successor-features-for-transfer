# successor-features-for-transfer
A reusable framework and independent implementation for successor features (SF) for transfer in (deep) reinforcement learning using keras, based on [1].

![](https://github.com/mike-gimelfarb/successor-features-for-transfer/blob/main/source/figures/sfql_cumulative_return_per_task.png)

Currently supports:
- tabular SF representations for discrete environments, based on an efficient hash table representation
- deep neural network SF representations for large or continuous-state environments, based on keras; allows existing keras models or custom architectures (e.g. CNNs) as inputs for easy training and tuning
- tasks with pre-defined state features only, although support for training features on-the-fly may be added later
- tasks structured according to the OpenAI gym framework

# Requirements
- python 3.8 or later
- tensorflow 2.3 or later
- pygame 2.0.1. or later (for some environments)

# References
[1] Barreto, André, et al. "Successor features for transfer in reinforcement learning." Advances in neural information processing systems. 2017.
[2] Dayan, Peter. "Improving generalization for temporal difference learning: The successor representation." Neural Computation 5.4 (1993): 613-624.
