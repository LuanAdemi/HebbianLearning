# HebbianLearning
Recent research suggests that our brain doesn't operate based on a global update rule, as proposed with the gradient descent algorithm, but on a "simple" local update rule. Thus comes the urge to find new and biologically more accurate training mechanisms.

One approach, which I will implement in this notebook, is based on a postulate from Donald Hebb in his book The Organization of Behavior, realeased in 1949. 

## Classic Reinforcement Learning vs Hebbian Learning
Unlike in classical reinforcement learning, our goal is not to learn a static weighted policy network, but a hebbian update rule, which adjusts our network based on the inputs at runtime.

<img src="https://raw.githubusercontent.com/LuanAdemi/HebbianLearning/ee53c6643c48c1cc74d0941ab2b379493403c796/assets/rlvshl.png">

## Hebbian Update Rule
<img src="https://raw.githubusercontent.com/LuanAdemi/HebbianLearning/ee53c6643c48c1cc74d0941ab2b379493403c796/assets/hebbianrule.png">
