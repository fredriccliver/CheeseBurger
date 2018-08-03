# CheeseBurger Machine Learning Library

Start : 28 Jun, 2018

## Main point of Algorithm.
- Need to develop the regression
- Making prototype of classifier is done.
- It deals all feature as categorical, not continuos.
- Entropy calcuation has contained. entropy of each feature is important to predicting.

## Order of Prediction.

1. feature **scaling** (not need scaling when calculation entropy)
2. calculate **entropy** (entropy is weight.)
3. calculate **feature weight** (some feature is how many important than others)
4. summary data point by level, stack to **the recipe** (the model)
5. call predict function with test data
6. get the suited point of every feature as test data's value from recipe (the making of **Burger Matrix**)
7. product **weight vector**(in 3.) **Burger Matrix**(in 6.)
8. find the best probable class.

---

## How to use this

See the practice file, know the using CheeseBurger Library simply.
> /example/titanic-cheeseburger.py

And compare with 'decision Tree' algorithm.
> /example/titanic-decision-tree.py
