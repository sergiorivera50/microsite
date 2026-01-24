---
title: Decision Trees
math: true
draft: "true"
---
- They require very little data preparation, no need for feature scaling or centering at all.

<img src="assets/Pasted image 20250726134321.png" alt="Image" width="300">

Interpreting nodes:

- `samples` consists of how many training instances it applies to.
- `value` attribute tells you how many training instances of each class this node applies to.
- `gini`  measures its *impurity*: a node is “pure” (`gini` = 0) if all training instances it applies to belong to the same class. Gini impurity is computed as:
	$$
	G_{i} = 1 - \sum\limits_{k=1}^{n}p_{i, k}^2
	$$
	where $p_{i, k}$ is the ratio of class $k$ instances among the training instances in the ith node.

**Predicting Class Probabilities**

A decision tree can also estimate the probability that an instance belongs to a particular class $k$. First, it traverses the tree to find the leaf node for this instance, and then it returns the ratio of training instances of class $k$ in this node.

For example, suppose you have a flower with petal length 1.5 and width 1.2. The corresponding leaf node is the depth-2 left node, so the decision tree should output the following probabilities: 0% for *Iris setosa* (0/54), 90.7% for *Iris versicolor* (49/54), and 9.3% for *Iris virginica* (5/54).

Notice that the estimated probabilities would be identical anywhere else in the range of values that lead to that leaf node, if the petals were 6cm long and 1.5cm wide, even though it seems obvious that it would most likely be an *Iris virginica* in this case.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=42)

tree_clf.fit(X, y)
```

**CART Training**

Typically decision trees are trained using the *Classification and Regression Tree* (CART) algorithm:

1. Split the training set into two subsets using a single feature $k$ and a threshold $t_k$ (e.g, “petal length ≤ 2.45cm”) by searching for a pair ($k$, $t_k$) that produces the purest subsets (weighted by their size).
2. Execute Step 1 recursively for the two subsets and stop only if we have reached maximum depth (`max_depth` hyperparameter), or if it cannot find a split that will reduce impurity (there are other hyperparameters that control early stopping like `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, and `max_leaf_nodes`).

As can be seen, CART is a *greedy* algorithm: it searches for an optimum split at the top level, then repeats the process at each subsequent level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a solution that’s reasonably good but not guaranteed to be optimal.

Unfortunately, finding the optimal tree is known to be an *NP-Complete* problem: it requires $O(\exp(m))$ time, making the problem intractable even for small training sets.

**Computational Complexity**

Making predictions requires traversing the decision tree from the root to a leaf. Decision trees are generally approximately balanced, so traversing them requires only going through $O(\log_{2}(m))$ nodes. Since each node only requires checking the value of one feature, the overall prediction complexity is $O(\log_{2}(m))$, independent of the number of features. Hence, predictions are very fast, even when dealing with large training sets.

The training algorithm compares all features on all samples at each node, resulting in a training complexity of $O(n \times m \log_{2}(m))$.

**Entropy**

Another criterion we can use other than Gini impurity is *entropy*. In physics, entropy approaches zero when molecules are still and well ordered. In *Shannon’s information theory*, it measures the average information content of a message: entropy is zero when all messages are identical (a reduction of entropy is often called an *information gain*).

In Machine Learning, entropy is frequently used as an impurity measure: a set’s entropy is zero when it contains instances of only one class.

$$
H_{i}= - \sum\limits_{\substack{k = 1 \\ p_{i, k} \ne 0}}^{n} p_{i, k} \log_{2} (p_{i, k})
$$

The reality is that most of the time it does not matter whether you choose Gini impurity or entropy. Gini impurity is slightly faster to compute, so it is a good default. However, when they differ, Gini impurity tends to isolate the most frequent class in its own branch, while entropy tends to produce slightly more balanced trees[^1].

**Regularization**

Decision tress make very few assumptions about the training data (as opposed to linear models, which assume the data is linear). If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely (most likely overfitting it).

Such a model is often called *nonparametric*, not because it doesn’t have any parameters (it has a lot), but because the number of parameters is not determined prior to training. In contrast, a *parametric* model (such as a linear model), has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting).

To avoid overfitting the training data, you need to restrict the tree’s freedom during training (i.e., apply regularization). Generally, you being by restricting the maximum depth of the decision tree (reducing it regularizes the model, hence reducing the risk of overfitting).

In scikit-learn, you can also use some hyperparameters that regularize the tree like: `min_samples_split` (the minimum number of samples a node must have before it can be split), `min_samples_leaf` (the minimum number of samples a leaf node must have), `min_weight_fraction_leaf`(same as `min_samples_leaf` but expressed as a fraction of the total number of weighted instances), and `max_leaf_nodes` (the maximum number of leaf nodes).

Other regularization algorithms work by first training a decision tree without restrictions, then *pruning* (deleting) unnecessary nodes. A node whose children are all leaf nodes is considered unnecessary if the purity improvement it provides is not statistically significant. Standard statistical tests, such as the $\chi^2$ test (chi-squared test), are used to estimate the probability that the improvement is purely the result of chance (*null hypothesis*). If this probability, called the *p-value*, is higher than a given threshold (typically 5%), then the node is considered unnecessary and its children are deleted. The pruning continues until all unnecessary nodes have been pruned.




[^1]: See Sebastian Raschka’s analysis.
