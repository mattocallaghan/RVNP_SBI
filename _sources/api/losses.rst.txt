Loss Functions
==============

SimplifiedPosteriorLoss
-----------------------

The main loss function for RVNP training, combining posterior likelihood,
KL divergence, and shrinkage regularization.

.. autoclass:: losses.SimplifiedPosteriorLoss
   :members:
   :undoc-members:
   :show-inheritance:

MaximumLikelihoodLoss
---------------------

Standard maximum likelihood loss for flow models.

.. autoclass:: losses.MaximumLikelihoodLoss
   :members:
   :undoc-members:
   :show-inheritance:

ShannonLossEmbedding
--------------------

Information-theoretic loss for embedding network training.

.. autoclass:: losses.ShannonLossEmbedding
   :members:
   :undoc-members:
   :show-inheritance:
