Evaluation
==========

Evaluator
---------

Unified evaluation class for computing metrics on trained models.

.. autoclass:: evaluate.Evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Metrics Computed
----------------

The Evaluator computes the following metrics:

ACAUC (Primary)
~~~~~~~~~~~~~~~

Average Coverage Area Under Curve - continuous calibration metric.

- **Ideal value**: 1.0 (perfect calibration)
- **< 1.0**: Under-covered (overconfident)
- **> 1.0**: Over-covered (too conservative)

AEPC
~~~~

Average Expected Posterior Coverage - discrete calibration at specific α levels.

LPP
~~~

Log Posterior Probability - measures likelihood quality. Higher is better.

NRMSE
~~~~~

Normalized Root Mean Square Error - parameter estimation accuracy. Lower is better.

ESS
~~~

Effective Sample Size - sample efficiency (SIR task only). Higher is better.
