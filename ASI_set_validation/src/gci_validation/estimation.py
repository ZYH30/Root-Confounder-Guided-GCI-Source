"""Common nonlinear outcome regression for all covariate specifications."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

def estimate_effect(train, test, adjustment):
    if adjustment is None:
        return None, None
    columns = ['t',*adjustment]
    model = make_pipeline(StandardScaler(),PolynomialFeatures(degree=3,include_bias=False),Ridge(alpha=1.0))
    model.fit(train[columns],train.y)
    factual = test[columns].copy(); counterfactual = factual.copy()
    counterfactual.t = counterfactual.t-1.0
    prediction = model.predict(factual)-model.predict(counterfactual)
    truth = (test.y-test.y_delta).to_numpy()
    error=prediction-truth
    return {'rmse':float(np.sqrt(np.mean(error**2))), 'bias':float(error.mean()),
            'mae':float(np.abs(error).mean())}, prediction
