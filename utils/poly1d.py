import statsmodels.api as sm

def poly1d(x,y,alpha=0.05):
    r"""
    Functs: - linear regression with 95% CI band
    """
    X = sm.add_constant(x)
    olsmodel = sm.OLS(y,X)
    est = olsmodel.fit()
    # linear fit line
    y_pred = est.predict(X)
    # ci band
    cibands = est.get_prediction(X).summary_frame()
    y_lower = cibands['mean_ci_lower']
    y_upper = cibands['mean_ci_upper']
    
    return y_pred,y_lower,y_upper