import numpy as np
from scipy.stats import ttest_ind,fisher_exact
from statsmodels.stats.multitest import fdrcorrection

weights = np.load(f'{result_path}')
pct = 85

mean = [np.percentile(weights[i], pct) for i in range(weights.shape[0])]
ps = []
for i in range(weights.shape[-1]):
    dt = ttest_ind(weights[:,i], mean, alternative='greater')
    ps.append(dt[1])
rejected, p_corrected = fdrcorrection(ps, method='n', alpha=0.1)

