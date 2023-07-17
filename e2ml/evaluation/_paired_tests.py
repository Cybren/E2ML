import numpy as np
import itertools

from sklearn.utils.validation import check_array
from scipy import stats

from ._one_sample_tests import t_test_one_sample


def t_test_paired(sample_data_1, sample_data_2=None, mu_0=0, test_type="two-sided"):
    """Perform a paired t-test.

    Parameters
    ----------
    sample_data_1 : array-like of shape (n_samples,)
        Sample data drawn from a population 1. If no sample data is given, `sample_data_1` is assumed to consist of
        differences.
    sample_data_2 : array-like of shape (n_samples,), optional (default=None)
        Sample data drawn from a population 2.
    mu_0 : float or int
        Population mean assumed by the null hypothesis.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    t_statistic : float
        Observed t-transformed test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data_1 = check_array(sample_data_1, ensure_2d=False)
    if sample_data_2 is not None:
        sample_data_2 = check_array(sample_data_2, ensure_2d=False)
    else:
       sample_data_2 = np.zeros_like(sample_data_1)
    diffs = sample_data_1 - sample_data_2
    return t_test_one_sample(diffs, mu_0=mu_0, test_type=test_type)


def wilcoxon_signed_rank_test(sample_data_1, sample_data_2=None, test_type="two-sided"):
    """Perform a Wilcoxon signed-rank test.

    Parameters
    ----------
    sample_data_1 : array-like of shape (n_samples,)
        Sample data drawn from a population 1. If no sample data is given, `sample_data_1` is assumed to consist of
        differences.
    sample_data_2 : array-like of shape (n_samples,), optional (default=None)
        Sample data drawn from a population 2.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    w_statistic : float
        Observed positive rank sum as test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data_1 = check_array(sample_data_1, ensure_2d=False)
    if sample_data_2 is not None:
        sample_data_2 = check_array(sample_data_2, ensure_2d=False)
    else:
       sample_data_2 = np.zeros_like(sample_data_1)
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")
    diffs = sample_data_1 - sample_data_2
    diffs = diffs[np.where(diffs != 0.0)[0]]
    idx = np.argsort(np.abs(diffs))
    sorted_diffs = diffs[idx]
    ranks = np.arange(1, len(sorted_diffs)+1)

    #ranks = stats.rankdata(sorted_diffs)

    l = []
    i_s = -1
    for i in range(len(sorted_diffs) - 1):
        if([i] == sorted_diffs[i+1]):
            i_s = i
            l.append(ranks[i])
            continue
        if(i_s != -1):
            l.append(ranks[i])
            ranks[i_s:i] = np.mean(np.array(l))
            l=[]
            i_s = -1
    if(i_s != -1):
        l.append(ranks[-1])
        ranks[i_s:-1] = np.mean(np.array(l))
    d_sum = 0
    for i, diff in enumerate(sorted_diffs):
        if(diff > 0):
            d_sum += ranks[i]
    m = len(sorted_diffs)

    if(m > 30):#use normaldistribution cause central-limit
        mean = (m*(m+1))/4
        var = (m*(m+1)*(2*m+1))/24

        z_m = (d_sum - mean)/np.sqrt(var) 

        dist = stats.norm(0,1)
        if(test_type == "right-tail"):
            p_value = 1 - dist.cdf(z_m)
        elif(test_type == "left-tail"):
            p_value = dist.cdf(z_m)
        else:
            p_value = 2 * min(dist.cdf(z_m), 1 - dist.cdf(z_m))
    
    else:
        w_dict = {}
        for comb in itertools.product([0, 1], repeat=m):
            w_statistic_ = np.sum(np.array(comb) * ranks[None,:])
            w_dict[w_statistic_] = w_dict.get(w_statistic_, 0) + 1
        w_stat_arr = np.array(list(w_dict.keys()))
        p_arr = np.array(list(w_dict.values())) / 2**m

        p_left = p_arr[w_stat_arr <= d_sum].sum()
        p_right =  p_arr[w_stat_arr >= d_sum].sum()
        #ergibt dann aber p > 1 für die summe

        if(test_type == "right-tail"):
            p_value = p_right
        elif(test_type == "left-tail"):
            p_value = p_left
        else:
            p_value = 2 * min(p_right, p_left)

    return d_sum, p_value


