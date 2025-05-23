import numpy as np
from scipy.spatial import distance
import torch
from torch.autograd import Variable

def statistical_parity(preds, A):
    '''
    ref:
    Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012, January). 
    Fairness through awareness. In Proceedings of the 3rd innovations in theoretical 
    computer science conference (pp. 214-226).
    '''
    sp = preds[A==1].sum()/preds[A==1].shape[0] - preds[A==0].sum()/preds[A==0].shape[0]
    return abs(sp)

def equalized_odds(preds, y, A):
    '''
    ref:
    Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. 
    Advances in neural information processing systems, 29.
    '''
    pos1 = preds[(A==1) & (y==1)].sum()/preds[(A==1) & (y==1)].shape[0]
    neg1 = preds[((A==0)) & (y==1)].sum()/preds[(A==0) & (y==1)].shape[0]
    pos0 = preds[(A==1) & (y==0)].sum()/preds[(A==1) & (y==0)].shape[0]
    neg0 = preds[(A==0) & (y==0)].sum()/preds[(A==0) & (y==0)].shape[0]
    eo = abs(pos1 - neg1) + abs(pos0 - neg0)
    return eo

class MMDStatistic:
    '''
    
    Source for MMD Statistic and all dependent functions: https://github.com/josipd/torch-two-sample/tree/23aa002b1999a853de1b798832255af65114e9d8
    '''

    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.

    The kernel used is equal to:

    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},

    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.

        The kernel used is

        .. math::

            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},

        for the provided ``alphas``.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd
    
    def pval(self, distances, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        if isinstance(distances, Variable):
            distances = distances.data
        return permutation_test_mat(distances.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def generate_explain(explainer, x_train, x_test):
    shap_result = explainer(torch.tensor(x_test, dtype=torch.float32))
    return shap_result

def permutation_test_mat(matrix, n_1, n_2, n_permutations, a00=1, a11=1, a01=0):
    """Compute the p-value of the following statistic (rejects when high)

        \sum_{i,j} a_{\pi(i), \pi(j)} matrix[i, j].
    """
    larger = 0 
    n = n_1 + n_2
    pi = np.zeros(n, dtype=np.int8)
    pi[n_1:] = 1

    for sample_n in range(1 + n_permutations):
        count = 0.
        for i in range(n):
            for j in range(i, n):
                mij = matrix[i, j] + matrix[j, i]
                if pi[i] == pi[j] == 0:
                    count += a00 * mij
                elif pi[i] == pi[j] == 1:
                    count += a11 * mij
                else:
                    count += a01 * mij
        if sample_n == 0:
            statistic = count
        elif statistic <= count:
            larger += 1

        np.random.shuffle(pi)

    return larger / n_permutations

def MMD(male_explain, female_explain):
    mmd_test = MMDStatistic(len(male_explain), len(female_explain))

    if len(male_explain.shape) == 1:
        male_explain = male_explain.reshape((len(male_explain), 1))
        female_explain = female_explain.reshape((len(female_explain), 1))
        all_dist = distance.cdist(male_explain, male_explain, 'euclidean')
    else:
        all_dist = distance.cdist(male_explain, female_explain, 'euclidean')
    median_dist = np.median(all_dist)

    # Calculate MMD.
    _, matrix = mmd_test(torch.autograd.Variable(torch.tensor(male_explain)),
                             torch.autograd.Variable(torch.tensor(female_explain)),
                             alphas=[1 / median_dist], ret_matrix=True)
    p_val = mmd_test.pval(matrix)
    return p_val

def GPF_FAE_metric(X_train, X_test, explainer, sensitive_feature_test, n=100):
    D1 = X_test[sensitive_feature_test == 1, :]
    D2 = X_test[sensitive_feature_test == 0, :]

    if D1.shape[0] < n or D2.shape[0] < n:
        n = min(D1.shape[0], D2.shape[0])
    ## Generate the datasets for evaluating D_1^{'} and D_2^{'}

    # Select n/2 data points from D_1 and D_2, respectively
    temp_D1 = D1[:int(n/2), :]
    temp_D2 = D2[:int(n/2), :]

    # Select the data points that are most similar to them, respectively
    similar_D1 = []
    similar_D2 = []
    for k in np.arange(int(n/2)):
        distances = np.sqrt(np.sum(np.square(D2 - temp_D1[k, :]), axis=1))
        min_index = np.argmin(distances)
        similar_D2.append(D2[min_index, :])

        distances = np.sqrt(np.sum(np.square(D1 - temp_D2[k, :]), axis=1))
        min_index = np.argmin(distances)
        similar_D1.append(D1[min_index, :])
    similar_D1 = np.array(similar_D1)
    similar_D2 = np.array(similar_D2)

    D1_select = np.concatenate((temp_D1, similar_D1))
    D2_select = np.concatenate((similar_D2, temp_D2))

    # Generate FAE explanation result
    D1_select_explain_result = generate_explain(explainer, X_train, D1_select).detach().numpy().astype('float32')
    D2_select_explain_result = generate_explain(explainer, X_train, D2_select).detach().numpy().astype('float32')

    GPF_FAE_result = MMD(D1_select_explain_result, D2_select_explain_result)

    # return D1_select, D2_select, D1_select_explain_result, D2_select_explain_result, GPF_FAE_result
    return GPF_FAE_result