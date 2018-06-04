from __future__ import absolute_import, division, print_function
import numpy as np
import scipy


# ===============================================
# the most basic example solver
# ===============================================
def generate_population(population_size, dim):
    return np.random.rand(population_size, dim)


def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    population_size = 100
    population_limit = 200
    tournament_size = 15
    tournament_number = 10
    winner_count = 2
    loser_count = 2
    sigma = 0.2
    mu = 0.0
    local_optimizer_num = 1

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    population_number = 0
    X = lbounds + (ubounds - lbounds) * generate_population(population_size, dim)
    while budget > 0 and population_number < population_limit:
        for i in range(tournament_number):
            if budget <= 0:
                break
            T = []
            I = []
            for i in range(tournament_size):
                row = np.random.randint(population_size)
                I.append(row)
                T.append(X[row, :])
            T = np.asarray(T)
            F = np.apply_along_axis(fun, 1, T)
            budget -= dim * tournament_size
            if budget <= 0:
                break
            I_and_F = sorted(zip(I, F), key=lambda x: x[1])
            x_min = I_and_F[0][0]
            normal_dist = np.random.normal(mu, sigma, (winner_count, dim))
            for p in range(winner_count):
                X[I_and_F[p][0], :] = X[I_and_F[p][0], :] + normal_dist[p, :]
            I_and_F = list(reversed(I_and_F))
            for p in range(loser_count):
                res = scipy.optimize.minimize(fun, X[I_and_F[p][0], :], args=(), method='BFGS', jac=None, tol=None, callback=None,
                                              options={'disp': False, 'maxiter': local_optimizer_num})
                X[I_and_F[p][0], :] = res.x
                budget -= res.nfev * dim
        population_number = population_number + 1
    return x_min