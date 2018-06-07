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
    winner_count = 4
    loser_count = 2
    sigma = 1.0
    mu = 0.0
    local_optimizer_num = 1

    log = open("log.txt", 'a+')
    log.write("Budget {}\n".format(budget))
    log.write("Function {}\n".format(fun))
    log.write("Population size {}, population limit {}\n".format(population_size, population_limit))
    log.write("Tournament size {}, tournament number {}\n".format(tournament_size, tournament_number))
    log.write("Winner_count {}, loser count {}\n".format(winner_count, loser_count))
    log.write("Mu {}, sigma {}, optimization iters {}\n".format(mu, sigma, local_optimizer_num))

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, i_min, x_min, f_min = len(lbounds), None, None, None
    population_number = 0
    X = lbounds + (ubounds - lbounds) * generate_population(population_size, dim)

    while budget > 0 and population_number < population_limit:
        for i in range(tournament_number):
            if budget <= 0 or fun.final_target_hit == 1:
                break
            T = []
            I = []
            for i in range(tournament_size):
                row = np.random.randint(population_size)
                I.append(row)
                T.append(X[row, :])
            T = np.asarray(T)
            F = np.apply_along_axis(fun, 1, T)
            budget -= tournament_size
            if budget <= 0 or fun.final_target_hit == 1:
                break
            I_and_F = sorted(zip(I, F), key=lambda x: x[1])
            x_min = X[I_and_F[0][0]]
            normal_dist = np.random.normal(mu, sigma, (winner_count, dim))
            for p in range(winner_count):
                X[I_and_F[p][0], :] = np.clip(X[I_and_F[p][0], :] + normal_dist[p, :], lbounds, ubounds)
            I_and_F = list(reversed(I_and_F))
            for p in range(loser_count):
                res = scipy.optimize.minimize(fun, X[I_and_F[p][0], :], args=(), method='BFGS', jac=None, tol=None, callback=None,
                                              options={'disp': False, 'maxiter': local_optimizer_num, 'maxfun': 1})
                X[I_and_F[p][0], :] = np.clip(res.x, lbounds, ubounds)
                budget -= res.nfev
        population_number = population_number + 1
        if fun.final_target_hit == 1:
            break

    log.write("Best target reached {}\n".format(fun.final_target_hit))
    print("{}\n".format("Success" if fun.final_target_hit == 1 else "Fail"))
    log.close()
    return x_min