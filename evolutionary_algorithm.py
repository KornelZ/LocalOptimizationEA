from __future__ import absolute_import, division, print_function
import numpy as np
import scipy


# ===============================================
# the most basic example solver
# ===============================================

population_size = 100
population_limit = 200
tournament_size = 15
tournament_number = 10
winner_count = 4
loser_count = 2
sigma = 1.0
mu = 0.0
local_optimizer_num = 1

num_success = 0
num_tries = 0

def get_num_success():
    return num_success
def get_num_tries():
    return num_tries
def set_num_success(num):
    global num_success
    num_success = num
def set_num_tries(num):
    global num_tries
    num_tries = num
def set_parameters(_population_size, _population_limit, _tournament_size, _tournament_number, _winner_count,
 _loser_count, _sigma, _mu, _local_optimizer_num):
    global population_size
    population_size = _population_size
    global population_limit  
    population_limit = _population_limit
    global tournament_size 
    tournament_size = _tournament_size
    global tournament_number 
    tournament_number = _tournament_number
    global winner_count 
    winner_count = _winner_count
    global loser_count 
    loser_count = _loser_count
    global sigma 
    sigma = _sigma
    global mu 
    mu = _mu
    global local_optimizer_num 
    local_optimizer_num = _local_optimizer_num


def generate_population(population_size, dim):
    return np.random.rand(population_size, dim)


def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min = len(lbounds), None,
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
            #evaluate points selected in tournament
            F = np.apply_along_axis(fun, 1, T)
            budget -= tournament_size
            if budget <= 0 or fun.final_target_hit == 1:
                break
            #sort according to value of target function
            I_and_F = sorted(zip(I, F), key=lambda x: x[1])
            x_min = X[I_and_F[0][0]]

            #mutate winners of the tournament
            normal_dist = np.random.normal(mu, sigma, (winner_count, dim))
            for p in range(winner_count):
                X[I_and_F[p][0], :] = np.clip(X[I_and_F[p][0], :] + normal_dist[p, :], lbounds, ubounds)
            #apply BFGS to losers of the tournament
            I_and_F = list(reversed(I_and_F))
            for p in range(loser_count):
                res = scipy.optimize.minimize(fun, X[I_and_F[p][0], :], args=(), method='BFGS', jac=None, tol=None, callback=None,
                                              options={'disp': False, 'maxiter': local_optimizer_num, 'maxfun': 1})
                X[I_and_F[p][0], :] = np.clip(res.x, lbounds, ubounds)
                budget -= res.nfev

        population_number = population_number + 1
        if fun.final_target_hit == 1:
            break

    if fun.final_target_hit == 1:
        global num_success
        num_success = num_success + 1
    global num_tries
    num_tries = num_tries + 1
    return x_min
