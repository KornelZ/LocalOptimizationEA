from __future__ import absolute_import, division, print_function
import numpy as np
import scipy
# ===============================================
# the most basic example solver
# ===============================================
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
    local_optimizer_num = 1
    
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    population_number = 0
    X = []
    for i in range(population_size):
		x = []
		for j in range(dim):
			x.append(np.random.rand(lbounds[j], ubounds[j]))
		X.append(x)
    while budget > 0 && population_number < population_limit:
		for i in range(tournament_number)
			T =[]
			for j in range(tournament_size):
				T.append(np.random.choice(X))
			T = sorted(T, key=fun)
			for p in range(winner_count):
				X = [np.random.normal(sigma, T[p]) if x==T[p] else x for x in X]
			T_r = list(reversed(T))
			for p in range(loser_count):
				res = scipy.optimize.minimize(fun, T_r[p], args=(), method='BFGS', jac=None, tol=None, callback=None, options={'disp': False, 'maxiter': local_optimizer_num})
				X = [res.x if x==T_r[p] else x for x in X]
		population_number = population_number + 1		
		result = sorted(X, key=fun)	
    return result[0]



