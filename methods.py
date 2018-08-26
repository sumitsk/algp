import numpy as np
from utils import entropy_from_cov, manhattan_distance


# greedily select samples from the unvisited locations
def greedy(X, sampled, cov, num_samples, pose=None, locs=None, max_distance=None, utility='entropy', entropy_constant=None):
	'''
	X - set of all possible sensing locations
	sampled - boolean vector indicating which locations have been sampled
	cov - K[X,X] covariance matrix of all locations
	num_samples - number of samples to collect
	utility - entropy/mutual information
	'''

	# if all three are provided, then distance constraint will be applied
	distance_constraint = pose is not None and locs is not None and max_distance is not None

	n = len(X)
	cumm_utilities = []
	new_samples = []
	ent_v = entropy_from_cov(cov[sampled].T[sampled].T, entropy_constant)

	for _ in range(num_samples):
		utilities = np.full(n, -np.inf)
		cond = ent_v + sum(cumm_utilities)

		for i in range(n):
			# if entropy is monotonic, then this step is not necessary 
			if sampled[i]:
				continue
			if distance_constraint:
				if manhattan_distance(pose, locs[i]) > max_distance:
					continue
			org = sampled[i]
			sampled[i] = True

			# a - set of all sampled locations 
			cov_a = cov[sampled].T[sampled].T
			ent_a = entropy_from_cov(cov_a, entropy_constant)
			utilities[i] = ent_a - cond
			sampled[i] = org

		# NOTE: there can be multiple samples with the same utility (ignoring that for now)
		best_sample = np.argmax(utilities)
		cumm_utilities.append(utilities[best_sample])
		new_samples.append(best_sample)
		sampled[best_sample] = True

	return new_samples, cumm_utilities