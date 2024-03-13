import torch
import more_itertools



# Dynamics examples used in paper 

def control_example(states,actions):
	states_denoised = states % 5
	state_noise = states//5
	next_state = torch.zeros_like(states)
	next_state[torch.logical_and(actions == 0 , states_denoised == 0)] = 4
	next_state[torch.logical_and(actions == 0 , states_denoised == 1)] = 1
	next_state[torch.logical_and(actions == 0 , states_denoised == 2)] = 0
	next_state[torch.logical_and(actions == 0 , states_denoised == 3)] = 2
	next_state[torch.logical_and(actions == 0 , states_denoised == 4)] = 4
	next_state[torch.logical_and(actions == 1 , states_denoised == 0)] = 0
	next_state[torch.logical_and(actions == 1 , states_denoised == 1)] = 4
	next_state[torch.logical_and(actions == 1 , states_denoised == 2)] = 1
	next_state[torch.logical_and(actions == 1 , states_denoised == 3)] = 1
	next_state[torch.logical_and(actions == 1 , states_denoised == 4)] = 3
	nn = ((state_noise +  torch.bernoulli(torch.ones(next_state.shape[0],) * 0.25).long() )%2)    *5
	next_state +=  nn
	return next_state


def periodic_example(states,actions):
	states_denoised = states % 5
	state_noise = states//5
	next_state = torch.zeros_like(states)
	next_state[torch.logical_and(actions == 0 , states_denoised == 0)] = 1
	next_state[torch.logical_and(actions == 1 , states_denoised == 0)] = 3
	next_state[states_denoised == 1] = 2
	next_state[states_denoised == 2] = 0
	next_state[states_denoised == 3] = 4
	next_state[states_denoised == 4] = 0
	nn = ((state_noise +  torch.bernoulli(torch.ones(next_state.shape[0],) * 0.25).long() )%2)    *5
	next_state +=  nn
	return next_state



def single_prime_loop(states,actions):
	states_denoised = states % 5
	state_noise = states//5
	next_state = (states_denoised +1)% 5
	next_state[torch.logical_and(actions == 1 , states_denoised == 0)] = 3
	nn = ((state_noise +  torch.bernoulli(torch.ones(next_state.shape[0],) * 0.25).long() )%2)    *5
	next_state +=  nn
	return next_state


def double_prime_loop(states,actions):
	next_state = (states +1)% 5 + 5* (states //5)
	next_state[torch.logical_and(actions == 1 , states == 0)] = 8
	next_state[torch.logical_and(actions == 1 , states == 5)] = 3
	return next_state


# Given all k-step transitions under a certain encoder (for all k < K_max), compute k-step prediction and forward model losses, for each K < K_max

def compute_rep_loss(counts_train, counts_val): # Counts is Batch * K_max * A * S * S
	K_S_S_frequencies_train = counts_train.sum(dim = 2,keepdim =True)
	K_S_S_frequencies_train_safe = torch.clone(K_S_S_frequencies_train)
	K_S_S_frequencies_train_safe[K_S_S_frequencies_train_safe == 0] = 1
	normalized_train = counts_train/K_S_S_frequencies_train_safe
	normalized_train[K_S_S_frequencies_train.expand(-1,-1,counts_train.shape[2],-1,-1) == 0] = 1./counts_train.shape[2]

	normalized_train[normalized_train == 0.] = 0.0000001 # To prevent nan for impossible actions

	crossent =  -(normalized_train.log() * counts_val)
	crossent_loss = crossent.sum(dim=-1).sum(dim=-1).sum(dim=-1)/counts_val.sum(dim=-1).sum(dim=-1).sum(dim=-1)
	crossent_loss = torch.cumsum(crossent_loss,dim=1)/torch.arange(1,crossent_loss.shape[1]+1).unsqueeze(0)
	if (torch.isnan(crossent_loss).any()):
		print(crossent_loss)
		print(crossent)
		print(normalized_train)
		print(K_S_S_frequencies_train_safe)
		print(K_S_S_frequencies_train)
		assert(1==2)
	A_S_frequencies_train = counts_train[:,0].sum(dim=-1,keepdim=True)
	A_S_frequencies_train_safe = torch.clone(A_S_frequencies_train)
	A_S_frequencies_train_safe[A_S_frequencies_train == 0]  =1
	normalized_A_S_train = counts_train[:,0]/A_S_frequencies_train_safe
	normalized_A_S_train[A_S_frequencies_train.expand(-1,-1,-1,counts_train.shape[-1]) == 0] = 1./counts_train.shape[3]

	normalized_A_S_train[normalized_A_S_train == 0] = 0.0000001  # To prevent nan for impossible actions

	forward =  -(normalized_A_S_train.log() * counts_val[:,0])
	forward_loss = forward.sum(dim=-1).sum(dim=-1).sum(dim=-1)/counts_val[:,0].sum(dim=-1).sum(dim=-1).sum(dim=-1)

	if (torch.isnan(forward_loss).any()):
		print(forward_loss)
		assert(1==2)
	return crossent_loss, forward_loss


# Simulate trajectories and return k-step transition frequencies in observation space (X)

def collect_data(initial_states, num_steps, step_function, num_actions, num_states, max_K):
	traces_states = torch.zeros((num_steps+1,initial_states.shape[0]),dtype=torch.int64)
	traces_actions = torch.zeros((num_steps,initial_states.shape[0]),dtype=torch.int64)
	curr_states = initial_states
	for i in range(num_steps):
		if (i%500 == 0):
			print("step "+ str(i))
		actions = torch.randint(num_actions, (initial_states.shape[0],))
		next_states = step_function(curr_states,actions)
		traces_states[i] = curr_states
		traces_actions[i] = actions
		curr_states = next_states
	traces_states[-1] = curr_states
	k_slices = torch.zeros((initial_states.shape[0],max_K,num_actions,num_states,num_states),dtype=torch.int32)
	traces_states= traces_states.permute((1,0))
	traces_actions= traces_actions.permute((1,0))
	for k in range(1,max_K+1):
		linear_indices = traces_actions[:,:-(max_K-1)] * (num_states *num_states) + traces_states[:,:-max_K] * (num_states) + (traces_states[:,k:] if k == max_K   else  traces_states[:,k:-max_K+k])
		flat_scatter = torch.zeros((initial_states.shape[0],  max( num_states *num_states*num_actions,  linear_indices.shape[1])  ),dtype=torch.int32)
		flat_scatter.scatter_add_(1,linear_indices,torch.ones_like(flat_scatter))
		flat_scatter = flat_scatter[:,:num_states *num_states*num_actions]
		k_slices[:,k-1] =  flat_scatter.reshape(initial_states.shape[0],num_actions,num_states,num_states)
	return k_slices

# Tools to vectorize and de-vectorize representations of encoders (where an encoder is a partition of X into sets representing an element of S)

def partition_to_vec(partition, num_states):
	ret = torch.zeros(num_states,dtype=torch.int32)
	for li in range(len(partition)):
		for i in partition[li]:
			ret[i] = li
	return ret

def vec_to_partition(part_vec):
	v = part_vec.max()
	out = []
	for i in range(v+1):
		out.append((part_vec==i).nonzero().flatten().tolist())
	return out

# Given observation k-step transition frequencies, compute losses under all possible encoders. Return lowest-loss encoder for each K and each size of encoded state space, with and without using the forward model
def compute_losses(counts_train, counts_val):
	
	partitions = []
	losses_no_forward = []
	losses_forward  = []
	ns = []

	for partition in more_itertools.set_partitions(range(counts_train.shape[3])):
		if (len(ns)% 500 == 0):
			print("partition "+ str(len(ns)))
		ns.append(len(partition))
		partitions.append(partition_to_vec(partition,counts_train.shape[3]))
		reduced_train_1 = torch.zeros(counts_train.shape[0],counts_train.shape[1],counts_train.shape[2],counts_train.shape[3],len(partition),dtype=torch.int32)
		reduced_val_1 = torch.zeros(counts_val.shape[0],counts_val.shape[1],counts_val.shape[2],counts_val.shape[3],len(partition),dtype=torch.int32)
		for se_1 in range(len(partition)):
			for row_ind_1 in partition[se_1]:
				reduced_train_1[:,:,:,:,se_1] += counts_train[:,:,:,:,row_ind_1]
				reduced_val_1[:,:,:,:,se_1] += counts_val[:,:,:,:,row_ind_1]
		reduced_train_2 = torch.zeros(counts_train.shape[0],counts_train.shape[1],counts_train.shape[2],len(partition),len(partition),dtype=torch.int32)
		reduced_val_2 = torch.zeros(counts_val.shape[0],counts_val.shape[1],counts_val.shape[2],len(partition),len(partition),dtype=torch.int32)
		for se_2 in range(len(partition)):
			for row_ind_2 in partition[se_2]:
				reduced_train_2[:,:,:,se_2,:] += reduced_train_1[:,:,:,row_ind_2,:]
				reduced_val_2[:,:,:,se_2,:] += reduced_val_1[:,:,:,row_ind_2,:]
		losses, fw = compute_rep_loss(reduced_train_2, reduced_val_2)
		losses_no_forward.append(losses)
		losses_forward.append(losses+fw.unsqueeze(-1))
	losses_no_forward = torch.stack(losses_no_forward)
	losses_forward = torch.stack(losses_forward)
	partitions = torch.stack(partitions)
	ns = torch.tensor(ns)


	best_partition = []
	best_losses = []
	best_partition_f = []
	best_losses_f = []
	for n in range(1,counts_train.shape[3]+1):
		partitions_n = partitions[ns == n]
		losses_n = losses_no_forward[ns == n]
		losses_f_n= losses_forward[ns == n]


		best_losses_n, indices = torch.min(losses_n,dim=0)
		best_losses_f_n, indices_f = torch.min(losses_f_n,dim=0)

		best_partition_n = torch.gather(partitions_n, 0,  indices.reshape(indices.shape[0]*indices.shape[1]).unsqueeze(-1).expand(-1,partitions_n.shape[1])  ).reshape(indices.shape[0],indices.shape[1],partitions_n.shape[1])
		best_partition_f_n = torch.gather(partitions_n,0, indices_f.reshape(indices_f.shape[0]*indices_f.shape[1]).unsqueeze(-1).expand(-1,partitions_n.shape[1])  ).reshape(indices_f.shape[0],indices_f.shape[1],partitions_n.shape[1])


		best_partition.append(best_partition_n)
		best_partition_f.append(best_partition_f_n)
		best_losses.append(best_losses_n)
		best_losses_f.append(best_losses_f_n)
	return torch.stack(best_partition).permute(1,0,2,3), torch.stack(best_partition_f).permute(1,0,2,3), torch.stack(best_losses).permute(1,0,2), torch.stack(best_losses_f).permute(1,0,2) # Outputs are Batch * Size of Encoder * K


# determine whether learned encoder is correct

def classify_output(to_classify, gt): # 0 = correct, 1 = too many states, but not incorrect, 2= incorrect. Note, may be invalid if exogenous noise is periodic or nonstochastic
	for se in to_classify:
		for se2 in gt:
			if len(set(se).intersection(set(se2)))!= 0  and not set(se).issubset(set(se2)):
				return 2
	if (len(to_classify) != len(gt)):
		return 1
	else:
		return 0

# Main method: given a number of monte-carlo simulations to run (num_tests), details of dynamics model 
# (num_states, step_func, num_actions, ground_truth), max_K (i.e, maximum K to consider), environment steps to collect per simulation (num_steps),
# and loss tolerance (i.e, will return the smallest-range encoder with loss within a (1+loss_tolerace) factor of the minimum observed loss),
# will run simulations and return two max_K * 3 matrices, representing the success rates with and wothout using a forward model. (See comment on classify_output for interpreting the three columns)
def run_test(num_tests,num_states,num_steps,step_func,num_actions,max_K, loss_tolerace, ground_truth):
	torch.manual_seed(0)
	results = torch.zeros(max_K,3,dtype=torch.int32)
	results_f = torch.zeros(max_K,3,dtype=torch.int32)

	counts_train = collect_data(torch.randint(num_states,(num_tests,)), num_steps, step_func, num_actions, num_states, max_K)
	counts_val = collect_data(torch.randint(num_states,(num_tests,)), num_steps, step_func, num_actions, num_states, max_K)
	best_partition, best_partition_f, best_losses, best_losses_f = compute_losses(counts_train, counts_val)
	for test in range(num_tests):
		for k in range(max_K):
			min_loss= torch.min(best_losses[test,:,k])
			best_n = torch.argmax(  (best_losses[test,:,k] <=(1+loss_tolerace)*min_loss).int()  ) + 1 # returns first -- i.e., smallest, result -- to within 0.1%
			results[k,classify_output(vec_to_partition(best_partition[test,best_n-1,k]), ground_truth)] += 1
			min_loss = torch.min(best_losses_f[test,:,k])
			best_n = torch.argmax(  (best_losses_f[test,:,k] <=(1+loss_tolerace)*min_loss ).int() ) + 1 # returns first -- i.e., smallest, result -- to within 0.1%
			results_f[k,classify_output(vec_to_partition(best_partition_f[test,best_n-1,k]), ground_truth)] += 1
	return results, results_f

# Print results with formatting
def formatted_print(results, results_f ):
	print("AC-State")
	print("K\tminimal_correct\tany_correct")
	for K in range(results.shape[0]):
		print(str(K) + '\t' + str(int(100*results[K,0]/results[K].sum())) + "%\t" +  str(int(100*results[K,:2].sum()/results[K].sum())) + "%")
	print("ADF")
	print("K\tminimal_correct\tany_correct")
	for K in range(results_f.shape[0]):
		print(str(K) + '\t' + str(int(100*results_f[K,0]/results_f[K].sum())) + "%\t" +  str(int(100*results_f[K,:2].sum()/results_f[K].sum())) + "%")


results, results_f = run_test(50,10,100,periodic_example,2,4, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
formatted_print(results, results_f )
torch.save({"results":results, "results_f": results_f}, "periodic_example_100.pth")


results, results_f = run_test(50,10,200,periodic_example,2,4, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
formatted_print(results, results_f )
torch.save({"results":results, "results_f": results_f}, "periodic_example_200.pth")

results, results_f = run_test(50,10,400,periodic_example,2,4, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
formatted_print(results, results_f )
torch.save({"results":results, "results_f": results_f}, "periodic_example_400.pth")

results, results_f = run_test(50,10,800,periodic_example,2,4, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
formatted_print(results, results_f )
torch.save({"results":results, "results_f": results_f}, "periodic_example_800.pth")

results, results_f = run_test(50,10,1600,periodic_example,2,4, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
formatted_print(results, results_f )
torch.save({"results":results, "results_f": results_f}, "periodic_example_1600.pth")


# results, results_f = run_test(50,10,100,control_example,2,5, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "random_example_100.pth")


# results, results_f = run_test(50,10,200,control_example,2,5, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "random_example_200.pth")

# results, results_f = run_test(50,10,400,control_example,2,5, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "random_example_400.pth")

# results, results_f = run_test(50,10,800,control_example,2,5, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "random_example_800.pth")

# results, results_f = run_test(50,10,1600,control_example,2,5, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "random_example_1600.pth")




# results, results_f = run_test(50,10,200,single_prime_loop,2,7, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "single_prime_loop_200.pth")

# results, results_f = run_test(50,10,400,single_prime_loop,2,7, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "single_prime_loop_400.pth")

# results, results_f = run_test(50,10,800,single_prime_loop,2,7, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "single_prime_loop_800.pth")

# results, results_f = run_test(50,10,1600,single_prime_loop,2,7, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "single_prime_loop_1600.pth")


# results, results_f = run_test(50,10,3200,single_prime_loop,2,7, .001, [[0,5],[1,6],[2,7],[3,8],[4,9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "single_prime_loop_3200.pth")


# results, results_f = run_test(50,10,1000,double_prime_loop,2,30, .001, [[0],[1],[2],[3], [4],[5],[6],[7],[8],[9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "double_prime_loop_1000.pth")

# results, results_f = run_test(50,10,2000,double_prime_loop,2,30, .001, [[0],[1],[2],[3], [4],[5],[6],[7],[8],[9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "double_prime_loop_2000.pth")

# results, results_f = run_test(50,10,4000,double_prime_loop,2,30, .001, [[0],[1],[2],[3], [4],[5],[6],[7],[8],[9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "double_prime_loop_4000.pth")

# results, results_f = run_test(50,10,8000,double_prime_loop,2,30, .001, [[0],[1],[2],[3], [4],[5],[6],[7],[8],[9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "double_prime_loop_8000.pth")

# results, results_f = run_test(50,10,16000,double_prime_loop,2,30, .001, [[0],[1],[2],[3], [4],[5],[6],[7],[8],[9]])
# formatted_print(results, results_f )
# torch.save({"results":results, "results_f": results_f}, "double_prime_loop_16000.pth")




