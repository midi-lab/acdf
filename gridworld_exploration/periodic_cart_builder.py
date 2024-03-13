import numpy as np




class PeriodicCartBuilder(object):
	def __init__(self, half_period, init_state):
		self.init_state = init_state
		self.half_period = half_period
		self.agent_position = init_state
		self.side_length = 2 * half_period
		self.total_states = self.side_length * 4
		self.template = np.zeros((self.side_length+1, self.side_length+1), dtype='int32')
		self.template[0,:] = 1
		self.template[:,0] = 1
		self.template[-1,:] = 1
		self.template[:,-1] = 1
		self.template[::self.half_period,:] = 2
		self.template[:,::self.half_period] = 2
		self.template[::self.half_period,0] = 3
		self.template[0,::self.half_period] = 3
		self.template[::self.half_period,-1] = 3
		self.template[-1,::self.half_period] = 3
		self.asym_map = {
			0:6,
			1:5,
			2:0,
			3:1,
			4:6,
			5:3,
			6:4,
			7:3
		}
		assert self.agent_position < self.total_states
	def step(self,action):
		assert action in [0,1]

		if (action == 1 and self.agent_position % self.half_period  == 0):
			rel_period= self.agent_position / self.half_period
			self.agent_position  = self.asym_map[rel_period] * self.half_period
			self.agent_position = (self.agent_position + 1) % self.total_states 
			return
		else:
			self.agent_position = (self.agent_position + 1) % self.total_states 
			return

	def img(self):
		model_ = np.array(self.template)
		if (self.agent_position < self.side_length):
			model_[0,self.agent_position] = 9
		elif (self.agent_position < 2*self.side_length):
			model_[self.agent_position % self.side_length,-1] = 9
		elif (self.agent_position < 3*self.side_length):
			model_[-1,-(self.agent_position  % self.side_length + 1)] = 9
		else:
			model_[-(self.agent_position  % self.side_length+ 1),0] = 9
		return model_

	def reset(self): #Note: does not randomize position, just resets it to initial value
		self.agent_position = self.init_state
		return

