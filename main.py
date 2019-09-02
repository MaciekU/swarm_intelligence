import numpy as np
from math import exp
from timeit import default_timer as timer

#Format
res='\x1b[0m'
op1 = '\x1b[32m'  # green
op2 = '\x1b[36m'  # cyan



def Func01(x,y):
	#eq = (x-50)**2 + (y-100)**2  # optimals x=50 y=100
	#eq = 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))  # Rastrigin x=0;y=0;v=0
	eq = x**2 + y**2  # sphere x=0;y=0;v=0
	return(eq) 

def Random_spread():
	# random starting location 
	particle_loc = np.random.uniform(-1,1,[swarm_len,2])
	# random starting velocity 
	particle_vel = np.random.uniform(-1,1,[swarm_len,2])

	best_val = Func01(particle_loc[:,0],particle_loc[:,1]) # starting values
	particle_best_location = np.copy(particle_loc) #starting locations

	global_best_value = np.min(best_val) # best value from all particles
	global_best_location = particle_loc[np.argmin(best_val)].copy() # best location from all
	return particle_loc, particle_vel, best_val, particle_best_location, global_best_value, global_best_location
								 
#https://en.wikipedia.org/wiki/Particle_swarm_optimization
#http://www.alife.pl/optymalizacja-rojem-czastek
#szukanie miniumum !!!
class PSO():
	def __init__(self,particle_loc, particle_vel, best_val, particle_best_location, 
		global_best_value, global_best_location,
		swarm_len, epochs, inertia, local_weight, global_weight, max_speed):
		self.particle_loc = particle_loc
		self.particle_vel = particle_vel
		self.best_val = best_val
		self.particle_best_location = particle_best_location
		self.global_best_value = global_best_value
		self.global_best_location = global_best_location
		self.swarm_len = swarm_len
		self.epochs = epochs
		self.inertia = inertia
		self.local_weight = local_weight
		self.global_weight = global_weight
		self.max_speed = max_speed

	def run(self):
		for _ in range(self.epochs):
			for par in range(self.swarm_len):
				for dim in range(2): #2D xy		
					error_local_best = self.particle_best_location[par,dim] - self.particle_loc[par,dim]
					error_global_best = self.global_best_location[dim] - self.particle_loc[par,dim]
					
					# ped = masa*predkosc			
					momentum = self.inertia*self.particle_vel[par, dim] + self.local_weight*np.random.uniform(0,1)*error_local_best + self.global_weight*np.random.uniform(0,1)*error_global_best
	   
					if momentum > self.max_speed:
						momentum = self.max_speed
					if momentum < 0.-self.max_speed:
						momentum = 0.-self.max_speed

					self.particle_loc[par,dim] = self.particle_loc[par,dim] + momentum
					self.particle_vel[par,dim] = momentum

				check_par = Func01(self.particle_loc[par,0],self.particle_loc[par,1])

				if check_par < best_val[par]: 
					best_val[par] = check_par
					self.particle_best_location[par,:] = self.particle_loc[par,:].copy()

				if check_par < self.global_best_value:
					self.global_best_value = check_par
					self.global_best_location = self.particle_loc[par,:].copy()

			print(f'Best find for x:{op1}{self.global_best_location[0]:.2f}{res} y:{op1}{self.global_best_location[1]:.2f}{res} value:{op2}{self.global_best_value:.2f}{res}')


class firefly:   
	def __init__(self, swarm_len, func_02, epochs, attr=0.9, light_lower=0.2):
		self.swarm_len = swarm_len 
		self.func_02 = func_02 
		self.epochs = epochs #
		self.agents = [] 
		self.global_best = 0
		self.attr = attr # attraction
		self.light_lower = light_lower # light absorption
		
	def move(self, x, y, t, alpha):
		r = np.linalg.norm(self.agents[x] - self.agents[y])

		dlu = exp(-(self.light_lower * r**2))
		beta = self.attr
		self.agents[x] = self.agents[y] + beta * dlu * r + alpha * np.random.normal(0, 1.0)
	
	#def random_move(self, ran):
		#self.agents[ran] += np.random.normal(0, 1.0)
		
	def run(self, agents_list, alpha_v = 0.1):		
		self.agents = agents_list 
		self.swarm_len = len(self.agents)
		
		current_best = self.agents[np.array([self.func_02(x,y) for x,y in self.agents]).argmin()]
		self.global_best = current_best
		
		for t in range(self.epochs):
			
			for x in range(self.swarm_len):
				
				light_intensities = [self.func_02(agentx,agenty) for agentx,agenty in self.agents]
				
				for y in range(self.swarm_len):
					
					if light_intensities[x] > light_intensities[y]:
						alpha = 0.9**t
						self.move(x, y, t, alpha)
					#else:
						#self.random_move(x)
					
			current_best = self.agents[np.array([self.func_02(x,y) for x,y in self.agents]).argmin()]

			check_current = self.func_02(current_best[0],current_best[1])
			check_global = self.func_02(self.global_best[0],self.global_best[1])

			if check_current < check_global:
				self.global_best = current_best


			print1=self.global_best[0]
			print2=self.global_best[1]
			print3=self.func_02(self.global_best[0],self.global_best[1])
			print(f'Best find for x:{op1}{print1:.2f}{res} y:{op1}{print2:.2f}{res} value:{op2}{print3:.2f}{res}')


#https://arxiv.org/pdf/1702.03389.pdf
class whale:
	def __init__(self, swarm_len, func_02, total_iterations, usound=2,
				 message_distortion_probability=0.1):
		self.swarm_len = swarm_len
		self.func_02 = func_02 
		self.total_iterations = total_iterations
		self.agents = [] 
		self.global_best = 0
		self.usound = usound #ultrasound intens
		self.message_distortion_probability = message_distortion_probability 

	def run(self, agents_list):		
		self.agents = agents_list 
		self.swarm_len = len(self.agents)
		
		current_best = self.agents[np.array([self.func_02(x,y) for x,y in self.agents]).argmin()]
		self.global_best = current_best
		
		for t in range(self.total_iterations):
			new_agents = self.agents
			for i in range(self.swarm_len):
				y = self.better_and_nearest_whale(i)
				if y:
					short_e = np.exp(-self.message_distortion_probability * self.whale_dist(i, y))
					new_agents[i] += np.dot(np.random.uniform(0, self.usound * short_e),
						np.asarray(self.agents[y]) - np.asarray(self.agents[i]))
			self.agents = new_agents

			current_best = self.agents[np.array([self.func_02(x,y)for x,y in self.agents]).argmin()]

			check_current = self.func_02(current_best[0],current_best[1])
			check_global = self.func_02(self.global_best[0],self.global_best[1])

			if check_current < check_global:
				self.global_best = current_best


			print1=self.global_best[0]
			print2=self.global_best[1]
			print3=self.func_02(self.global_best[0],self.global_best[1])
			print(f'Best find for x:{op1}{print1:.2f}{res} y:{op1}{print2:.2f}{res} value:{op2}{print3:.2f}{res}')
		


	def whale_dist(self, i, j):
		return np.linalg.norm(np.asarray(self.agents[i]) - np.asarray(self.agents[j]))

	def better_and_nearest_whale(self, u):
		temp = float('inf')

		v = None
		for i in range(self.swarm_len):
			if self.func_02(self.agents[i][0],self.agents[i][1]) < self.func_02(self.agents[u][0],self.agents[u][1]):
				dist_iu = self.whale_dist(i, u)
				if dist_iu < temp:
					v = i
					temp = dist_iu
		return v



if __name__ == "__main__":
	swarm_len = 100 # amounts of particles
	epochs = 30

	inertia = 1 # bezwladnosc czastki (fizyczna interpretacja: masa)	
	local_weight = 1 # współczynniki dążenia do lokalnego maksimum
	global_weight = 4 # globalnego maksimum				  
	max_speed = 1					

	start = timer()
	particle_loc, particle_vel, best_val, particle_best_location, global_best_value, global_best_location = Random_spread()

	pso = PSO(particle_loc, particle_vel, best_val, particle_best_location, 
				global_best_value, global_best_location,
				swarm_len, epochs, inertia, local_weight, global_weight, max_speed)

	pso.run()
	end = timer()
	print(end - start)


	print('\nfirefly')
	start = timer()
	particle_loc, particle_vel, best_val, particle_best_location, global_best_value, global_best_location = Random_spread()
	fire = firefly(swarm_len, Func01, epochs)
	fire.run(particle_loc)
	end = timer()
	print(end - start)

	print('\nwhale')
	start = timer()
	particle_loc, particle_vel, best_val, particle_best_location, global_best_value, global_best_location = Random_spread()
	wh=whale(swarm_len, Func01, epochs)
	wh.run(particle_loc)
	end = timer()
	print(end - start)
	



