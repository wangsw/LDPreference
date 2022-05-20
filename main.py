__author__ = 'westbrick'

import simulator

from setting_random import *


simulation = simulator.Simulator()
#print("main", na, nd)
simulation.init(n, ds, values, eps, repeat, mechanisms, permutations, na, nd)
simulation.simulate()
simulation.write('random_u'+str(n)+'_d'+"-".join(map(str, ds))+'_rule'+"-".join(map(str, values[0]))+'_na'+"-".join(map(str, na))+'_nd'+"-".join(map(str, nd))+".json")


#testing = tester.Tester()
#testing.init(n, ds, ms, eps, repeat, mechanisms, dist)
#testing.test()
#testing.write('uniform_10000')



