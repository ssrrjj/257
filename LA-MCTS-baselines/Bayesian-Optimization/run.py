# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
#from functions import *
from functions.functions import *
from functions.mujoco_functions import *
import argparse
import os
from skopt import gp_minimize
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--iterations', type=int, help='specify the iterations to collect in the search')


args = parser.parse_args()
class NegSimmer:
    def __init__(self):
        self.swimmer = Swimmer()
        self.lb = self.swimmer.lb
        self.ub = self.swimmer.ub
    def __call__(self, x):
        return -self.swimmer(x)

f = None
iteration = 0
text_file = "bo_"+args.func+str(args.iterations) + ".txt"
img_file = "bo_"+args.func+str(args.iterations)  + ".png"
if args.func == 'ackley':
    assert args.dims > 0
    f = Ackley(dims =args.dims)
elif args.func == 'levy':
    f = Levy(dims = args.dims)
elif args.func == 'lunar': 
    f = Lunarlanding()
elif args.func == 'swimmer':
    f = NegSimmer()
elif args.func == 'hopper':
    f = Hopper()
else:
    print('function not defined')
    os._exit(1)

#assert args.dims > 0
assert f is not None
assert args.iterations > 0

lower = f.lb
upper = f.ub

bounds = []
for idx in range(0, len(f.lb) ):
    bounds.append( ( float(f.lb[idx]), float(f.ub[idx])) )

res = gp_minimize(f,                          # the function to minimize
                  bounds,                     # the bounds on each dimension of x
                  acq_func="EI",              # the acquisition function
                  n_calls=args.iterations,
                  acq_optimizer = "sampling", # using sampling to be consisent with our BO implementation
                  n_initial_points=40
                  )
vals = -np.array(res.func_vals)
reward = [0]
for i in range(len(vals)):
    reward.append(max(reward[-1], vals[i]))
plt.plot(reward)
plt.savefig(img_file)
np.savetxt(text_file, reward)
#print(res)