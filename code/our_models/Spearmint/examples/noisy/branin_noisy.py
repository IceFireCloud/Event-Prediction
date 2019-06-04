from __future__ import division
from __future__ import print_function
from past.utils import old_div
import numpy as np
import math

def branin(x, y):

    result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + 
         (old_div(5,math.pi))*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
    
    result = float(result)
    noise = np.random.normal() * 50.
    
    print('Result = %f, noise %f, total %f' % (result, noise, result+noise))
    #time.sleep(np.random.randint(60))
    return result + noise

# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return branin(params['x'], params['y'])
