import numpy as np
import pandas as pd

# https://sungmooncho.com/2012/08/26/pagerank/
# PR(A) = (1-d) + d(PR(T1)/C(T1)) ~

damping_factor = 0.85
N = 7.0
prior = 1.0 / N

page = [
    [ 0 , 1 , 1 , 1 , 1 , 1 , 1 ],
    [ 0 , 0 , 1 , 0 , 1 , 0 , 1 ],
    [ 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
    [ 0 , 0 , 1 , 0 , 0 , 0 , 0 ],
    [ 0 , 0 , 0 , 1 , 0 , 1 , 0 ],
    [ 0 , 0 , 0 , 1 , 0 , 0 , 1 ],
    [ 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
]

page = np.array(page)

node_count  = page.sum(axis=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

for i in range(0, int(N)):
    result = 0

    for j in range(1, int(N+1)):
        if i+1 == j:
            pass
        if page[i] == 1:
            result += prior / node_count[j]