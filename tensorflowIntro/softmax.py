import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    
    npmat = np.matrix(x)
 
    for c in npmat:
        for i in c:
           print(i) 
    
    return npmat

logits = [1.0, 2.0, 3.0]
print softmax(logits)