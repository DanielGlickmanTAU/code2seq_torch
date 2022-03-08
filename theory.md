###Attention models on Pattern dataset
attention should work as good as random guess on Pattern
As the node features(x) is not indicative.
Attention averages the nodes of the graph.. but averaging x(random) always results in randomness that is not helpful
in gin that is not the case, as (epsilon) can be used to learn the nodes degree which is helpful for the task.

### Softmax position attention

There is a prob right now of that the network cant ignore nodes which are too far away  

Imagine a graph a -> b -> c    
And we want a to ignore c.    
We have that the edge a->c features is (0.,0.)  
And so the position bias MLP will score it (0*w1 + 0*w2 + bias) = bias  
And it will score a->b as w2 + bias  
  
Now in the attention, we have that att(a,b) = exp(w2 + bias) /(exp(w2+bias) + exp(bias)  ) = exp(w2) / (exp(w2) + 1)  
So for attention to ignore far away nodes this way, large weights need to be learned  

Options to fix it:
1) mask far away nodes(all zero features)..  
The benefit of this is that it is consistent with non full graph implementation, i.e if we implement everything in torch geometric with virtual nodes
2) hope the network will learn large weights
3) deeper net can express larger numbers

