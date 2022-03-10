

## Content attention: unit testing with and without distance:
* training acc usually starts at about 0.5 and reaches 0.61 after 7-8 epochs.  
Increasing or decreasing embedding dim does not change that(unless decreasing to like 3, which hurts acc)  
  Does not seem to be able to overfit a small dataset, but this may be reasonable, as the transformer only looks at x(random 3 dim vector) and the whole graph, so good memorization is impossible(the same graph can have 2 nodes with same x features but different labels, and the model cannot memorize, as it sees only the complete graph and not the node's enviourments)
* Expected acc **without distance** on **real data, i.e test acc** should be 0.5, as shown in "A generalization of transformers to graph"  
* **Gradeitns seem to be small**


Next actions:
So content attention tests does not immidiatliy tell there is a probelm with learning or the model.
I should, for now, disable the pure content overfit test.

Now I should check the distance content attention.
It should give better results then only content.
Here I should be able to overfit if I am using a stack of only [0,1] as it is as expressive as gin

So my steps will be
0) fix init of AdjStackAttentionWeights.. should not receive use_distance_bias..!
   
1) create a test for only position attention with use_distance .. ! that assign a large negative bias and equal positive weights.   
   That should simulate regular gin, by setting bias to -inf 
    1.1) create a method for mocking the attention bias weights
   

2) create a way to freeze the learning of the positional attention weights



## 6/3

###todo:
-why are there zeros in new_adj??
    in sigmoid, doing unreachable bias = -inf or -inf will work I think

-VERIFY PROPER MASKING IN SOFTMAX TOO(fix_nans function)!!
-maybe -1000 is causing trouble for the gradient and I should just let it be..?
-try norm first
- try single layer(if achives above 0.6, stick with it for debugging)
-look at gin architecture in more detail(batch norms relu etc)



y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

might be the batch norms

norm first is probabliy better

###done:
-mock as relu: about the same thing happens
-work also with jk=last
- emb_dim =1 does not learn
-gnn increases slowly but surly
-see unreachable bias is added to unreablcle really... and so init to -1 or -10. not +1.
-vverify that not mocking unreachable to a negative number leads to worse performance
- see what happens when I do not init  weights and bias to my defaults, but just normally: works just as well
  -mock feedforward to eye


### problem with using gating(sigmoid) - gradients exploding at some point
single layer position+distance :https://www.comet.ml/danielglickmantau/test/2d87d20ab4c64d5aa5d72b6a9d52cdba?experiment-tab=chart&search=grad_norm&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
looks like the gradients are starting small, and on some epoch, ~70 they rapdily increase  
it causes the predictions to change very fast(e,.g)
Evaluating epoch 79...acc: {'acc_0': 0.9931260229132569, 'acc_1': 0.2013888888888889, 'acc': 0.5972574559010729} 
Evaluating epoch 80...acc: {'acc_0': 0.7080196399345335, 'acc_1': 0.890625, 'acc': 0.7993223199672668}
Evaluating epoch 81...acc: {'acc_0': 0.3656301145662848, 'acc_1': 0.9861111111111112, 'acc': 0.675870612838698}

maybe batch norm will fix it................


##Patten - number of neigbours distrubtion
(1/stacks[real_nodes_edge_mask][stacks[real_nodes_edge_mask].sum(dim=-1) > 0].sum(dim=-1)).unique(return_counts=True,sorted=True)

 tensor([ 1.0000, 24.0000, 25.0000, 26.0000, 27.0000, 28.0000, 29.0000, 30.0000,
        31.0000, 32.0000, 33.0000, 34.0000, 35.0000, 36.0000, 37.0000, 38.0000,
        39.0000, 40.0000, 41.0000, 42.0000, 43.0000, 44.0000, 45.0000, 46.0000,
        47.0000, 48.0000, 49.0000, 50.0000, 51.0000, 52.0000, 53.0000, 54.0000,
        55.0000, 56.0000, 57.0000, 58.0000, 59.0000, 60.0000, 61.0000, 62.0000,
        63.0000, 64.0000, 65.0000, 66.0000, 67.0000, 68.0000, 69.0000, 70.0000,
        71.0000, 72.0000, 73.0000, 74.0000, 75.0000, 76.0000, 77.0000, 78.0000,
        79.0000, 81.0000, 82.0000])

torch.Size([59]) tensor([3631,   72,   75,  104,  189,  252,  435,  450,  744,  992, 1089, 1496,
        2660, 2700, 3922, 3686, 4134, 4960, 5330, 6258, 5633, 6908, 7605, 7406,
        7332, 7392, 9114, 8000, 7497, 7020, 6731, 6372, 4950, 5656, 4446, 3828,
        3835, 3840, 2867, 3162, 2142, 1792, 2145, 1122, 1675, 1224, 1173,  420,
         426,  360,  219,  370,  150,  304,  385,  312,  158,   81,  164])

### Should gating work in theory as good as gnn?
using adj_stack = [0,1] ,
setting unreachable_bias to -inf

### Making gating work
When running overfitting test, looks like the acc does improve.
But closer than gnn.. 


----
## 8.3 

###Effects of batch norm 
In Gnn, having batch norm before the relu is paramount. 
Indeed variance seems very high in that intermediate layer.

* Tried running gnn without batch norm in between layer to see if that what may cause the problem in my model...
Gnn is still working fine.. good performance, stable accuracy which raises linearly
  https://www.comet.ml/danielglickmantau/test/480df44051ba45479c8082e11d63377c?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step

* Tried also disabling batch norm in GNN's MLP(and also in between) - now model gets 50%!! 
https://www.comet.ml/danielglickmantau/test/96367b1099bc4109b11255b5e58e2b79?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
  
only mlp batch norm disabled and batch norm between gnn enabled
https://www.comet.ml/danielglickmantau/test/5e260c893fa7426fb6f5b189bbd5b64f

batch norm in mlp enabled and disabled between layers
working!!
https://www.comet.ml/danielglickmantau/test/69ab0eb4723d4aba881be86202d993c5

### Debugging networks by observing gradients

The grad/weight rule does seem reasonable.
With a working gnn training(https://www.comet.ml/danielglickmantau/test/69ab0eb4723d4aba881be86202d993c5?experiment-tab=chart&search=grad%2Fweight&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
the ratio decreases niclely and is in the ballpark of 0.1-0.001

With non working https://www.comet.ml/danielglickmantau/test/5e260c893fa7426fb6f5b189bbd5b64f?experiment-tab=chart&search=grad%2Fweight&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
you can see the ratio of mlp.bias changes(goes up and down) and it is always very low.. 


### 9.3

Debugge hit the wall(in git stash)..
we can see, in the case of line graph. that polyndrome edges are indeed unique(and symetric)

* verfied self attention does not change "padding" nodes, even when they are not zero(later layers)


* running gnn with sigmoid on eps( (1 + torch.sigmoid(self.eps) ) * x)
increases lineary and reaches good performance > 0.84

with a single position attention gating layer , trainign is stable and increases nicly up to 0.7

with 2 layers problems already start to araise.. nice


softmax
    grad_output shape 32,142,142
    zero for zero nodes.. not zeros else where

    grad_input: no zeros
____


###10.3

Want reaL_nodes_edge_mask to contain only entries for which 
stacks.sum(dim=-1) != 0
verfied stacks all zeros entries masks more than mask
True == all(map(lambda tup: not tup[0] or (tup[0] and tup[1]), list(zip(stacks.sum(dim=-1) != 0, real_nodes_edge_mask)))) 


ratio of edges types: p[neiougbour y | self y]
{key: value[:,1].sum() / (edges[(key[0],1-key[1])][:,1].sum()   + value[:,1].sum()  ) for key,value in edges.items() }
(0, 0) = {Tensor} torch.Size([]) tensor(0.7970)
(0, 1) = {Tensor} torch.Size([]) tensor(0.2030)
(1, 0) = {Tensor} torch.Size([]) tensor(0.8416)
(1, 1) = {Tensor} torch.Size([]) tensor(0.1584)

p[connect | (y_neigbour,y_self) = {key: value[:,1].sum() / ( len( edges[(key[0],1-key[1])]  ) + len(value)  ) for key,value in edges.items() }
(0, 0) = {Tensor} torch.Size([]) tensor(0.0074)
(0, 1) = {Tensor} torch.Size([]) tensor(0.0019)
(1, 0) = {Tensor} torch.Size([]) tensor(0.0078)
(1, 1) = {Tensor} torch.Size([]) tensor(0.0015)


Issues:


