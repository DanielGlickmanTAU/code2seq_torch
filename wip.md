

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



### 13/3
* positional gating is much faster than gin somehow

### Comparing modified gin with positonal gating with use_distance 

### Architecture
modfied gin( use sigmoid on wights)
gin layer:
         x_mid = (1 + sigmoid(self.eps)) * x + sigmoid(self.eps2) * relu(neigbours) )
         out = x_mid -> linear -> batch norm -> relu -> linear
         out = batch_norm(out)
         out = x + relu(out)

positional gating norm first and use_distance with adj_stack=[0,1]:
 x_mid = gating( norm1(x) )
        gating: sigmoid(w1) * x + sigmoid(w2) * neigbours
 x_mid = x + x_mid
 out = x_mid -> norm2 -> linear -> relu -> linear
 out = x_mid + out
 

possible things that can cause gin to learn and positional to not learn:
(1 + sigmoid(eps))  removing it still gives almost same results   https://www.comet.ml/danielglickmantau/test/ce88b397e3e843379fa667ab438c8eb1?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
relu(neigbours)    removing it still gives good results, 0.846    https://www.comet.ml/danielglickmantau/test/4029891ad91a43acb4ba8042aa41a291?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
removing both:https://www.comet.ml/danielglickmantau/test/578e362837554878844204b5e2e8565a?experiment-tab=chart&search=score&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step

### Unstable gradients:
Training single layer position attention(+distance): https://www.comet.ml/danielglickmantau/test/074867d9e5bf4aaba9db7a291c4edc33?experiment-tab=chart&search=grad%2Fweight&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
seems stable. only norm2 is somewhat unstable.

2 layers position attention with distance:
more unstable but reasonable..
once again norm2 seems like the culpritiuyi


masking in second norm does not seem to effect.. 
masking layer norm does not seem to effect now..

4 layers no layer norm at all: low score(0.5) and goes up and downww https://www.comet.ml/danielglickmantau/test/5553a00927ad4ee0ba2c02290e029e0e?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
4 layers.. no layer norm masking and small learning rate: https://www.comet.ml/danielglickmantau/test/312b94fbdfb04ef180dacbc924135a1b
4 layers.. no layer norm masking and large learning rate: probably https://www.comet.ml/danielglickmantau/test/2c50c4c894c14c3a87dcbce2134caefe
4 layers layer norm masking and large learning rate: https://www.comet.ml/danielglickmantau/test/6ece693e5bc349ed8570902e13eb5ba6?experiment-tab=chart&search=score&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
4 layers layer norm masking and small learning rate https://www.comet.ml/danielglickmantau/test/8f68e70672d749348b05d5714778ff81?experiment-tab=stdout


no distance:
4 layers.. 1e-4 learning rate.. score jumps up and down masked norm https://www.comet.ml/danielglickmantau/test/780c26fb02b245ddbb5e7c9d702fb2bf?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
4 layers 1e-5 learning rate  small jumps up and down but generally goes up.. very slowly.. https://www.comet.ml/danielglickmantau/test/6bdc5cf6d313442e963efb9a2f85a6d9?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
4 layers 1e-5 learning rate and no masking in norm https://www.comet.ml/danielglickmantau/test/93ab8630c69f406f842cded506370dcc

learning rate seems to have the real effect.. masking in norm nop


batch norms:
attention batch https://www.comet.ml/danielglickmantau/test/f5f67994ad554e34936b1213d032f40b
ff batch norm: https://www.comet.ml/danielglickmantau/test/6b0c579ab68b4473b5b37e5ea0aa17a6
both batch norm: https://www.comet.ml/danielglickmantau/test/268bd590e6ab4ec38ed5ac6e08a0355d
batch norm added only in transformer MLP(after first linear) WORKS OVER 0.9!! https://www.comet.ml/danielglickmantau/test/96e6be36c88a405ab9c052b822d4e7ec?experiment-tab=chart&search=score&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step


bug receptive field(8) https://www.comet.ml/danielglickmantau/test/b7aa42dbc713447a876c3680e3d4491c?experiment-tab=chart&search=score&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step


### 14.3 
Gating with larger adj_stack=[0,1,2,3,4]


Attention(no gating) with batch norm in MLP, seems to work on pattern... it seems mandatory for larger adj stacj seems even better..
https://www.comet.ml/danielglickmantau/position-attention-mlp-batch-norm/view/new/experiments

with gating maybe batch norm helps larger receptive field I dont know...
https://www.comet.ml/danielglickmantau/gin-pattern/view/new/experiments


### 15.3
attention with batch norm sota on pattern https://www.comet.ml/danielglickmantau/position-attention-mlp-batch-norm/view/new/experiments
adj_stack=[0,1,2,3,4] is better than [0,1]
learning rate does not effect.. higher lr is better if anything

Seems like it is not the learning rate and not the batch norm. it is the masking
results pattern gating 4 layers https://www.comet.ml/danielglickmantau/gin-pattern/view/YjK8ZWP0KIBaQK2I0qAUzLaHa/experiments:
note that things may improve as it is still running
MLP-BN:False Attention norm:batch FF norm:batch  0.839
MLP-BN:False Attention norm:batch FF norm:layer  0.852
MLP-BN:False Attention norm:layer FF norm:batch  0.856 https://www.comet.ml/danielglickmantau/gin-pattern/e92451205a7a46b1b8005a1949cb02ef
MLP-BN:False Attention norm:layer FF norm:layer  0.835 https://www.comet.ml/danielglickmantau/gin-pattern/2048382a6d84477094677ad4891f89ed

MLP-BN:True Attention norm:batch FF norm:batch  0.837 https://www.comet.ml/danielglickmantau/gin-pattern/fe9c8e46f02e453f8305bcf9d472d57b
MLP-BN:True Attention norm:batch FF norm:layer  0.839 https://www.comet.ml/danielglickmantau/gin-pattern/cdedfc02c77e4d2d9539d336c76b762d
MLP-BN:True Attention norm:layer FF norm:batch  0.851 https://www.comet.ml/danielglickmantau/gin-pattern/53bb1888a82c422b96e8e84fb211b9bb
MLP-BN:True Attention norm:layer FF norm:layer  0.835


### 24.3

debugging polyndrome example. I am expecting that if I give oppsite weights(+1,-1) to my heads. multuplying the 2 heads
will yield maximum results in opposing(symetric) nodes

This is not the case.
Q1: does multiplying heads make sense?
yes each head h, h[i,j] gives a feature to the edge(i,j). 
each row h[i,:] gives a feature vector for i, which shows its compatibility with the rest of the nodes 

Q2: do I need to transpose the second head?
Lets assume both heads are the same, h1=h2.
I want h1h2[i,i] to contain the dot product of h1[i] and h2[i]
so YES, I need to transpose h2

Q3: why is (h1@h2.T)[1,2] < (h1@h2.T)[1,3], if 1,2 are opposite in the polnydrome?
h1[1] [0.875, 1.75, 1.125, 0.25]
h2[2] [-0.25, -1.125, -1.75, -0.875]
h2[3] [-0.25, -0.5, -1.75, -1.5]


I want product of opposining node heads to be maximal..
Lets look at the stacks.
We have that stacks.permute(1,2,0)[1][2] == stacks.permute(1,2,0)[2][1].
The edge (1,2) is same as (2,1) They are symetric.

works:
(stacks.permute(1,2,0).reshape(4,-1)@-stacks.permute(1,2,0).reshape(4,-1).T).detach().numpy()
no casting to heads...

so it works when I reverse feature individually, but not when I sum.. consider that..
needs to be fixed but move on for now


We can expect the dot product to be maximal?
gusses: batch normalize feature(original stack)...
if i do 2x1 or 3x1 convlustion, I need to 

issue: 

stacks[1][2] = torch.Size([5]) tensor([0.0000, 0.5000, 0.0000, 0.6250, 0.0000])

I am multiplying the edges of each graph


### 11/4
with pyramid graph min size=1 max size=5. and num_stacks=3 
I get 23 unique edges.. and only zero edge are ambigoious.


### 12/4
------------  ---------------  ------------  ------------------  --------------------------
pyramid base  receptive field  unique edges  of which ambiguous  % pairs in receptive field
10            1                5             0                   0.1074380165289256
10            2                28            0                   0.25024793388429756
10            4                269           0                   0.5834710743801652
10            9                2530          0                   1.0
10            19               3026          0                   1.0
20            1                5             0                   0.030612244897959218
20            2                26            0                   0.07714285714285718
20            4                235           0                   0.21360544217687072
20            9                6615          0                   0.6333333333333333
20            19               43229         0                   1.0
30            1                5             0                   0.014221297259798815
30            2                27            0                   0.03675338189386057
30            4                290           0                   0.10718002081165456
30            9                10626         1                   0.36968435657301424
30            19               138187        1                   0.8641692681234825
50            1                5             0                   0.005305651672433687
50            2                26            0                   0.013986620530565208
50            4                283           0                   0.04243598615916955
50            9                12620         1                   0.16316493656286046
50            19               382368        1                   0.5011395617070358
------------  ---------------  ------------  ------------------  --------------------------


00078 = {tuple: 3} ((0.0, 0.0, 0.0, 0.015625, 0.012442131), (0.0, 0.0, 0.0, 0.01388889, 0.014467594), 7.11658106627034e-06)
00079 = {tuple: 3} ((0.0, 0.0, 0.0, 0.015625, 0.012442131), (0.0, 0.0, 0.0, 0.01388889, 0.014467595), 7.116584838991191e-06)
00080 = {tuple: 3} ((0.0, 0.0, 0.0, 0.0, 0.006365741), (0.0, 0.0, 0.0, 0.0, 0.0034722227), 8.37244886806905e-06)

size 10:
(4,2) to (7,5) : torch.Size([5]) tensor([0.0000, 0.0000, 0.0000, 0.0046, 0.009  3])
(4,2) to (7,4)      torch.Size([5]) tensor([0.0000, 0.0000, 0.0000, 0.0139, 0.0123])

(8,7) to (9,9) [0.0, 0.0, 0.0833333358168602, 0.0416666679084301, 0.05844907835125923]
(8,7) to (6,6) [0.0, 0.0, 0.0694444477558136, 0.033564817160367966, 0.04976852238178253]

##
### 13/4
### Complexity of 3 coloring using transformer+edge bias vs simple og transformer
Main difference concepptually, if og transformer learns how to re distributae the weights with a deep network,
while transformer+edge collpases after every averaging

e = dim edge ; d = dim x  
edge
   params: assuming passing through ffn:  8e^2 ...=( e*4e + e*4e)
   complexity: (nxn)*(8e^2)
edge bias: e->2 
 params: e*2
MHA
 params: d^2 * 4    ... (4 for wq,wk,wv,wo)
 complexity: 3*(nxd) + (n^2)d ... (q*qk, k*wk, v*wo + (kq)*v)
FFN: 
 params: 8d^2
  complexity n * 8d^2

Edge bais transformer parameters:
 edge * l + L * transformer = 8e^2 * l + L( (4*d^2) + 8*d^2) 
                                            q,k,w,o
L means num layers 
l means 1 or L, depending what we choose

simple og transformer:
edge ffn: e -> e
edge bias:e-> 2: wq, wk: e-> 1.. need 2 weights to break transetivity

each layer:
e = ffn(e)
(e@wq)@ (e@wk) .. not sure if I want softmax here in hidden layers.. other options are relu and gaussian normnization

params:
ffn: 8*e^2
wq,wk: 2e



path attention:
(E @ wk) @ (E @ wk )

cluster attention:
(E@ wk) @ (E.T @ wk)



