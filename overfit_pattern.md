
* Saw I  cannot overfit using attention(gating stuck at about 0.6 with stable training)

* thought about the problem with pen and paper and got to the conclustion that this problem cannot be solved using attention(averaging)

* Decided to implement gating(sigmoid), as an alternative to averaging(softmax)

* gating supressed 0.5, but training was very unstable, with scores jumping up and down.

* plot weight, gradients, grad/weight

* Realize the problem is with unstable gradients and probably somewhere in the layer norm(as it was having the largest and most unstable gradients)

* Implement proper masking, that I was more confident was working, that masks nodes that are out of reach of the adj_stack

* Tried Multiple things to get a better feeling for the problem and how to fix it:
  - Broke down gnn baseline to try and understand the effects of different components.  
  Tried disabling batch norm
        [
         1) no batch norm in between layer. still working fine. good performance, stable accuracy which raises linearly https://www.comet.ml/danielglickmantau/test/480df44051ba45479c8082e11d63377c?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
         2) disabling batch norm in GNN's MLP(and also in between).  model gets 50% https://www.comet.ml/danielglickmantau/test/96367b1099bc4109b11255b5e58e2b79?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
         3) only mlp batch norm disabled and batch norm between gnn enabled https://www.comet.ml/danielglickmantau/test/5e260c893fa7426fb6f5b189bbd5b64f
         4) batch norm in mlp enabled and disabled between layers. Working https://www.comet.ml/danielglickmantau/test/69ab0eb4723d4aba881be86202d993c5
         ]
    Batch norm in MLP seems most important
  - Simplified transformer and made it closer to GNN  
    * Saw that training is tstable with 1 layer and gets unstable when adding more layers  
    with layer norm looking most suspicous
    * Work with adj_stack=[0,1] and use_distance=True to try and emulate gin
    * Try Replace layer norms with batch norms. try adding batch norm in transformer ff
    * reduce learning rate
    
  Reducing learning rate and adding batch norm in between seems most effective.


Take aways:
* Start with testing simple things(learning rate)
* Dumbify the model complete to get it close to something that already works
* Having a fast running test pipeline is very useful
* Keep all parameters in args, makes life easier
* Experiment with one thing at a time. log the experiment somewhere and log the conclustion before moving forward