import sys
sys.path.append("../")

import torch 
from torchviz import make_dot

from model_utils.mlp import  MLPNet

model =  MLPNet(p=0.5, logvar=True, n_feature=1,
                 n_hidden=32, n_output=1)
x= torch.randn(32, 1)
y , logvar = model(x)

dot = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.format = 'png'
dot.render('mlp')