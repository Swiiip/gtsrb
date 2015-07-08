-- MLP model --

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

-- 43-class problem
noutputs = 43

-- input dimensions
if params.use_3_channels then nfeats = 3 else nfeats = 1 end -- # of channels
width = 32
height = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes:
nhidden = 64

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, nhidden))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden, noutputs))
model:add(nn.Select(1, 1)) -- in order to be a Tensor of size 43 and not 1x43

-- weighted criterion as the distribution is no uniform accross classes
weights = torch.Tensor({210, 2220, 2250, 1410, 1980, 1860, 420, 1440, 1410, 1470, 2010, 1320, 2100, 2160, 780, 630, 420, 1110, 1200, 210, 360, 330, 390, 510, 270, 1500, 600, 240, 540, 270, 450, 780, 240, 689, 420, 1200, 390, 210, 2070, 300, 360, 240, 240})
weights:div(37919)

-- classification criterion
criterion = nn.CrossEntropyCriterion(weights)

