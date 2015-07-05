----------------------------------------------------------------------
-- This script contains the ConvNet multi-scale model used for the 
-- gtsrb challenge
-- 
-- Prior to using this script, we need to generate the datasets with 
-- createDataSet.lua, then load the dataset in dataset.lua and pre-
-- process it using preProcess.lua.
--
-- These results are based on Yann Lecun et al. work :
-- http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/
-- reference_papers/2-sermanet-ijcnn-11-mscnn.pdf
--
-- Hugo Duthil
----------------------------------------------------------------------
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
nstates = {32, 64, 100}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

-- A typical convolutional network, with locally-normalized hidden
-- units, and L2-pooling
model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- replicate the input so we can use it in the final classifier
model:add(nn.Replicate(2))

-- create a parallel module to compute stage 2 and keep stage 1's output
paral = nn.Parallel(1,1)

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
stage2 = nn.Sequential()
stage2:add(nn.SpatialConvolutionMap(nn.tables.random(nstates[1], nstates[2], math.floor(0.8*nstates[1])), filtsize, filtsize))
stage2:add(nn.Tanh())
stage2:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
stage2:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))
stage2:add(nn.Reshape(nstates[2]*filtsize*filtsize))

-- identity module that keep stage1 features and down sample it to feed final classifier
idmodule = nn.Sequential()
idmodule:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
idmodule:add(nn.Reshape((width-filtsize+1)*(width-filtsize+1)/16*nstates[1]))

-- add stage2 and identity modules to parallel module
paral:add(stage2)
paral:add(idmodule)

-- add parallel module to model
model:add(paral)

-- stage 3 : standard 2-layer neural network
model:add(nn.Linear((width-filtsize+1)*(width-filtsize+1)/16*nstates[1]+nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))

-- weighted criterion as the distribution is no uniform accross classes
weights = torch.Tensor({210, 2220, 2250, 1410, 1980, 1860, 420, 1440, 1410, 1470, 2010, 1320, 2100, 2160, 780, 630, 420, 1110, 1200, 210, 360, 330, 390, 510, 270, 1500, 600, 240, 540, 270, 450, 780, 240, 689, 420, 1200, 390, 210, 2070, 300, 360, 240, 240})
if use_validation_set then
    weights:add(-30)
end
weights:div(37919)

-- classification criterion
criterion = nn.CrossEntropyCriterion(weights)

print("Model loaded :")
print(model)
