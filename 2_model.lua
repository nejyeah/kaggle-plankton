----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
--require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('kaggle plankton Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 121-class problem
noutputs = nclasses

-- input dimensions
nfeats = sampleSize[1]
width = sampleSize[2]
height = sampleSize[3]
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,128,128,256}
filtsize = 5
poolsize = 2
convnet_out =( (sampleSize[2]-filtsize+1)/poolsize-filtsize+1 )/poolsize
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs,false))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs,false))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet' then

   if opt.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> max pooling 
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], 5, 5)) --96-5+1 = 92   
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) --92/2 = 46

      -- stage 2 : filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], 5, 5))-- 46-5+1 = 42
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) --42/2 = 21

      -- stage 3 : filter bank -> squashing  
      model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], 4, 4))-- 21-4+1 = 18
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) --18/2 = 9
      
      -- stage 4 : filter bank -> squashing -> max pooling 
      model:add(nn.SpatialConvolutionMM(nstates[3], nstates[4], 4, 4))-- 9-4+1 = 6
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) --6/2 = 3

	  -- stage 3 : standard 2-layer neural network
      model:add(nn.View(nstates[4]*3*3))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[4]*3*3, nstates[4]*3))
      model:add(nn.ReLU())
      model:add(nn.Linear(nstates[4]*3, noutputs))

   else
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(nstates[2]*convnet_out*convnet_out))
      model:add(nn.Linear(nstates[2]*convnet_out*convnet_out, nstates[2]*convnet_out))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[2]*convnet_out, noutputs))
   end
else
   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using gfx.image().

if opt.visualize then
   if opt.model == 'convnet' then
      print '==> visualizing ConvNet filters'
      --gfx.image(model:get(1).weight, {zoom=2, legend='L1'})
      --gfx.image(model:get(5).weight, {zoom=2, legend='L2'})
   end
end
