----------------------------------------------------------------------
-- This tutorial shows how to train different models on the street
-- view house number dataset (SVHN),
-- using multiple optimization techniques (SGD, ASGD, CG), and
-- multiple types of models.
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'number of threads')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.6, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:option('-progressBar',true,'control to use xlua.progress()')
cmd:option('-earlystopflagnll',true,'choose which flag nll|accuracy to early-stopping')
cmd:option('-retrain',"none",'control to load model to train')
cmd:option('-test',false,'test or train,default is train ')
cmd:option('-average',true,'expand the test set and average the output')
cmd:option('-symbol','batch32','add to the the results filename to vary from different trail')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'

dofile '5_validation.lua'
dofile '6_test.lua'
----------------------------------------------------------------------
--print '==> training!'
--define some parameters of early-stopping
--if opt.test then
	--dofile '6_test.lua'
   	--preprocess() --cost about  
	--test()
--else
	--dofile '5_validation.lua'
	savemodel_file = 'model_'..opt.symbol..'.net'
	n_epochs = 300
	best_validation_loss = 1000
	validation_loss = 0
	best_validation_nll = 10
	validation_nll = 0 
	done_looping = false
	
	toleration = 30
	descend_toleration = 1
	tolerationflag = 1 
	epoch = 1
	local time = sys.clock()
	while epoch<n_epochs and not done_looping do
   		--preprocess() --cost about 3565 ms   
   		local r_nll,r_accuracy = train()
   		collectgarbage()
   		validation_nll, validation_loss = validation()
   		collectgarbage()
   		if opt.earlystopflagnll then
   			if validation_nll < best_validation_nll then
				tolerationflag = 1
				descend_toleration = 1 
				best_validation_nll = validation_nll
				
				--save_model
				if best_validation_nll < 0.782 then
					local filename = paths.concat(opt.save,'epoch'..tostring(epoch)..'_'..savemodel_file)
					torch.save(filename,model)
					test()
				end
			else
				tolerationflag = tolerationflag+1
				descend_toleration = descend_toleration+1
  			end
  		 
  	 	else
   			if validation_loss < best_validation_loss then 
				tolerationflag = 1
				best_validation_loss = validation_loss
				--save_model
				if best_validation_loss > 0.75 then
					local filename = paths.concat(opt.save,'epoch'..tostring(epoch)..'_'..savemodel_file)
					torch.save(filename,model)
					test()
				end
			else 
				tolerationflag = tolerationflag+1
  			end
  		end
  		if tolerationflag > toleration then done_looping = true end
		if descend_toleration==10 then  
			optimState.learningRate = optimState.learningRate/5
			descend_toleration = 1
		end 
  		epoch = epoch + 1
	end
	time = sys.clock() - time
	time = time/60
	print('epoch:'..epoch..'\tall time cost:'..time..' min')
--end
--]]
