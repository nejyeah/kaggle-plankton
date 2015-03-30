----------------------------------------------------------------------
-- This script implements a validation procedure, to report accuracy
-- on the validation data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining validation procedure'

-- validation function
function validation()
   -- local vars
   local time = sys.clock()
   -- set model to evaluate mode (for modules that differ in training and validationing, like Dropout)
   model:evaluate()
   local va_nll = 0
   local va_accuracy = 0
   -- validation over validation data
   print('==> validationing on validation set:')
   for chunk=1,math.ceil(vasize/chunksize) do
    collectgarbage()
	local lim = math.min(chunksize,vasize-chunksize*(chunk-1))
	local index = chunksize*(chunk-1)
	loadchunk(chunk,1)
   for t = 1,lim do
      -- disp progress
      if opt.progressBar then xlua.progress(index+t, vasize) end

      -- get new sample
      local input = validationData.expand_data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = validationData.labels[t]
	  
      -- validation sample
      local pred = model:forward(input)
      pred = pred:float()
	  pred:exp()
	  pred = pred:mean(1)[1]
	  pred:div(pred:sum())
	  pred:log()
      confusion:add(pred, target)
	  local err = criterion:forward(pred,target)
      va_nll = va_nll + err
   end
   end
   -- timing
   time = sys.clock() - time
   time = time / vasize

   -- print confusion matrix
   print(confusion)
   
   -- get target information
   local atime = time
   local ava_nll = va_nll / vasize
   local ava_accuracy = confusion.totalValid*100
	
   print("\n==> time to validation 1 sample = (average)'..(atime*1000)..'ms' ")
   print('\t validation_nll = (average)'.. ava_nll)
   print('\t validation_accuracy =(average)'..ava_accuracy)
   print('')
   print('')
   print('')
   -- update log/plot
   validationLogger:add{['accuracy%mean'] = ava_accuracy,['validation_nll(average)'] = ava_nll }
   
   -- next iteration:
   confusion:zero()

   return ava_nll, ava_accuracy
end
