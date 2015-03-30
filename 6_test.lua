----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------

print '==>read sampleSubmission'
sampleSubmission = './results/sampleSubmission.csv'
local samplefile = io.open(sampleSubmission,"r")
io.input(samplefile)
sample_class_title_s = io.read()
--print(sample_class_title_s)
sample_class_title = split(sample_class_title_s,',')
io.close(samplefile)

submissionfile = "submission_"..opt.symbol..".csv"
classes[nclasses] = "stomatopod"

label_turn = {}
for i=1 ,nclasses do
	for j =1, nclasses do
		if sample_class_title[i+1] == classes[j] then 
			label_turn[i] = j
			break 
		end
    end
	if label_turn[i] == nil then print(sample_class_title[i+1]..i);label_turn[i] = 0 end
end
print(label_turn)
--[[
print(sample_class_title[108])
print(classes[nclasses])
print(sample_class_title[108]==classes[nclasses])
print('==>sort')
table.sort(label_turn)
for i=1,#label_turn do
	print(label_turn[i])
end
--]]


print '==> defining predict procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   -- test over test data
   print('==> predicting on test set:')

   --write the head
   local submissionfilename = tostring(epoch)..submissionfile
   local fp = io.open(paths.concat(opt.save,submissionfilename),'w')
   fp:write(sample_class_title_s..'\n')
   for chunk=1,math.ceil(tesize/chunksize) do
	local lim = math.min(chunksize,tesize-chunksize*(chunk-1))
	local index = chunksize*(chunk-1)
	loadchunk(chunk,2)
   for t = 1,lim do
      -- disp progress
      if opt.progressBar then xlua.progress(index+t, tesize) end

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
	  
      -- test sample
      local pred = model:forward(input)
      pred = pred:float()
	  pred:exp()
	  pred = pred:mean(1)[1]
	  pred:div(pred:sum())
	  pred:log() 
      --print(10^pred)
       
      -- write to submission file
	  
      local line = tostring(test_label[index+t][1])..'.jpg'
      local sum =0 
	  for i=1,nclasses do 
		 line = line..','..tostring(math.exp(pred[label_turn[i]]))
		 sum = sum+math.exp(pred[label_turn[i]])
		 --line = line..','..tostring(pred[label_turn[i]])
		 --sum = sum+pred[label_turn[i]]
         --print(10^pred[i])
	  end
	  --print(t..'  sum:'..sum)
	  fp:write(line..'\n')
   end
   end
   fp:close()
   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   
end
--]=]
