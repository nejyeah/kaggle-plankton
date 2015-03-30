require 'torch'
require 'csvigo'
require 'xlua'
require 'paths'
require 'nn'
require 'image'

-------------------------------------------------------------------------------
--parse command line 
if not opt then
  print '==> processing options'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('GalaxyZoo Training script')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-visualize', true , 'visual sanity checks for data loading')
  cmd:text()
  opt = cmd:parse(arg or {})
end

loadSize = {1, 106, 106} --define the original picture size
sampleSize = {1, 96, 96} --define the processed picture size
dataroot_train = '../data/plankton/trainN' --define the training images root path
--dataroot_train = '../data/plankton/expand_train8' --define the training images root path
dataroot_validation = '../data/plankton/validationN'
dataroot_test = '../data/plankton/testN'
chunksize = 10000
--expand_train_solutions = '../data/plankton/expand8_label.t7'

train_solutions_file = '../data/plankton/training.csv'
validation_solutions_file = '../data/plankton/validation.csv'
test_solutions_file = '../data/plankton/test.csv'
------------------------------------------------------------------------------
--read the label information

os.execute('mkdir -p cache')
dofile('1_datafunctions.lua')
print '==> 1_data.lua'
print('==> Loading data')

--train_file = 'train.t7'
--test_file = 'test.t7'

if not paths.filep('cache/label.t7') then
   --read and store train class labels
	local train_file = io.open(train_solutions_file,'r')
	io.input(train_file)
	local class_title_s = io.read()
	local train_class_title = split(class_title_s,' ')
	io.close(train_file)

   local train_csvdata = csvigo.load{path=train_solutions_file, separator =' ', verbose=false}
   trsize = #train_csvdata[ train_class_title[1] ]
   train_label = torch.Tensor(trsize, 2)
   for i=1,trsize do
      train_label[i][1] = tonumber(train_csvdata[train_class_title[1]][i])
      train_label[i][2] = tonumber(train_csvdata[train_class_title[2]][i])
   end   
	
   -- read and store validation class labels
	local validation_file = io.open(validation_solutions_file,'r')
	io.input(validation_file)
	class_title_s = io.read()
	local validation_class_title = split(class_title_s,' ')
	io.close(validation_file)

   local validation_csvdata = csvigo.load{path=validation_solutions_file, separator =' ', verbose=false}
   vasize = #validation_csvdata[ validation_class_title[1] ]
   validation_label = torch.Tensor(vasize,2)
   for i=1,vasize do
      validation_label[i][1] = tonumber(validation_csvdata[validation_class_title[1]][i])
      validation_label[i][2] = tonumber(validation_csvdata[validation_class_title[2]][i])
   end   
 	
   -- read and store test class labels
   local test_file = io.open(test_solutions_file,'r')
   io.input(test_file)
   class_title_s = io.read()
   local test_class_title = split(class_title_s,' ')
   io.close(test_file)
	
   local test_csvdata = csvigo.load{path = test_solutions_file, separator = ' ', verbose = false}
   tesize = #test_csvdata[ test_class_title[1]]
   test_label = torch.Tensor(tesize,2) 
   for i=1,tesize do
	  test_label[i][1] = tonumber(test_csvdata[test_class_title[1]][i])
	  test_label[i][2] = 1
   end

   -- pick out classes names  
   classes = {}
   for i =1,#train_class_title-2 do
	  table.insert(classes,train_class_title[i+2])
   end
   nclasses = #classes

   -- store all information in a table
   label = {}
   table.insert(label,classes)
   table.insert(label,train_label)
   table.insert(label,validation_label)
   table.insert(label,test_label)

   torch.save('cache/label.t7', label)
else
   print('Loading from cache')
   label = torch.load('cache/label.t7')
   -- data: torch.DoubleTensor{30336x122}
   classes = label[1]
   train_label = label[2]  --> DoubleTensor - size 27238x2
   --train_label = torch.load(expand_train_solutions)[1]  --> DoubleTensor - size 27238x2
   validation_label = label[3] --> DoubleTensor - size 3098x2
   test_label = label[4]
   trsize = train_label:size(1)
   vasize = validation_label:size(1)
   tesize = test_label:size(1)
   --tesize = 5
   nclasses = #classes
end

--print('label..:')
--print(label)
--print('train_label..:')
--print(train_label[1][1],train_label[1][2])
--print('validation_label:')
--print(validation_label[1][1],validation_label[1][2])
print('trsize:'..trsize) --27238
print('vasize:'..vasize) --3098
print('tesize:'..tesize) --130400
print('nclasses:'..nclasses) --121
-----------------------------------------------------------------------------
print('==> load data')
--local time = sys.clock()
--if opt.test then
	tedata = torch.Tensor(tesize,loadSize[1],loadSize[2],loadSize[3])
	for i=1,tesize do
		tedata[i] = getSample(dataroot_test,test_label[i][1])
	end
--else
	trdata = torch.Tensor(trsize, loadSize[1], loadSize[2], loadSize[3])
	for i=1,trsize do
   		trdata[i] = getSample(dataroot_train,train_label[i][1])
	end

	vadata= torch.Tensor(vasize, loadSize[1], loadSize[2], loadSize[3])
	for i=1,vasize do
   		vadata[i] = getSample(dataroot_validation,validation_label[i][1])
	end
--end
--time = sys.clock() - time
--print('load data time cost:'..time..' ms') --> 8098 ms
------------------------------------------------------------------------------
-- define the trainData and validationData struct for training, which has ben divided 9:1 mannually 
print('==> define trainData and validationData struct')
--if opt.test then
	expand_size = 8
	testData = {}
	testData.data = torch.Tensor(chunksize,expand_size,sampleSize[1],sampleSize[2],sampleSize[3])
	--testData.size = function() return tesize end
--else
	trainData = {}
	trainData.data = torch.Tensor(chunksize, sampleSize[1], sampleSize[2], sampleSize[3])
	trainData.labels = torch.DoubleTensor(chunksize)
	--trainData.size = function() return trsize end

	validationData = {}
	--validationData.data = torch.Tensor(chunksize, sampleSize[1], sampleSize[2], sampleSize[3])
	validationData.expand_data = torch.Tensor(chunksize, expand_size,sampleSize[1], sampleSize[2], sampleSize[3])
	validationData.labels = torch.DoubleTensor(chunksize)

	--validationData.size = function() return vasize end
--end
collectgarbage()

------------------------------------------------------------------------------
--print '==> preprocessing data'
function loadchunk(index,flag,trandIndices)
	-- load train
	local mean, std , lim , absolute_index
	if flag==0 then
		lim = math.min(chunksize,trsize-chunksize*(index-1))
		absolute_index = chunksize*(index-1)
		for i=1,lim do
			local s = trdata[trandIndices[absolute_index+i]]
   			trainData.labels[i] = train_label[trandIndices[absolute_index+i] ][2]
        	trainData.data[i] = jitter_try(s)
			mean = trainData.data[i]:mean()
			std = trainData.data[i]:std()
			trainData.data[i]:add(-mean)
			trainData.data[i]:div(std)	
		end 
	elseif flag==1 then
		lim = math.min(chunksize,vasize-chunksize*(index-1))
		absolute_index = chunksize*(index-1)
		for i=1,lim do
        	local s = vadata[absolute_index+i]
   			validationData.labels[i] = validation_label[absolute_index+i][2]
			validationData.expand_data[i] = expandTestSampleN(s,expand_size)
		end
	elseif flag==2 then
		lim = math.min(chunksize,tesize-chunksize*(index-1))
		absolute_index = chunksize*(index-1)
		for i=1,lim do
			local s = tedata[absolute_index+i]
		    testData.data[i] = expandTestSampleN(s,expand_size)
		end	
	else error("wrong flag in function loadchunk(index,flag)!") end
end

function preprocess()
	-- load and preprocess train data
	--if opt.test then
		local mean, std
		for i=1,tesize do
			local s = tedata[i]
		    testData.data[i] = jitter(s)
			mean = testData.data[i]:mean()
			std = testData.data[i]:std()
			testData.data[i]:add(-mean)
			testData.data[i]:div(std)	
		    --testData.data[i] = image.scale(s,sampleSize[2],sampleSize[3])	
		end
        
	--else
		local trandIndices = torch.randperm(trsize)
		for i=1,trsize do
   			trainData.labels[i] = train_label[trandIndices[i] ][2]
   			--local s = getSample(dataroot_train,train_label[trandIndices[i] ][1])
        	local s = trdata[trandIndices[i]]
        	trainData.data[i] = jitter(s)
			mean = trainData.data[i]:mean()
			std = trainData.data[i]:std()
			trainData.data[i]:add(-mean)
			trainData.data[i]:div(std)	
		end
    
    	-- load and preprocess validation data
		for i=1,vasize do
   			validationData.labels[i] = validation_label[i][2]
   			--local s = getSample(dataroot_validation,validation_label[i][1])
        	local s = vadata[i]
        	validationData.data[i] = jitter(s)
			validationData.expand_data[i] = expandTestSampleN(s,expand_size)
			mean = validationData.data[i]:mean()
			std = validationData.data[i]:std()
			validationData.data[i]:add(-mean)
			validationData.data[i]:div(std)	
		end
		trainData.data = trainData.data:float()
		validationData.data = validationData.data:float()
		
end

--[[
local time = sys.clock()
preprocess()
time = (sys.clock() - time)*1000
print('preprocess time cost:'..time ..' ms') --3565.27 ms
--]]
-- Local normalization
--[=[
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c=1, loadSize[1] do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,validationData:size() do
      validationData.data[{ i,{c},{},{} }] = normalization:forward(validationData.data[{ i,{c},{},{} }])
   end
end
--]]

-----------------------------------------------------------------------------
if opt.visualize then
   --local lena = expandTestSample(image.scale(image.lena(), loadSize[2], loadSize[3]))
   --image.display{image=lena, nrow=16}
   --print(#getTest(1))
   --local validationImage, validationgt = getTest(1)
   --print(tostring(validationData[1][1] .. '.png'))
   --print(#validationImage) --torch.LongStorage of size 4 : 256x1x111x111
   --print(validationgt)
   ---image.display{image=validationImage[1], legend='original-image after mean'}
   --image.display{image=validationImage, nrow=16}
   --local a,b = getBatch(156)
   --print(#a) 
   -- out >> torch.LongStorage of size 4 : 32x1x111x111
   --print(#a[1]) 
   -- out >> torch.LongStorage of size 3 : 1x111x111
   --print(a:size(1)) 
   -- out >> 32
   --print(b) 
   -- out >> torch.LongStorage of size 2 : 32x121
   --image.display{image=a:float(), nrow=16}
   --print(#a)
   --print(#b)
end
--]=]
