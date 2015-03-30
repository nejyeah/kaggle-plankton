require 'image'
require 'nn'

function jitter(s)
   local d = torch.rand(10)
   -- vflip
   if d[1] > 0.5 then
      s = image.vflip(s)
   end
   -- hflip
   if d[2] > 0.5 then
      s = image.hflip(s)
   end
   -- rotation
   if d[3] > 0.5 then
      s = image.rotate(s, math.pi * d[4])
   end
   -- crop a 0.9 to 1.1 sized random patch and resize it to 128
   if d[5] > 0.5 then
      local scalef = torch.uniform(0.9, 1.1) 
      local size = {3, sampleSize[2] * scalef, sampleSize[3] * scalef}
      local startX = math.ceil(d[6] * (loadSize[2] - size[2] - 1))
      local startY = math.ceil(d[7] * (loadSize[3] - size[3] - 1))
      local endX = startX + size[2]
      local endY = startY + size[3]
      s = image.crop(s, startX, startY, endX, endY)
      -- now rescale it to sampleSize
      s = image.scale(s, sampleSize[2], sampleSize[3])
   else
      -- crop a sampleSize[2]xsampleSize[3] random patch
      local startX = math.ceil(d[6] * (loadSize[2] - sampleSize[2] - 1))
      local startY = math.ceil(d[7] * (loadSize[3] - sampleSize[3] - 1))
      local endX = startX + sampleSize[2]
      local endY = startY + sampleSize[3]
      s = image.crop(s, startX, startY, endX, endY)
   end
   return s
end

function jitter_try(s)
   local d = torch.rand(10)
   -- vflip
   if d[1] > 0.5 then
      s = image.vflip(s)
   end
   -- hflip
   if d[2] > 0.5 then
      s = image.hflip(s)
   end
   -- rotation
   local angle = math.ceil(d[3]/0.125) 
   s = image.rotate(s, math.pi *angle/8)

   -- translate
   s = image.translate(s,math.ceil(10*d[4]),math.ceil(10*d[5]))
   s = image.scale(s, sampleSize[2],sampleSize[3])
   return s
end

local function test_t(im, o)
   local x1 = math.ceil((loadSize[2] - sampleSize[2])/2)
   local size = sampleSize[2]
   local t = math.floor(loadSize[2] * 0.04)
   o[1] = image.crop(im, x1, x1, x1+size, x1+size) -- center patch
   o[2] = image.crop(im, x1-t, x1, x1-t+size, x1+size)
   o[3] = image.crop(im, x1+t, x1, x1+t+size, x1+size)
   o[4] = image.crop(im, x1, x1-t, x1+size, x1-t+size)
   o[5] = image.crop(im, x1, x1+t, x1+size, x1+t+size)
   o[6] = image.crop(im, x1-t, x1-t, x1-t+size, x1-t+size)
   o[7] = image.crop(im, x1+t, x1-t, x1+t+size, x1-t+size)
   o[8] = image.crop(im, x1+t, x1+t, x1+t+size, x1+t+size)
end

local function test_rt(im, o)
   -- rotate further 0
   test_t(im, o[{{1,8},{},{},{}}])
   -- rotate further 45
   local im2 =image.rotate(im, math.pi/4)   
   test_t(im2, o[{{9,16},{},{},{}}])
   -- rotate further 30
   im2 =image.rotate(im, math.pi/6)   
   test_t(im2, o[{{17,24},{},{},{}}])
   -- rotate further 60
   im2 =image.rotate(im, math.pi/3)
   test_t(im2, o[{{25,32},{},{},{}}])
end

local function test_rrt(im, o, lightTesting)
   -- rotate 0
   test_rt(im, o[{{1,32},{},{},{}}])
   if not lightTesting then
      -- rotate -90
      local minus90 = torch.Tensor(im:size())
      for i=1,im:size(1) do
	 	minus90[i] = im[i]:t()
      end
      test_rt(minus90, o[{{33,64},{},{},{}}])
      -- rotate 90
      local plus90 = image.hflip(image.vflip(minus90))
      test_rt(plus90, o[{{65,96},{},{},{}}])
      -- rotate 180
      local plus180 = image.hflip(image.vflip(im))
      test_rt(plus180, o[{{97,128},{},{},{}}])
   end
end

function expandTestSample(im, lightTesting)
   -- produce the 256 combos, given an input image (3D tensor)
   local o
   if lightTesting then
      o = torch.Tensor(32, sampleSize[1], sampleSize[2], sampleSize[3])
      test_rrt(im, o[{{1,32},{},{},{}}], lightTesting)
   else
      o = torch.Tensor(256, sampleSize[1], sampleSize[2], sampleSize[3])
      -- original
      test_rrt(im, o[{{1,128},{},{},{}}], lightTesting)
      -- vflip
      test_rrt(image.vflip(im), o[{{129,256},{},{},{}}], lightTesting)
   end
   --[[
   for i=1,o:size(1) do
      o[i]:add(-o[i]:mean())
      o[i]:div(o[i]:std())
   end
   --]]
   if bmode == 'BDHW' then
      return o
   else
      return o
   end
end

function expandTestSampleN(im,n)
	local o = torch.Tensor(n,sampleSize[1],sampleSize[2],sampleSize[3])
	for i=1 , n do
    	local d = torch.rand(10)
    	local s =image.rotate(im, i*2*math.pi/n)
		-- vflip   
   		if d[1] > 0.5 then
      		s = image.vflip(s)
   		end
   		-- hflip
   		if d[2] > 0.5 then
      		s = image.hflip(s)
   		end
		-- scale
      	s = image.scale(s, sampleSize[2], sampleSize[3])
		o[i] = s
        o[i]:add(-o[i]:mean())
        o[i]:div(o[i]:std())
	end
	return o
end
	
function split(str,delim)
	local res = {}
	local pos = 1
	while true do 
		local nextpos = string.find(str, delim, pos)
		if not nextpos then 
			res[#res+1] = string.sub(str, pos, #str)
			break
		end
		local item = string.sub(str, pos, nextpos-1)
		res[#res+1] = item
		pos = nextpos + 1
	end
	return res
end

function getlabel(data)
	local numberOfSample = data:size(1)
	local numberOfclasses = data:size(2)-1
	local llabel = torch.Tensor(numberOfSample,2)
	for i=1, numberOfSample do
		local index = 1
		for j=1,numberOfclasses do
			llabel[i][1] = data[i][1]
            llabel[i][2] = 1
			if data[i][j+1] == 1 then
		    	index = j 
				llabel[i][2] = index
                break
			end
		end
	end
	return llabel
end

function getSample(dataroot,name)
	local filename = paths.concat(dataroot, tostring(name)..'.png')
	local im = image.load(filename, loadSize[1])
	im = image.scale(im, loadSize[2], loadSize[3])
	--im = wrap(im,loadSize[2])
	return im
end

function wrap(im,size)
    local im_wrap
    if im:size(2)<size then
        im_wrap = torch.Tensor(im:size(1),size,size)
        local start = math.floor((size-im:size(2))/2)
        --print(start)
        im_wrap:zero()
        for i=1 ,im:size(1) do
            for j=1,im:size(2) do
                for k=1,im:size(3) do
                    im_wrap[i][start+j][start+k] = im[i][j][k]
                end 
            end 
        end 
    else
        im_wrap = image.scale(im,size,size)
    end 
    return im_wrap
end

	
