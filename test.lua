require 'torch'
require 'csvigo'

dofile('1_datafunctions.lua')

submissionfile = './results/test.csv'
test_file = io.open('../data/plankton/test.csv','r')
io.input(test_file)
class_title_s = io.read()
test_class_title = split(class_title_s,' ')
print(test_class_title[1])
io.close()

test_csvdata = csvigo.load{path ='../data/plankton/test.csv',separator = ' ',verbose = false}
tesize = #test_csvdata[ test_class_title[1] ]
print(tesize)

--predict_csvdata = {}
--[=[
for i=1,10 do
    local imagename = tostring(test_csvdata[test_class_title[1]][i]) .. ".jpg"
    test_csvdata[test_class_title[1]][i] = imagename
    print(imagename)
	for j=2,#test_class_title do
		test_csvdata[test_class_title[j]][i] = 0.5
	end
end
--]=]
--csvigo.save{path = submissionfile,data = test_csvdata,verbose = false}
--csvigo.save{path = submissionfile,data = test_csvdata,mode ='tidy'}

fp = io.open(submissionfile,"w")
for i =1,10 do
    local line = ""
	for j=1,#test_class_title do
		if i==1 then line = line..','..test_class_title[j]
        elseif j==1 then line = line..','..test_csvdata[test_class_title[1]][i]..'.jpg'
		else line = line..','..'0.5' end
	end
	line = string.sub(line,2)
	print(line)
    fp:write(line.."\n")
end
fp:close()

