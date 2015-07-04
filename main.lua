----------------------------------------------------------------------
--
-- Prior to using this script, we need to generate the datasets with 
-- createDataSet.lua, and pre-process them using preProcess.lua.
--
-- It uses the ConvNet model described in model.lua
--
-- Required :
--      + model described in model.lua
--      + validation set loaded with dataset.lua
--
-- These results are based on Yann Lecun et al. work :
-- http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/
-- reference_papers/2-sermanet-ijcnn-11-mscnn.pdf
--
-- Hugo Duthil
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a double 2-stage ConvNet with 1st-stage output fully connected to the final 2-layer MLP classifier')
cmd:text()
cmd:text('Options')
-- general parameters
cmd:option('-lr',0.1,'learning rate')
cmd:option('-batch_size',1,'batch size for SGD')
cmd:option('-use_3_channels',false, 'use YUV channels or just Y channel in the computation')
cmd:option('-save_model_iterations',200, 'save model each xxx batch iterations , 0 if you don\'t want to save the model')
cmd:option('-model_name','model.t7', 'save model  under this name')
cmd:option('-save_f_iterations',200, 'save objective function each xxx iterations, 0 if you don\'t want to save the model')
cmd:option('-f_name','f.t7', 'save objective function  under this name')
cmd:option('-use_3_channels',false, 'use YUV channels or just Y channel in the computation')
cmd:option('-use_validation_set',false, 'use validation set if you don\'t have an annoted test set')
-- set path files
cmd:option('-train_set', 'gtsrb_train.t7','path of the training set')
cmd:option('-valid_set', 'gtsrb_valid.t7','path of the validation set')
cmd:option('-test_set',  'gtsrb_test.t7','path of the test set')
-- pre processing options
cmd:option('-no_global_contrast_norm',false,'don\'t use global contrast normalization for pre-processing')
cmd:option('-no_local_contrast_norm',false,'don\'t use local contrast normalization for pre-processing')
cmd:text()

-- parse input params
params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})

local script_dir = paths.dirname(paths.thisfile()).."/"

-- load training set, validation set and testing set
print("---------------- loading datasets ---------------")
dofile(script_dir.."dataset.lua")

-- pre-process these image sets, using per-image global normalization
-- and local contrast normalization of Y channel
print("\n--------------- preprocessing data -------------")
dofile(script_dir.."preprocess.lua")

-- load model
print("\n----------------- loading model ----------------")
dofile(script_dir.."model.lua")

-- load training function
dofile(script_dir.."train.lua")
print("\n---------- training function loaded -----------")
print("Use train() or change some parameters to perform a full SGD on the training set (params : batch_size, learning_rate, ...)")

-- load validation function
dofile(script_dir.."validate.lua")
print("\n--------- validation function loaded ----------")
print("Use validate() to check the model performances on the validation set")

-- load test function
dofile(script_dir.."test.lua")
print("\n------------ test function loaded -------------")
print("Use test() to check the model performances on the test set")
