----------------------------------------------------------------------
-- Prior to using this script, we need to generate the datasets with 
-- createDataSets.lua
--
-- This script loads the main global variables into Luajit :
--      + train_set          : loaded by dataset.lua
--      + test_set           : loaded by dataset.lua
--      + model              : loaded by models/MSmodel.lua
--      + learning_rate      : loaded by train.lua
--      + batch_size         : loaded by train.lua
--
-- Required :
--      + models/MSmodel.lua : describes a multi-scale architecture
--      + dataset.lua        : loads the training and test sets in Luajit
--      + train.lua          : loads a train() function using SGD optimization method
--      + test.lua           : loads a test() function describing the testing method
--
-- This programming architecture is modular, you can use your own preprocessing/train/test functions
-- as well as your models, as long as they respect the model/dataset interface described 
-- in the corresponding files (dataset.lua, models/MSmodel.lua, ...)
--
-- Just run th -i main.lua to load the elements from the different modules and start interactively
-- changing the model parameters, loading an aldready trained model, tweaking the parameters
-- (learning_rate, batch_size, ...) and using the train() and test() functions.
--
-- Run th main.lu -help to see the available options.
--
-- Hugo Duthil
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Canevas for training and testing cNN networks on gtsrb database')
cmd:text()
cmd:text('Options')
-- general parameters
cmd:option('-lr',0.1,'initial learning rate')
cmd:option('-batch_size',1,'batch size for SGD')
cmd:option('-use_pp_sets',false,'use already preprocessed data sets, you can specify their paths using -pp_train_set and pp_test_set')
cmd:option('-use_3_channels',false, 'use YUV channels or just Y channel in the computation')
cmd:option('-save_model_iterations',200, 'save model each xxx batch iterations , 0 if you don\'t want to save the model')
cmd:option('-model_name','saves/model.t7', 'save model in this path, the folder must exist')
cmd:option('-save_f_iterations',200, 'save objective function each xxx iterations, 0 if you don\'t want to save the model')
cmd:option('-f_name','saves/f.t7', 'save objective function  in this path, the folder must exist')
cmd:option('-use_3_channels',false, 'use YUV channels or just Y channel in the computation')
-- set path files
cmd:option('-train_set', 'sets/gtsrb_train.t7','path of training set')
cmd:option('-test_set',  'sets/gtsrb_test.t7','path of test set')
cmd:option('-pp_train_set','sets/gtsrb_pp_train_set.t7','path of preprocessed train set')
cmd:option('-pp_test_set','sets/gtsrb_pp_test_set.t7','path of preprocessed test set')
cmd:option('-train_func','train.lua','path of train() function')
cmd:option('-test_func','test.lua','path of test() function')
cmd:option('-model','models/MSmodel.lua', 'path to the model you want to use')
-- pre processing options
cmd:option('-no_global_contrast_norm',false,'don\'t use global contrast normalization for pre-processing')
cmd:option('-no_local_contrast_norm',false,'don\'t use local contrast normalization for pre-processing')
cmd:text()

-- parse input params
params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})

local script_dir = paths.dirname(paths.thisfile()).."/"

-- load training and testing set
print("---------------- loading datasets ---------------")
dofile(script_dir.."dataset.lua")

-- pre-process these image sets, using per-image global normalization
-- and local contrast normalization of Y channel
if not params.use_pp_sets then
print("\n--------------- preprocessing data -------------")
dofile(script_dir.."preprocess.lua")

    print("\nDo you want to save preprocessed datasets [y/n]? ( in "..params.pp_train_set.." and "..params.pp_test_set.." )")
    if io.read() == "y" then

        -- saving pp training set --
        print("\nSaving preprocessed training set ...")
        torch.save(script_dir..params.pp_train_set, train_set)
        print("Saved under "..params.pp_train_set)


        -- saving pp training set --
        print("\nSaving preprocessed test set ...")
        torch.save(script_dir..params.pp_test_set, test_set)
        print("Saved under "..params.pp_test_set)
    end
end


-- load model
print("\n----------------- loading model ----------------")
dofile(script_dir..params.model)
print("Model loaded :")
print(model)

-- load training function
dofile(script_dir..params.train_func)
print("\n---------- training function loaded -----------")
print("Use train() or change some parameters to perform a full SGD on the training set (params : batch_size, learning_rate, ...)")

-- load test function
dofile(script_dir..params.test_func)
print("\n------------ test function loaded -------------")
print("Use test() to check the model performances on the test set")
