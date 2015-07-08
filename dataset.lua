----------------------------------------------------------------------
-- Use this script to define the training and validating datasets.
--
-- A dataset is an object which implements the operator dataset[index] 
-- and the method dataset:size(). The size() methods returns the number 
-- of examples and dataset[i] has to return the i-th example.
--
-- An example has to be an object which implements the operator 
-- example[field], where field might take the value 1 (input features) 
-- or 2 (corresponding label which will be given to the criterion)
--
-- Hugo Duthil
----------------------------------------------------------------------

-- Each data structure is a table of 2-field elements :
--      + [1]  : a 3D tensor {channel, x, y} representing a 32x32 YUV image
--      + [2]  : image label

require 'torch'
require 'paths'

local script_dir = paths.dirname(paths.thisfile()).."/"

train_file            =   script_dir..params.train_set                  -- path to the training set
test_file             =   script_dir..params.test_set                   -- path to the test set
pp_train_file         =   script_dir..params.pp_train_set               -- path to the training set
pp_test_file          =   script_dir..params.pp_test_set                -- path to the training set
use_pp_sets           =   params.use_pp_sets                            -- load already preprocessed sets


-- Set the default type of Tensor to float
torch.setdefaulttensortype('torch.FloatTensor')

-- if we don't use already preprocessed data sets
if not use_pp_sets then 

    -- check if train set already exists
    if paths.filep(train_file) then                         
        if not train_set then
            print("\nLoading training set")
            train_set = torch.load(train_file) 
            function train_set:size() return #train_set end
            print(train_file)
            print("Training set loaded")
        else
            print("\nTraining set already loaded")
        end
    else
        print("\nNo training set found")
    end

    if paths.filep(test_file) then

        -- check if test set already exists
        if not test_set then
            print("\nLoading test set")
            test_set = torch.load(test_file) 
            function test_set:size() return #test_set end
            print(test_file)
            print("Test set loaded")
        else
            print("\nTest set already loaded")
        end
    else
        print("\nNo test set found")
    end

    -- if we use preprocessed data sets, load them    
else
    
    -- check if preprocessed train set already exists
    if paths.filep(pp_train_file) then        
        if not train_set then
            print("\nLoading preprocessed training set")
            train_set = torch.load(pp_train_file) 
            function train_set:size() return #train_set end
            print("Training set loaded")
            print(pp_train_file)
        else
            print("\nTraining set already loaded")
        end
    else
        print("\nNo training set found")
    end

    -- check if preprocessed test set already exists
    if paths.filep(pp_test_file) then 
        if not test_set then
            print("\nLoading preprocessed test set")
            test_set = torch.load(pp_test_file) 
            function test_set:size() return #test_set end
            print(pp_test_file)
            print("Test set loaded")
        else
            print("\nTest set already loaded")
        end
    else
        print("\nNo test set found")
    end
end
