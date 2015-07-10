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

-- if we use already preprocessed data sets
if use_pp_sets then 
    tr_file = pp_train_file
    ts_file = pp_test_file
else
    tr_file = train_file
    ts_file = test_file
end

-- check if train set already exists
if paths.filep(tr_file) then                         
    if not train_set then
        print("\nLoading training set")
        train_set = torch.load(tr_file) 
        function train_set:size() return #train_set end
        print(tr_file)
        print("Training set loaded")
    else
        print("\nTraining set already loaded")
    end
else
    print("\nNo training set found")
end

if paths.filep(ts_file) then

    -- check if test set already exists
    if not test_set then
        print("\nLoading test set")
        test_set = torch.load(ts_file) 
        function test_set:size() return #test_set end
        print(ts_file)
        print("Test set loaded")
    else
        print("\nTest set already loaded")
    end
else
    print("\nNo test set found")
end
