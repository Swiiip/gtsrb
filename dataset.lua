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

train_file         =   script_dir..params.train_set               -- path to the training set
valid_file         =   script_dir..params.valid_set               -- path to the validation set
test_file          =   script_dir..params.test_set                -- path to the test set
use_validation_set =   params.use_validation_set

print(train_file)

-- Set the default type of Tensor to float
torch.setdefaulttensortype('torch.FloatTensor')

if paths.filep(train_file) then                         -- check if set exists
    if not train_set then
        print("\nLoading training set")
        train_set = torch.load(train_file) 
        function train_set:size() return #train_set end
        print("Training set loaded")
    else
        print("\nTraining set already loaded")
    end
else
    print("\nNo training set found")
end
if paths.filep(valid_file) then                         -- check if set exists
    if not valid_set then
        print("\nLoading validation set")
        valid_set = torch.load(valid_file) 
        function valid_set:size() return #valid_set end
        print("Validation set loaded")
    else
        print("\nValidation set already loaded")
    end
else
    print("\nNo validation set found")
end

if paths.filep(test_file) then                         -- check if set exists
    if not test_set then
        print("\nLoading test set")
        test_set = torch.load(test_file) 
        function test_set:size() return #test_set end
        print("Test set loaded")
    else
        print("\nTest set already loaded")
    end
else
    print("\nNo test set found")
end
