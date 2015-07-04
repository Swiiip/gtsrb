----------------------------------------------------------------------
-- This script contains the main test function 
--
-- Prior to using this script, we need to generate the datasets with 
-- createDataSet.lua, and pre-process them using preProcess.lua.
--
-- It uses the ConvNet model described in model.lua
--
-- Required :
--      + model described in model.lua
--      + test set loaded with dataset.lua
--
-- These results are based on Yann Lecun et al. work :
-- http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/
-- reference_papers/2-sermanet-ijcnn-11-mscnn.pdf
--
-- Hugo Duthil
----------------------------------------------------------------------

require 'torch'
require 'optim'

function test()
    confusion:zero()
    print("Testing network")
    
    -- shuffle the validation test
    shuffle = torch.randperm(test_set:size())

    for i=1, test_set:size() do--test_set:size() do
        -- progress bar
        xlua.progress(i, test_set:size())
        local input
        -- extract Y channel
        if params.use_3_channels then
            input = test_set[shuffle[i]][1]
        else
            input = test_set[shuffle[i]][1][{{1}, {}, {}}]
        end
        local label = test_set[shuffle[i]][2]
        -- add prediction to confusion matrix
         confusion:add(model:forward(input), label)
    end

    print(confusion)
    confusion:zero()
end
