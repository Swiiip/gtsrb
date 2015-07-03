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
    print("Testing network")
    
    -- shuffle the validation test
    shuffle = torch.randperm(test_set:size())

    for i=1, test_set:size() do--test_set:size() do
        -- progress bar
        xlua.progress(i, test_set:size())

        -- extract Y channel
        if use_3_channels then
            local input = test_set[shuffle[i]][1]
        else
            local input = test_set[shuffle[i]][1][{{1}, {}, {}}]
        end
        print(shuffle[i])
        print(test_set[9956])
        local label = test_set[shuffle[i]][2]
        print(label)
        -- add prediction to confusion matrix
         confusion:add(model:forward(input), label)
    end

    print(confusion)
    confusion:zero()
end
