----------------------------------------------------------------------
-- This script contains the main validation function 
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

require 'torch'

function validate()
    print("Testing network")
    -- shuffle the validation test
    shuffle = torch.randperm(valid_set:size())

    for i=1, valid_set:size() do
        -- progress bar
        xlua.progress(i, valid_set:size())

        -- extract Y channel
        if use_3_channels then
            local input = valid_set[shuffle[i]][1]
        else
            local input = valid_set[shuffle[i]][1][{{1}, {}, {}}]
        end
        local label = valid_set[shuffle[i]][2]

        -- add prediction to confusion matrix
        confusion:add(model:forward(input), label)
    end

    print(confusion)
    confusion:zero()
end
