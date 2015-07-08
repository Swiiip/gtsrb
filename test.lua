----------------------------------------------------------------------
-- This script contains the main test function 
--
-- It uses the model described in models/MSmodel.lua
--
-- Required :
--      + model 
--      + test_set loaded with dataset.lua
--
-- Hugo Duthil
----------------------------------------------------------------------

require 'torch'
require 'optim'

function test()
    -- Classes
    classes = {}
    for i = 1, 43 do classes[i] = (i-1).."" end

    -- this matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()
    print("Testing network")

    -- shuffle validation test
    shuffle = torch.randperm(test_set:size())

    for i=1, test_set:size() do
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

        -- the output needs to a xxxStorage of size 43 (not 1x43)
        local output = model:forward(input) 

        -- add prediction to confusion matrix
        confusion:add(output, label)
    end

    print(confusion)
    torch.save("saves/confusion.t7", confusion)
    confusion:zero()
end
