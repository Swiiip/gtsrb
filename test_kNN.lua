----------------------------------------------------------------------
-- This is another testing function, which uses a 1NN classifier
-- on features extracted from an already trained model.
-- 
-- Input:
--      + mdl_output_layer_idx : index of the output module you want to consider as input features
--
-- Hugo Duthil
----------------------------------------------------------------------

local script_dir = paths.dirname(paths.thisfile()).."/"

function test_kNN(mdl_output_layer_idx)
    -- Classes
    local classes = {}
    for i = 1, 43 do classes[i] = (i-1).."" end

    -- this matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()

    local pwd = nn.PairwiseDistance(2)

    -- compute average example for each class
    local res = {}
    local exp_count = {}
    for i=1, 43 do exp_count[i] = 0 end

    print("Building reference table")

    for i=1, train_set:size() do
        xlua.progress(i, train_set:size())
        local sample = train_set[i]
        if use_3_channels then
            model:forward(sample[1])
        else
            model:forward(sample[1][{ {1}, {}, {} }])
        end
        local output = model:get(mdl_output_layer_idx).output
        local clas = sample[2]
        if res[clas] then 
            res[clas]:add(torch.Tensor(output:clone()))
        else
            res[clas] = torch.Tensor(output:clone())
        end
        exp_count[clas] = exp_count[clas] + 1
    end

    for i=1, 43 do res[i]:div(exp_count[i]) end

    print("Matching testing ...")

    for t=1, test_set:size() do
        xlua.progress(t, test_set:size())
        local test_sample = test_set[t]
        if use_3_channels then
            model:forward(test_sample[1])
        else
            model:forward(test_sample[1][{ {1}, {}, {} }])
        end
        local output = model:get(mdl_output_layer_idx).output
        local min_dist = 9999999
        local best_class = 1
        for i=1, #res do
            local r_reshaped = torch.reshape(res[i], #res[i]:storage())
            local o_reshaped = torch.reshape(output, #output:storage())
            local dist = pwd:forward({o_reshaped,r_reshaped})[1]
            if dist < min_dist then
                min_dist = dist
                best_class = i
            end
            confusion:add(best_class, test_sample[2])
        end
    end
    print(confusion)
    torch.save("saves/confusion.t7", confusion)
    confusion:zero()
end
