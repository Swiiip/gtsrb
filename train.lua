----------------------------------------------------------------------
-- This script contains the main training function 
--
-- Prior to using this script, we need to generate the datasets with 
-- createDataSet.lua, and pre-process it using preProcess.lua.
--
-- It uses the ConvNet model described in model.lua
--
-- Required :
--      + model described in model.lua
--      + training set loaded with dataset.lua
--
-- These results are based on Yann Lecun et al. work :
-- http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/
-- reference_papers/2-sermanet-ijcnn-11-mscnn.pdf
--
-- Hugo Duthil
----------------------------------------------------------------------

require 'torch'
require 'optim'
require 'paths'

local time = sys.clock()

-- Parameters
batch_size         = params.batch_size                                            -- batch size
learning_rate      = params.lr                                                    -- learning rate
save_model_it      = params.save_model_iterations                                 -- save model every 200 batch iterations
model_file         = paths.dirname(paths.thisfile()).."/"..params.model_name      -- save model under this path
record_f_it        = params.save_f_iterations                                     -- save f every 200 batch iterations
f_file             = paths.dirname(paths.thisfile()).."/"..params.f_name          -- save objective function graph under this path

-- Classes
classes = {}
for i = 1, 43 do classes[i] = i.."" end

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- get the learnable parameters of the model and the gradient of the cost function
-- with respect to the learnable parameters
if model then 
    parameters, gradParameters= model:getParameters()
else
    print("No model found, please load a model with model.lua")
end

-- visualize ojective function with itorch
saved_f = {}


-- Main taining loop accross the entire dataset
-- The optimization method is a classic batch sgd
function train()
    print("Training network")
    local m = 1

    -- shuffle the training set
    shuffle = torch.randperm(train_set:size())

    for t =1, train_set:size(), batch_size do
        -- progress bar
        xlua.progress(t, train_set:size())

        -- table containing batch examples
        local batch_examples = {}
        for i=t, math.min(t+batch_size-1, train_set:size()) do
            batch_examples[#batch_examples+1] = train_set[shuffle[i]]
        end

        -- reset gradients
        model:zeroGradParameters()

        -- objective function
        local f = 0 

        -- compute gradient for the batch
        for i=1, #batch_examples do
            -- extract Y channel
        if use_3_channels then
            local input = train_set[shuffle[i]][1]
        else
            local input = train_set[shuffle[i]][1][{{1}, {}, {}}]
        end
            
            -- extract corresponding label
            local label = batch_examples[i][2]
            
            -- forward propagation of the input through the model
            local output = model:forward(input)

            -- accumulate f
            f = f+criterion:forward(output, label)

            -- estimate df/dw
            local df_d0 = criterion:backward(output, label)
            model:backward(input, df_d0)

            -- update confusion matrix
            confusion:add(output, label)
        end

        -- normalize gradients and f(X)
        gradParameters:div(#batch_examples)
        f = f/#batch_examples

        -- save model every save_model_it iterations
        if m%save_model_it == 0 and save_model_it ~= 0 then
            torch.save(paths.thisfile()..model_file,model)
        end
        -- record f every record_f_it iterations
        if m%record_f_it == 0 and record_f_i ~= 0 then
            table.insert(paths.thisfile()..saved_f, f)
            torch.save(f_file,saved_f)
        end
        model:updateParameters(learning_rate)
        m =m+1

    end

    -- print confusion matrix
    print(confusion)
    confusion:zero()
end
