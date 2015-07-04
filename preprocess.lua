----------------------------------------------------------------------
-- This script contains the pre-processing steps needed to ensure a 
-- quick learning.
--
-- Prior to using this script, we need to generate the datasets with 
-- createDataSet.lua, and load them with dataset.lua.
--
-- The Y channel of each image is preprocessed with global and local 
-- contrast normalization
--
-- These results are based on Yann Lecun et al. work :
-- http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/
-- reference_papers/2-sermanet-ijcnn-11-mscnn.pdf
--
-- Hugo Duthil
----------------------------------------------------------------------

require 'torch'                               -- torch
require 'image'                               -- for image transforms
require 'dp'                                  -- provides all sorts of preprocessing modules (LeCun LCN)


global_contrast_norm = not params.no_global_contrast_norm
local_contrast_norm  = not params.no_local_contrast_norm

if train_set then

    -- Global normalization and local contrast
    -- normalization of Y channel

    --  utility tables

    local train_tensors_list = {}
    local valid_tensors_list = {}
    local test_tensors_list = {}
    
    -- define the normalization neighborhood:
    local neighborhood = image.gaussian1D(7)

    -- define the normalization operator
    normalization = dp.LeCunLCN({progress = true, kernel = neighborhood, channels = {1}}) 

    print("\nNormalization of training set")

    -- per-example Y channel mean substraction
    -- normalize for each exemple
    for i = 1,train_set:size() do
        
        local img = train_set[i][1][{{1}, {}, {}}]
        if global_contrast_norm then
            xlua.progress(i, train_set:size())
            local mean = img:mean()
            
            --mean substraction
            img:add(-mean)
            local std = img:std()
            
            -- std division
            img:div(std)
        end
        train_tensors_list[i] = torch.Tensor(img)
    end
    local train_set_tensor = nn.JoinTable(1):forward(train_tensors_list)
    train_set_tensor = nn.Reshape(train_set:size(), 1, 32, 32):forward(train_set_tensor)

    -- local contrast normalization of Y channel:
    if local_contrast_norm then
        normalization:apply(dp.ImageView("bchw", train_set_tensor))
    end

    -- if we use a validation set
    if valid_set then
        print("\nNormalization of validation set")

        -- per-example Y channel mean substraction
        for i = 1,valid_set:size() do
            local img = valid_set[i][1][{ {1}, {}, {} }]
            if global_contrast_norm then
                xlua.progress(i, valid_set:size())
                local mean = img:mean()
                img:add(-mean)
                local std = img:std()
                img:div(std)
            end
            valid_tensors_list[i] = torch.Tensor(img)
        end
        local valid_set_tensor = nn.JoinTable(1):forward(valid_tensors_list)
        valid_set_tensor = nn.Reshape(valid_set:size(), 1, 32, 32):forward(valid_set_tensor)

        -- local contrast normalization of Y channel:
        if local_contrast_norm then
            normalization:apply(dp.ImageView("bchw", valid_set_tensor))
        end
    end

    print("Normalization of test set")

    -- per-example mean substraction
    for i = 1,test_set:size() do
        local img = test_set[i][1][{{1}, {}, {}}]
        if global_contrast_norm then
            xlua.progress(i, test_set:size())
            local mean = img:mean()
            img:add(-mean)
            local std = img:std()
            img:div(std)
        end
        test_tensors_list[i] = torch.Tensor(img)
    end
    local test_set_tensor = nn.JoinTable(1):forward(test_tensors_list)
    test_set_tensor = nn.Reshape(test_set:size(), 1, 32, 32):forward(test_set_tensor)

    -- local contrast normalization of Y channel:
    if local_contrast_norm then
        normalization:apply(dp.ImageView("bchw", test_set_tensor))
    end
else
    print("Databases missing, please run createDataSet.lua to build the databases, and load them with dataset.lua")
end
