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


global_contrast_norm = params.global_contrast_norm
local_contrast_norm  = params.local_contrast_norm

if train_set then

    -- Global normalization and local contrast
    -- normalization of Y channel

    --  utility tensors
    local test_set_tensor  = torch.Tensor(test_set:size(), 3, 32, 32)
    local train_set_tensor = torch.Tensor(train_set:size(), 3, 32, 32)

    -- define the normalization neighborhood:
    local neighborhood = image.gaussian1D(7)

    -- Define the normalization operator
    normalization = dp.LeCunLCN({progress = true, kernel = neighborhood, channels = {1}}) 

    print("\nNormalization of training set")

    -- per-example Y channel mean substraction
    if global_contrast_norm then
        print("Global normalization")
        
        -- normalize for each exemple
        for i = 1,train_set:size() do
            xlua.progress(i, train_set:size())
            local img = train_set[i][1][{1, {}, {}}]
            local mean = img:mean()
            --mean substraction
            img:add(-mean)
            local std = img:std()
            -- std division
            img:div(std)

            train_set_tensor[i]:set(img)
        end
    end

    -- local contrast normalization of Y channel:
    if local_contrast_norm then
        normalization:apply(dp.ImageView("bchw", train_set_tensor))
    end

    -- if we use a validation set
    if valid_set then
        print("\nNormalization of validation set")
        
        --utility tensor
        local valid_set_tensor = torch.Tensor(valid_set:size(), 3, 32, 32)
        
        if global_contrast_norm then
            print("Global normalization")

            -- per-example Y channel mean substraction
            for i = 1,valid_set:size() do
                xlua.progress(i, valid_set:size())
                local img = valid_set[i][1][{1, {}, {}}]
                local mean = img:mean()
                img:add(-mean)
                local std = img:std()
                img:div(std)
                valid_set_tensor[i]:set(img)
            end
        end
        -- local contrast normalization of Y channel:
        if local_contrast_norm then
            normalization:apply(dp.ImageView("bchw", valid_set_tensor))
        end
    end

    print("Normalization of test set")

    if global_contrast_norm then
        print("Global normalization")
        -- per-example mean substraction
        for i = 1,test_set:size() do
            xlua.progress(i, test_set:size())
            local img = test_set[i][1][{1, {}, {}}]
            local mean = img:mean()
            img:add(-mean)
            test_set_tensor[i]:set(img)
        end
    end

    -- local contrast normalization of Y channel:
    if local_contrast_norm then
        normalization:apply(dp.ImageView("bchw", test_set_tensor))
    end
else
    print("Databases missing, please run createDataSet.lua to build the databases, and load them with dataset.lua")
end
