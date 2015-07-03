----------------------------------------------------------------------
-- This script creates the training, validation and test sets for GTSRB
-- challenge from classes image directories.
--
-- train_images_dir must be like [train_images_dir]/[classes]/[img.jpg, png, ...]
-- test_images_dir  must be like [test_images_dir]/[img.jpg, png, ...]
--
-- Images from the raw set are :
--      + scaled to 32x32
--      + converted to YUV space
--
-- 3 files are created :
--      + a training set (default "gtsrb_train.t7")
--      + a validation set - 1 track per class - (default "gtsrb_valid.t7")
--      + a testation set (default "gtsrb_test.t7")
--
-- We extract 1 track (30 images) at random per class for validation, 
-- yielding 1,290 samples for validation and 37,919  for training.
--
-- If we have an already labeled test set, we don't need to use a 
-- validation set
--
-- These results are based on Yann Lecun et al. work :
-- http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/
-- reference_papers/2-sermanet-ijcnn-11-mscnn.pdf
--
-- Hugo Duthil
----------------------------------------------------------------------

require 'torch'                                               -- torch
require 'image'                                               -- color transforms and image load
require 'paths'                                               -- path manipulation
require 'nn'                                                  -- neural networks
require 'csvigo'                                              -- csv parsing

samples = 30                                                  -- # of samples/track

local script_dir = paths.dirname(paths.thisfile()).."/"

train_images_dir   =   script_dir.."GTSRB/Final_Training/Images/"       -- directory of the class directories for the training set
test_images_dir    =   script_dir.."GTSRB/Final_Test/Images/"           -- directory of the images for the testation set
test_set_labels    =   script_dir.."GTSRB/Final_Test/GT-final_test.csv" -- labels for the test set
train_file         =   script_dir.."gtsrb_train.t7"                     -- name of the training set
valid_file         =   script_dir.."gtsrb_valid.t7"
test_file          =   script_dir.."gtsrb_test.t7"                      -- name of the test set
use_validation_set =   false


-- set the default type of Tensor to float
torch.setdefaulttensortype('torch.FloatTensor')

-- Training and Validation sets
-- parsing of all the images in classes directories
if not paths.filep(train_file) or not paths.filep(valid_file) and use_validation_set then                 -- check if sets already exists
    print("Training set not found, building a new one")

    -- utility tables
    local train_imgs = {}
    local valid_imgs = {}
    local m=1
    local classes = paths.files(train_images_dir)                                          -- iterator on classes directories
    for c in classes do                                                                    -- for each class directory, add images to db
        -- skip ".", "..", ".DS_Store" on mac
        if not c:find("^%.") then

            local built_dir = paths.concat(train_images_dir, c)                            -- path to the class directory
            local images = paths.dir(built_dir)                                            -- table of image names 
            local random_track = math.random(1, math.floor(0.5 + (#images-3)/samples))     -- select a random track for validation set
            print("Adding class "..m.." to training set ("..(#images - 3).." images)")     -- #images = #files - ., .. and *.csv
            if use_validation_set then print("- Validation track : "..random_track) end
            local img_nbr = 0
            for i, img in ipairs(images) do
                if not img:find("^%.") and not img:find(".csv") then
                    local cur_dir = paths.concat(built_dir, img)                           -- full path to the image
                    local cur_img = image.load(cur_dir)
                    cur_img = image.scale(cur_img, 32, 32)                                 -- scales the input images
                    cur_img = image.rgb2yuv(cur_img)                                       -- map from RGB to YUV space
                    if use_validation_set then
                        if math.floor(img_nbr/30) + 1 == random_track then
                            valid_imgs[#valid_imgs + 1 ] = {cur_img, m}                    -- add image to the list of validation images
                        else
                            train_imgs[#train_imgs + 1 ] = {cur_img, m}                    -- add image to the list of training images
                        end
                    else
                        train_imgs[#train_imgs + 1 ] = {cur_img, m}                        -- add image to the list of training images
                    end
                    img_nbr = img_nbr+1
                end
            end
            m = m+1
        end
    end

    -- save training set on disk
    print("\nSaving training set ...")
    torch.save(train_file, train_imgs)
    print("Saved under "..train_file)
    print("Number of training examples : "..#train_imgs)

    if use_validation_set then
        -- save validation set on disk
        print("\nSaving validation set ...")
        torch.save(valid_file, valid_imgs)
        print("Saved under "..valid_file)
        print("Number of training examples : "..#valid_imgs)
    end
else
    print("\nTraining and validation  sets already exist : ")
    print(train_file)
    print(valid_file)
end

-- Test set
-- parsing images in directory
if not paths.filep(test_file) then                             -- check if set already exists
    print("\nTest set not found, building a new one")

    -- loading labels 
    if paths.filep(test_set_labels) then
        local query = csvigo.load({path = test_set_labels, separator = ";", mode = "query", verbose = false })
        local labels = query().ClassId

        -- utility tables
        local test_imgs = {}
        local m =1
        local images = paths.dir(test_images_dir)                               -- table of image names 
        for i, img in ipairs(images) do
            if not img:find("^%.") and not img:find(".csv") then
                local cur_dir = paths.concat(test_images_dir, img)              -- path to the image
                local cur_img = image.load(cur_dir)
                cur_img = image.scale(cur_img, 32, 32)                          -- scales the input images
                cur_img = image.rgb2yuv(cur_img)                                -- map from RGB to YUV space
                test_imgs[#test_imgs + 1 ] = {cur_img, tonumber(labels[m])}     -- add image to the list of training images
                print(labels[m])
                m = m+1
            end
        end

        -- save test set on disk
        print("\nSaving test set ...")
        torch.save(test_file, test_imgs)
        print("Saved under "..test_file)
        print("Number of test examples : "..#test_imgs)
    else
        print("No test labels loaded")
    end
else
    print("\nTest set already exists : ")
    print(test_file)
end

