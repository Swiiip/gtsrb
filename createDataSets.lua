----------------------------------------------------------------------
-- This script creates the training and test sets for GTSRB
-- challenge from classes image directories and a csv file
-- describing test images labels.
--
-- Images from the raw set are :
--      + scaled to 32x32
--      + converted to YUV space
--
-- 2 files are created :
--      + a training set (default "sets/gtsrb_train.t7")
--      + a test set (default "sets/gtsrb_test.t7")
--
-- Just run th createDataSets.lua and it will create :
--      + sets/gtsrb_train.t7 : training set file
--      + sets/gtsrb_test.t7  : test set file
--
-- Run th createDataSets.lua -help for help on option commands
--
-- Hugo Duthil
----------------------------------------------------------------------

-- Parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("Creates train set and test set from zip files for gtsrb challenge")
cmd:text()
cmd:text('Options')
-- general parameters
cmd:option('-zip_train_dir',"GTSRB_Final_Training_Images.zip",'Directory of the train set zip file')
cmd:option('-zip_test_dir',"GTSRB_Final_Test_Images.zip",'Directory of the test set zip file')
cmd:option('-csv_labels_dir',"GT-final_test.csv",'Directory of the csv for test set labels')
cmd:option('-save_train',"sets/gtsrb_train.t7",'Save train set under this path')
cmd:option('-save_test',"sets/gtsrb_test.t7",'Save train set under this path')
cmd:text()

-- parse input params
params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})

require 'torch'                                               -- torch
require 'image'                                               -- color transforms and image load
require 'paths'                                               -- path manipulation
require 'nn'                                                  -- neural networks
require 'csvigo'                                              -- csv parsing

samples = 30                                                  -- # of samples/track

local script_dir = paths.dirname(paths.thisfile()).."/"

train_images_dir   =   script_dir..params.zip_train_dir           -- directory of the zipped training set
test_images_dir    =   script_dir..params.zip_test_dir            -- directory of the images for the testation set
test_set_labels    =   script_dir..params.csv_labels_dir           -- labels for the test set
train_file         =   script_dir..params.save_train               -- name of the training set
test_file          =   script_dir..params.save_test                -- name of the test set


-- set the default type of Tensor to float
torch.setdefaulttensortype('torch.FloatTensor')

-- download training set
if not paths.filep(params.zip_train_dir) then
    os.execute("wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip")
end

-- download test set
if not paths.filep(params.zip_test_dir) then
    os.execute("wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip")
end

-- download test labels
if not paths.filep(params.csv_labels_dir) then
    os.execute("wget https://raw.githubusercontent.com/Swiiip/gtsrb/master/GT-final_test.csv")
end

-- Training set
-- parsing of all the images in the dir of zipped training set
if not paths.filep(train_file) then                                                  -- check if training set already exists
    print("\nTraining set not found, building a new one")
    print("Unzipping "..train_images_dir..' in tmp_train/')
    os.execute('unzip -q '..train_images_dir..' -d "tmp_train/"')
    print("Unzipped\n")

    -- container for images
    local train_imgs = {}
    local images_dir = script_dir.."tmp_train/GTSRB/Final_Training/Images" 
    local classes = paths.files(images_dir)                                          -- iterator on classes directories

    local m=1
    for c in classes do
        -- skip ".", "..", ".DS_Store" on mac
        if not c:find("^%.") then

            -- path to the class directory
            local built_dir = paths.concat(images_dir, c)                           

            -- table of image names
            local images = paths.dir(built_dir)

            local img_nbr = 0

            for i, img in ipairs(images) do
                if not img:find("^%.") and not img:find(".csv") then

                    -- full path to the image
                    local cur_dir = paths.concat(built_dir, img)                           
                    local cur_img = image.load(cur_dir)

                    -- crop along min_dim to have a squared image and keep the ratio as we scale it
                    local min_dim = math.min(cur_img:size(2), cur_img:size(3))
                    cur_img = image.crop(cur_img, 0, 0, min_dim, min_dim)

                    -- scaling image to 32x32
                    cur_img = image.scale(cur_img, 32, 32)                           

                    -- map from rgb to yuv space
                    cur_img = image.rgb2yuv(cur_img)                                 

                    -- add image to the list of training images
                    train_imgs[#train_imgs + 1 ] = {cur_img, m}                      

                    img_nbr = img_nbr+1
                end
            end
            print("Added class "..m.." to training set ("..img_nbr.." images)")
            m = m+1
        end
    end

    -- save training set on disk
    print("\nSaving training set ...")
    torch.save(train_file, train_imgs)
    print("Saved under "..train_file)
    print("Number of training examples : "..#train_imgs)

    print("\nDo you want to remove tmp_train/ temporary dir [y/n]?")
    if(io.read() == "y") then
        os.execute("rm -r tmp_train/")
        print("tmp_train/ removed")
    else
        print("tmp_train kept")
    end

else
    print("\nTraining set already exists : ")
    print(train_file)
end

-- Test set
-- parsing images in directory
if not paths.filep(test_file) then                             -- check if set already exists
    print("\nTest set not found, building a new one")

    -- loading labels 
    if paths.filep(test_set_labels) then
        local query = csvigo.load({path = test_set_labels, separator = ";", mode = "query", verbose = false })
        local labels = query().ClassId

        -- Unizp test set
    print("Unzipping "..test_images_dir..' in tmp_test/')
    os.execute('unzip -q '..test_images_dir..' -d "tmp_test/"')
    print("Unzipped\n")

    local images_dir = script_dir.."tmp_test/GTSRB/Final_Test/Images" 

        -- container for images
        local test_imgs = {}
        local m =1

        -- table of image names
        local images = paths.dir(images_dir)                              
        for i, img in ipairs(images) do
            if not img:find("^%.") and not img:find(".csv") then

                -- path to the image
                local cur_dir = paths.concat(images_dir, img)            
                local cur_img = image.load(cur_dir)
                 
                    -- crop along min_dim to have a squared image and keep the ratio as we scale it
                    local min_dim = math.min(cur_img:size(2), cur_img:size(3))
                    cur_img = image.crop(cur_img, 0, 0, min_dim, min_dim)
                
                    -- scales input image
                    cur_img = image.scale(cur_img, 32, 32)              
                
                    -- map to yuv space
                    cur_img = image.rgb2yuv(cur_img)                   
                
                    -- add image to the list of training images  
                    test_imgs[#test_imgs + 1 ] = {cur_img, tonumber(labels[m])+1}
                m = m+1
            end
        end

        -- save test set on disk
        print("\nSaving test set ...")
        torch.save(test_file, test_imgs)
        print("Saved under "..test_file)
        print("Number of test examples : "..#test_imgs)


    print("\nDo you want to remove tmp_test/ temporary dir [y/n]?")
    if(io.read() == "y") then
        os.execute("rm -r tmp_test/")
        print("tmp_test/ removed")
    else
        print("tmp_test kept")
    end
    else
        print("No test labels loaded")
    end
else
    print("\nTest set already exists : ")
    print(test_file)
end

