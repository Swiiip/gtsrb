# Create datasets

First, you need to create training and test datasets, run `th createDataSets.lua`.

Requires:
* csvigo
* torch7
* wget (UNIX standard command tool)
* zip/unzip (UNIX standard command tool)

This will :
* Download training dataset
* Download test dataset
* Download test set labels
* Unzip training dataset and save it to a torch7 compliant format
* Unzip test dataset and labels, join them and save them to a torch7 compliant format
* Erase temporary created directories on demand

Run `th createDataSets.lua -help` to see all available options

# Load the interactive environment

 This script loads the main global variables into Luajit :
 * **train_set**          : loaded by dataset.lua
 * **test_set**           : loaded by dataset.lua
 * **model**              : loaded by models/MSmodel.lua
 * **learning_rate**      : loaded by train.lua
 * **batch_size**         : loaded by train.lua

This programming architecture is modular, you can use your own preprocessing/train/test functions as well as your models, as long as they respect the model/dataset interface described in the corresponding files (dataset.lua, models/MSmodel.lua, ...)

Just run `th -i main.lua` to load the elements from the different modules and start interactively changing the model parameters, loading an aldready trained model, tweaking the parameters (learning_rate, batch_size, ...) and using the train() and test() functions.

The first time you run `th -i main.lua`, the data sets will be preprocessed using the code in *preprocessing.lua*. You will be asked if you want to save the preprocessed data. Once saved, you can skip this step by using `th -i main.lua -use_pp_sets`.

If you want to use a different model for instance, just use `th -i main.lua -model "path/to/the/model.lua"`.
If you want to load an already trained model, use `model = torch.load("path/to/model.t7")`in Luajit.

Run `th main.lua -help` to see all available options.

# GTSRB Challenge

a.k.a German Traffic Sign Recognition Benchmark :de: :no_entry: :no_bicycles:
:no_entry_sign: ...

## Goal

Use [Torch](http://torch.ch/) to train and evaluate a 2-stage convolutional
neural network able to classify German traffic sign images (43 classes):

* fork the repository under your account,
* go to Settings > Features and enable Issues,
* create an issue under your repo describing your approach,
* report your result(s),
* commit your code,
* edit the README with pre-requisites and usage,
* boost accuracy by experimenting the multi-scale architecture,
* compare with the results obtained in matching mode (i.e use the features with a distance-based search).

## Paper

[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/reference_papers/2-sermanet-ijcnn-11-mscnn.pdf), by Yann LeCun et al.

## Dataset

### Training

`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip` (263 MB)

### Testing

`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip` (84 MB)
`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip` (98 kB)
