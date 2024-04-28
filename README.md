The file CreationIA creat an AI with a given dataset at the given path, example of use: python CreationIA --dataset_path ../Data/MyData --save_path ../Model/ModelTest
With MyData a directory with all the dataset, which as for each word a directory. And with ModelTest a directory which not exist and will be create.

The file LoadDataset wil download the dataset: https://huggingface.co/datasets/ngdiana/uaspeech_severity_low at given directory, example of use: python LoadDataset --save_path ../Data/uaSpeech
With uaSpeech a empty directory.

The file TestIA test a model of AI with your mike, every prediction you will have two prediction the first: Module prediction, is from the library speech_recognition and the scond: Our prediction, is from the given model.
The result notfound is that the model have an error when he try to predict the given audio. The program will stop when one the two prediction predict 'off'.
Example of use: python TestIA --model_path ../Model/ModelTest  with ModelTest a directory create with the CreationIA file.
