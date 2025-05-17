Enhancing Pesticide-likeness Prediction via Multimodal Deep Learning Ensemble Framework


#Environment
python:3.13.2
rdkit:2023.3.2
torch:2.7.0+cu126

#Data
We classify molecules into two categories: pesticides and non-pesticides, with the pesticide category further divided into three types: fungicides, insecticides, and herbicides.
Pesticides dataset: From BCPC # data/pesticides.txt 
Non-pesticide dataset: Molecules from the ZINC database. We construct the negative sample set using molecular fingerprint similarity and clustering algorithm, ensuring broader coverage and more comprehensive information. #data/neg.txt
Additionally, we collected molecules with high fungicidal activity (with no overlap with the above dataset). But these molecules have not undergone safety, stability, or other regulatory testing,so they have not been practically used as fungicides. These molecules were included in the test set for experimental validation.   #data/pesticidal molecules.txt

#Usage
If you want to use our model RGG to predict your dataset, you can change the #data/work.txt file and run the #run.py file. the fun_classifier function can be used to classify fungicides and non-fungicides, ins_classifier predicts insecticide types, her_classifier predicts herbicide types, and pes_classifier predicts pesticide categories.

If you want to retrain the model, you can the #train/train.py and put your data into data directory. In the train.py file, you can change the hyperparameters of the the neural network.

The #models directory saves the detail structure of the RGG model. #model/para saves the train result(model's parameters).

For more details, you can read on our thesis.
