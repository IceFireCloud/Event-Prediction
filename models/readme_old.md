Event Prediction 2.0
Built by Kieran, Xiao, Gavin.

-------------------------------------------
To Run:

1) Activate Python environment:
    - Use "source /home/xzhou/tensorflow/bin/activate"

2) Ensure hard-coded directories are correct.
    - The directories are in the file "run_RNN_original.py" (or CNN)
    - The programs use data and results directories that may
        need to be adjusted. 
    - Additional changes may need to be made depending
        on if you want to use all cases or loop over datasets from 0504 and 0508.
        The code is hopefully clear about how that works.

3) Assign a GPU:
    - Use "nvidia-smi" to check for running processes.
    - Pick an open one and use "export CUDA_VISIBLE_DEVICES=0"
        or the number you want.
  
4) Navigate to the directory you want, for instance /united_models/RNN.

5) Run file, for instance "python3 run_RNN_original.py".

6) If you used the CNN, you will need to run a separate program to plot the data.
    - Navigate to /united_models/Plotting_Tool
    - Check "Plot_all_files3.m" for the hard-coded directories
    - Execute "matlab < Plot_all_files3.m" and check the results.  


-------------------------------------------
To Adjust Code:

*Note: CNN currently slightly different, still being worked on*
    CNN class to build/run model is in Case.py
    
Shared functions are in utils.py. These functions are:
  load_data()
    Loads a single dataset, does not process it.
  run_model()
    Given a model, data, and a parameter set, trains the model.
  predict_and_save()
    Given a trained model and testing data, make predictions and save them out.

Functions specific to one's model are built in (for instance) RNN_utils.py:
  build_model()
  process()
    Given data, return it processed (including test/train split).

For fitting and evaluating the models, execute run_RNN_original.py
  Trains model on dataset, tests it on dataset.
For the combined model, execute run_RNN_combined.py
  Trains model on datasets of same events, then retrains on individual dataset before testing on that dataset.

*case number indexes the location of the train/test split.
