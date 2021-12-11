# MTLSys: Latency-aware Pruning for Multi-task Learning

## Run examples/demos
#### Install dependencies

* Run the following command to set up the conda environment: 
	* Windows: `conda env create --name smr_env --file smr_env.yml`
	* Linux: `conda env create --name smr_env --file smr_env_linux.yml`
	* Linux: Install PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
* Activate the conda environment by running: `conda activate smr_env`
* Open Jupyter notebooks by running: `jupyter notebook`
* Once you open the ipynb file of interest, open the "Kernel" menu, then "Change Kernel" and select the smr_env kernel

#### Download trained model-variants
* From `repo_team14`, change directory to the `examples/models` directory: `cd examples/models`
* Run the command: `gdown "https://drive.google.com/uc?id=1pG_6ncWFn8Gy4pIaz4Q4EE6JbGmD6RkM"`
* Unzip the `model-variants.zip` file. The model files should populate the `examples/models/model_variants` directory.
* If prompted to replace existing files, yes allow the files to be replaced

#### Model Variant Generation
To run the example of the MTL model variant generation on a small sample dataset - 
* Open the notebook - `examples/Generate_Model_Variants_MTL_Example.ipynb`
* Run all cells
* The model variant files will be created and visible in the `examples/models/model_variants` directory.

#### Inference
To run the example of the inference system on a small sample dataset - 
* Open the notebook - `examples/Inference_System_Demo.ipynb`
* Run all cells
* Wait for a few minutes as profiling takes time.


## Run full system
## To Download dataset - 

* Run the command: `gdown "https://drive.google.com/uc?id=18YAbwahQT808HjJ0ZthqX6oKNkYZd-Yf"`
* unzip all files.


## Build and Run Code

#### Install dependencies

* `pip install -r requirements.txt`
* [Install Jupyter Notebook](https://jupyter.org/install)


#### Variant Generator

Run `Generate_Model_Variants_MTL.ipynb` to generate all the models-variants for the MTL version of the system.
Run `Generate_Model_Variants_SingleTask.ipynb` to generate all the model-variants for the single-task version (with no MTL models) of the system.

#### Inference System

* Run `inference/Inference.ipynb` to simulate the different systems and run the queries on the system and generate the results in the report.
* `RequestDetails` can be used to specify requests to the system
* By default, the system runs on CPU hardware. To run it on GPU, replace the followin in `inference/Inference.ipynb` and `inference/utils.py`. - 
	* Replace instances of `model = torch.load(model_path, map_location=torch.device('cpu'))` with `model = torch.load(model_to_use.file_path)` 
	* Replace `model.cpu()` with `model.cuda()`
	* Replace `request_details.input_image` with `request_details.input_image.cuda()`
	* Replace `output = model(image)` with `output = model(image.cuda())`

## View Experimental Results
* `TaskHead_Length_Experiments.ipynb` shows our code to test the effect of task head length on latency
* `Pruning_Robustness_Experiments.ipynb` shows code testing relationship between pruning amount and accuracy
* `inference/Inference.ipynb` shows code testing the system inference performance, including Pareto curves and mishit plots
