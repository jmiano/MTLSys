# MTLSys: Latency-aware Pruning for Multi-task Learning

## To run demo:

#### Install dependencies

* `pip install -r requirements.txt`
* [Install Jupiter Notebook](https://jupyter.org/install)


#### Download trained model-variants

`gdown "https://drive.google.com/uc?id=1pG_6ncWFn8Gy4pIaz4Q4EE6JbGmD6RkM"e`
`unzip model-variants.zip`
`cp -r model-variants models/`

#### Run the demo notebook

To run the demo of the inference system on small sample dataset - 

* Open the notebook - `inference/demo.ipynb`
* Run all cells
* Wait for a few minutes as profiling takes time.

## To Download dataset - 

`gdown "https://drive.google.com/uc?id=18YAbwahQT808HjJ0ZthqX6oKNkYZd-Yf"`

unzip all files.


## Build and Run Code

#### Install dependencies

* `pip install -r requirements.txt`
* [Install Jupiter Notebook](https://jupyter.org/install)


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
