import json
import time
import sys
import os
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def init():
    global session
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'resnet50v2.onnx')
    session = onnxruntime.InferenceSession(model, None)

def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    img_data = np.array(json.loads(input_data_json)['data']).astype('float32')
    
    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    return norm_img_data

def postprocess(result):
    return softmax(np.array(result)).tolist()

def run(input_data_json):
    try:
        start = time.time()
        # load in our data which is expected as NCHW 224x224 image
        input_data = preprocess(input_data_json)
        input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
        result = session.run([], {input_name: input_data})
        end = time.time()     # stop timer
        return {"result": postprocess(result),
                "time": end - start}
    except Exception as e:
        result = str(e)
        return {"error": result}
