import glob
import time
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import os
import gc

'''
how to create above .npy file from image
from PIL import Image
img = Image.open("cat.bmp")
from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])
img_t = transform(img)
x = img_t.detach().cpu().numpy()
np.save("./n02124075_Egyptian_cat.npy", x)
'''

def main():
    with open('./synset.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    model = "./out_squeezenet1.1.onnx"
    print(model)
    cat = np.load("./n02124075_Egyptian_cat.npy")
    sess = onnxruntime.InferenceSession(model)
    model = model.split("out_")[-1].split(".")[0]
    num_of_runs=10
    # get the name of the first input of the model
    input_name = sess.get_inputs()[0].name
    results = []
    for batch_size in [1, 32, 64, 128, 256, 640]:
    tensor = np.array([cat]*batch_size)
    print("Image input dimension:", tensor.shape)
    gc.collect()
    #Warm up for 3 time
    print("Warmup 3 runs")
    for i in range(3):
        sess.run([], {input_name: tensor})
        pass
    print("Inference Running")
    total_time = 0
    for i in range(num_of_runs):
        gc.collect()
        start_time = time.time()
        sess.run([], {input_name: tensor})
        end_time = time.time()
        total_time = total_time + (end_time - start_time)
        pass
    del tensor, i
    average_time = total_time / num_of_runs
    throughput = batch_size / average_time
    print("Average time for",batch_size, "batch_size is", average_time, "s")
    print("Throughput for", batch_size, "images is", throughput,"imgs/s\n")
    results.append([batch_size, throughput, average_time])

    print("batch_size, Throughput, Average Time")
    for i in results:
        print(i)
        pass
    print('\n\n')
    del sess, throughput, average_time, num_of_runs, batch_size, total_time, i, results
    gc.collect()
     pass
if __name__ == "__main__":
    gc.collect()
    main()
    del glob, time, os, np, csv, onnxruntime
    gc.collect()
    del gc
    exit()