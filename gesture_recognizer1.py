#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

"""
IMPORTS 
"""
from skimage import data, io, filters
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import hog 
from skimage.transform import resize
from PIL import Image

import pickle
import numpy as np
import pandas as pd

import math
import glob
import random
import csv
from os import listdir
from os.path import isfile, join

from sklearn.ensemble import  RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import json
import time
import gzip

#given a list of filenames return s a dictionary of images 
def getfiles( filenames):
    dir_files = {}
    for x in filenames:
        dir_files[x]=io.imread(x)
    return dir_files

#return hog of a particular image vector
def convertToGrayToHOG(imgVector):
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)

#takes returns cropped image 
def crop(img,x1,x2,y1,y2):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((128,128)))#resize
    return crp

#save classifier
def dumpclassifier(filename,model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)    

#load classifier
def loadClassifier(picklefile):
    fd = open(picklefile, 'r+')
    model = pickle.load(fd)
    fd.close()
    return model

"""
This function randomly generates bounding boxes 
Return: hog vector of those cropped bounding boxes along with label 
Label : 1 if hand ,0 otherwise 
"""
def buildhandnothand_lis(frame,imgset):
    poslis =[]
    neglis =[]

    for nameimg in frame.image:
        # tupl: ['user_3/Y4.jpg' 138 10 258 130 120 1]
        # frame hand: 1 在之前添加到frame的时候，初始化所有的hand是1
        tupl = frame[frame['image']==nameimg].values[0]
        # print("tuple from frame", tupl)
        # x_tl: top left of x
        x_tl = tupl[1]
        # y_tl: top left of y
        y_tl = tupl[2]
        # frame side: bottom_right_x - top_left_x的值
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        
        arg1 = [x_tl,y_tl,conf,side,side]
        poslis.append(convertToGrayToHOG(crop(imgset[nameimg],x_tl,x_tl+side,y_tl,y_tl+side)))
        # 直到找到不是hand的部分，就加到neglis中然后break， 继续下一个图片
        while dic[0] <= 1 or dic[1] < 1:
            x = random.randint(0,320-side)
            y = random.randint(0,240-side) 
            crp = crop(imgset[nameimg],x,x+side,y,y+side)
            hogv = convertToGrayToHOG(crp)
            arg2 = [x,y, conf, side, side]
            
            z = overlapping_area(arg1,arg2)
            if dic[0] <= 1 and z <= 0.5:
                neglis.append(hogv)
                dic[0] += 1
            if dic[0]== 1:
                break
    # neglis 和 poslis里面保存的是not hand的hog vector和 hand的hog vector
    label_1 = [1 for i in range(0,len(poslis)) ] # assign all labels to 1 in hand list
    label_0 = [0 for i in range(0,len(neglis))] # assign all labels to 0 in not-hand list
    label_1.extend(label_0)
    poslis.extend(neglis)
    return poslis,label_1

#returns imageset and bounding box for a list of users 
def train_binary(train_list, data_directory):
    frame = pd.DataFrame()
    list_ = []
    for user in train_list:
        list_.append(pd.read_csv(data_directory+'/'+user+'/'+user+'_loc.csv',index_col=None,header=0))
    
    frame = pd.concat(list_)
    frame['side']=frame['bottom_right_x']-frame['top_left_x']
    frame['hand']=1
    # print("frame", frame)
    # frame              image          top_left_x  top_left_y  bottom_right_x  bottom_right_y  side  hand
    # 0               user_3/A0.jpg         124          18           214             108        90     1
    imageset = getfiles(frame.image.unique())

    #returns actual images and dataframe 
    return imageset,frame

#loads data for binary classification (hand/not-hand)
def load_binary_data(user_list, data_directory):
    data1,df  =train_binary(user_list, data_directory) # data 1 - actual images , df is actual bounding box
    
    # third return, i.e., z is a list of hog vecs, labels
    z = buildhandnothand_lis(df,data1)
    return data1,df,z[0],z[1]


#loads data for multiclass 
def get_data(user_list, img_dict, data_directory):
    X = []
    Y = []
    
    for user in user_list:    
        user_images = glob.glob(data_directory+user+'/*.jpg')
        boundingbox_df = pd.read_csv(data_directory+'/'+user+'/'+user+'_loc.csv')
        for rows in boundingbox_df.iterrows():
            cropped_img = crop(img_dict[rows[1]['image']], rows[1]['top_left_x'], rows[1]['bottom_right_x'], rows[1]['top_left_y'], rows[1]['bottom_right_y'])
            hogvector = convertToGrayToHOG(cropped_img)
            X.append(hogvector.tolist())
            Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y

#utility funtcion to compute area of overlap
def overlapping_area(detection_1, detection_2):
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    # detection_1[3] is side
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

"""
Does hard negative mining and returns list of hog vectos , label list and no_of_false_positives after sliding 
"""
def do_hardNegativeMining(cached_window,frame, imgset, model, step_x, step_y):   
    lis = []
    no_of_false_positives = 0
    for nameimg in frame.image:
        tupl = frame[frame['image']==nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        arg1 = [x_tl,y_tl,conf,side,side]
        for x in range(0,320-side,step_x):
            for y in range(0,240-side,step_y):
                arg2 = [x,y,conf,side,side]
                z = overlapping_area(arg1,arg2)
                
                
                prediction = model.predict([cached_window[str(nameimg)+str(x)+str(y)]])[0]

                if prediction == 1 and z<=0.5:
                    lis.append(cached_window[str(nameimg)+str(x)+str(y)])
                    no_of_false_positives += 1
    
    label = [0 for i in range(0,len(lis))]
    return lis,label, no_of_false_positives

"""
Modifying to cache image values before hand so as to not redo that again and again 

"""
def cacheSteps(imgset, frame ,step_x,step_y):
    # print "Cache-ing steps"
    list_dic_of_hogs = []
    dic = {}
    i = 0
    for img in frame.image:
        tupl = frame[frame['image']==img].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        i += 1 
        # if i%10 == 0:
        #     print "{0} images cached ".format(i)
        imaage = imgset[img]
        for x in range(0,320-side,step_x):
            for y in range(0,240-side,step_y):
                dic[str(img+str(x)+str(y))]=convertToGrayToHOG(crop(imaage,x,x+side,y,y+side))
    return dic


# classifier for hand detector
def improve_Classifier_using_HNM(hog_list, label_list, frame, imgset, threshold=50, max_iterations=25): # frame - bounding boxes-df; yn_df - yes_or_no df
    print ("Performing HNM ...")
    no_of_false_positives = 1000000     # Initialise to some random high value
    i = 0

    step_x = 32
    step_y = 24

    mnb  = MultinomialNB()
    cached_wind = cacheSteps(imgset, frame, step_x, step_y)

    while True:
        i += 1
        model = mnb.partial_fit(hog_list, label_list, classes = [0,1])

        ret = do_hardNegativeMining(cached_wind,frame, imgset, model, step_x=step_x, step_y=step_y)
        
        hog_list = ret[0]
        label_list = ret[1]
        no_of_false_positives = ret[2]
        
        if no_of_false_positives == 0:
            return model
        
        print ("Iteration {0} - No_of_false_positives: {1}".format(i, no_of_false_positives))
        
        if no_of_false_positives <= threshold:
            return model
        
        if i>max_iterations:
             return model

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # print "Perfmorinf NMS:"
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

# Returns the tuple with the highest prediction probability of hand
def image_pyramid_step(model, img, scale=1.0):
    max_confidence_seen = -1
    rescaled_img = rescale(img, scale)
    detected_box = []
    side = 128
    x_border = rescaled_img.shape[1]
    y_border = rescaled_img.shape[0]
 
    for x in range(0,x_border-side,32):
        for y in range(0,y_border-side,24):
            cropped_img = crop(rescaled_img,x,x+side,y,y+side)
            hogvector = convertToGrayToHOG(cropped_img)

            confidence = model.predict_proba([hogvector])

            if confidence[0][1] > max_confidence_seen:
                detected_box = [x, y, confidence[0][1], scale]
                max_confidence_seen = confidence[0][1]

    return detected_box

"""
=================================================================================================================================
"""

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return (self.output_layer.feed_forward(hidden_layer_outputs))

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
        print('calculated output error')
        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()
            # pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_softmax_drivative()
            # pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_rectifier_derivative()
        print('calculated hidden error')
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
        print('updated output weights')
        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight
        print('updated hidden weights')
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            print('Sequence: ', t)
            output = self.feed_forward(training_inputs)
            print('  Output: ', output)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

    def calculate_misclassification_error(self, training_inputs, training_outputs):
        # print ("Inside calculate_misclassification_error")
        total_error = 0
        misclassified = 0
        error_val = 0
        self.feed_forward(training_inputs)
        min_error = 10000
        min_error_index = -1
        for o in range(len(training_outputs)):
            error_val = self.output_layer.neurons[o].calculate_error(training_outputs[o])
            print ("Error for ",o,error_val, end=', ')
            total_error += error_val
            if(error_val < min_error):
                min_error_index = o
                min_error = error_val
        if(training_outputs[min_error_index] == 1):
            misclassified = 0
        else:
            misclassified = 1
        print ("min_error_index", min_error_index)
        return misclassified,total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        tempinput = self.calculate_total_net_input()
        self.output = self.squash(tempinput)
        # self.output = self.softmax(tempinput)
        # self.output = self.rectify(tempinput)
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        sig = (1 / (1 + np.exp(-total_net_input)))
        # sigmoid always has result of 1.0 no matter float or int
        return sig
        # 1 / (1 + np.exp(-x))

    def softmax(self, total_net_input):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(total_net_input - np.max(total_net_input))
        result = e_x / e_x.sum()
        return result

    def rectify(self, total_net_input):
        return max(0, total_net_input)

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        # return self.calculate_pd_error_wrt_output(target_output) * self.calculate_rectifier_derivative()
        # return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_softmax_drivative()
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)


    # def calculate_pd_total_softmax_drivative(self):
    #     print('self output', self.output)
    #     jacobian_m = np.diag(self.output)
    #     for i in range(len(jacobian_m)):
    #         for j in range(len(jacobian_m)):
    #             if i == j:
    #                 jacobian_m[i][j] = self.output[i] * (1-self.output[i])
    #             else: 
    #                 jacobian_m[i][j] = -self.output[i]* self.output[j]
    #     return jacobian_m

    def calculate_rectifier_derivative(self):
        if self.output > 0:
            return 1
        else:
            return 0

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


# class BPNN(object):
#     def __init__(self, X_mul, Y_mul, hog_list, label_list, boundbox, imageset, data_director='./',learning_rate = 0.01, n_iter = 10000, momentum = 0.9, shutdown_condition = 0.01):
#         self.data_directory = data_director
#         self.hog_list = hog_list
#         self.label_list = label_list
#         self.boundbox = boundbox
#         self.imageset = imageset
#         self.n_iter = n_iter
#         self.learning_rate = learning_rate
#         self.shutdown_condition = shutdown_condition
#         self.cost = []
#         self.momentum = momentum
#         self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
#           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
#         self.x = np.array(X_mul)
#         self.Y = np.array(self.label_encoder.fit_transform(Y_mul))
#         self.handDetector = None
#         self.setup()

#     def setup(self):
#         self.set_nn_architecture()
#         self.set_weight()
#     # step 1 input_node is input units, output_node is output units, hidden_node is hidden units
#     def set_nn_architecture(self):
#         # 15876 input units, 24 output units（同一个类输出10次）
#         self.input_node = self.x.shape[1]
#         self.output_node = self.Y.shape[0]/10
#         self.hidden_node = math.sqrt(self.input_node + self.output_node)+5
#         # bias
#         self.h_b = np.random.random(self.hidden_node) * 0.3 + 0.1
#         self.y_b = np.random.random(self.output_node) * 0.3 + 0.1

#     # step 2 only one hidden layer, 2 different weights
#     def set_weight(self):
#         for j in range(self.hidden_node):
#             for i in range(self.input_node):
#                 self.w1[i] = np.random.random((self.output_node , self.hidden_node))
#         for m in self.hidden_node:
#             for n in self.output_node:
#                 self.w2[m][n] = np.random.random((self.output_node , self.hidden_node))
#         # self.w1 = np.random.random((self.input_node , self.hidden_node))
#         # self.w2 = np.random.random((self.hidden_node , self.output_node))

#     # step 3 forward feeding 
#     def predict(self , x , Y):
#         print('BPNN feed forward...')
#         # sigmoid of (output from input to hidden layer plus bias)
#         self.h = self.sigmoid((np.dot(x , self.w1) + self.h_b))
#         # sigmoid of (ouput from hidden to output layer plus bias)
#         self.y = self.sigmoid((np.dot(self.h , self.w2) + self.y_b))
#         zy = np.where(self.y > 0.5 , 1 , 0)
#         p_y = Y - zy
#         self.acc = 0
#         for i in p_y:
#             if (i.sum() == 0):
#                 self.acc += 1
#         self.acc = self.acc / Y.shape[0] * 100.0
#         return self
#     # step 4 back propagation update the weights
#     def backend(self):
#         print('BPNN updating...')
#         E = (self.Y - self.y)
#         errors = np.sum(np.square(E)/2)
#         # errors = np.sum(np.square(E)) / self.Y.shape[1] / self.Y.shape[0]
#         #### 輸出層 delta 計算
#         delta_y = E * self.y * (1 - self.y)
#         ### 隱藏層 delta 計算
#         delta_h = (1 - self.h) * self.h * np.dot(delta_y , self.w2.T)
#         # self.w2 += self.learning_rate * self.h.T.dot(delta_y) + self.momentum * self.h.T.dot(delta_y)
#         # self.w1 += self.learning_rate * self.x.T.dot(delta_h) + self.momentum * self.x.T.dot(delta_h)
#         self.w2 += self.learning_rate * self.h.T.dot(delta_y)
#         self.w1 += self.learning_rate * self.x.T.dot(delta_h)
#         self.y_b = self.learning_rate * delta_y.sum()
#         self.h_b = self.learning_rate * delta_h.sum()
#         return errors

#     def predictOutput(self, x):
#         print('BPNN start predicting output')
#         # sigmoid of (output from input to hidden layer plus bias)
#         # self.h = self.sigmoid((np.dot(x , self.w1) + self.h_b))
#         # sigmoid of (ouput from hidden to output layer plus bias)
#         # self.y = self.sigmoid((np.dot(self.h , self.w2) + self.y_b))
#         # zy = np.where(self.y > 0.5 , 1 , 0)
#         self.y = max(0, (np.dot(self.h , self.w2) + self.y_b))
#         print('-----y: ', self.y)
#         return self

#     def train(self, train_list):
#         print('BPNN start training')
#         if self.handDetector == None:
#             # Build binary classifier for hand-nothand classification
#             self.handDetector = improve_Classifier_using_HNM(self.hog_list, self.label_list, self.boundbox, self.imageset, threshold=40, max_iterations=35)

#         self.error = 0
#         for _iter in range(0 , self.n_iter):
#             self.predict(self.x, self.Y)
#             self.error = self.backend()
#             self.cost.append(self.error)
#             # if (_iter % 1000 == 0):
#             #     print("Accuracy：%.2f" % self.acc)
#             if (self.acc >= 98):
#                 return self
#         return self

#     def test(self, image):
#         scales = [   1.25,
#                  1.015625,
#                  0.78125,
#                  0.546875,
#                  1.5625,
#                  1.328125,
#                  1.09375,
#                  0.859375,
#                  0.625,
#                  1.40625,
#                  1.171875,
#                  0.9375,
#                  0.703125,
#                  1.71875,
#                  1.484375
#             ]

#         detectedBoxes = [] ## [x,y,conf,scale]
#         for sc in scales:
#             detectedBoxes.append(image_pyramid_step(self.handDetector,image,scale=sc))
#         # self.handDetector
#         side = [0 for i in range(len(scales))]
#         for i in range(len(scales)):
#             side[i]= 128/scales[i]

#         for i in range(len(detectedBoxes)):
#             detectedBoxes[i][0]=detectedBoxes[i][0]/scales[i] #x
#             detectedBoxes[i][1]=detectedBoxes[i][1]/scales[i] #y

#         nms_lis = [] #[x1,x2,y1,y2]
#         for i in range(len(detectedBoxes)):
#             nms_lis.append([detectedBoxes[i][0],detectedBoxes[i][1],
#                             detectedBoxes[i][0]+side[i],detectedBoxes[i][1]+side[i],detectedBoxes[i][2]])
#         nms_lis = np.array(nms_lis)

#         res = non_max_suppression_fast(nms_lis,0.4)

#         output_det = res[0]
#         x_top = output_det[0]
#         y_top = output_det[1]
#         side = output_det[2]-output_det[0]
#         position = [x_top, y_top, x_top+side, y_top+side]
        
#         croppedImage = crop(image, x_top, x_top+side, y_top, y_top+side)
#         hogvec = convertToGrayToHOG(croppedImage)
#         self.predictOutput(hogvec)
#         return self

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))


class GestureRecognizer(object):
    """class to perform gesture recognition"""

    def __init__(self, data_director='./'):
        """
            data_directory : path like /home/sanket/mlproj/dataset/    
            includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.data_directory = data_director
        self.handDetector = None
        self.signDetector = None
        self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

    def __init2__(self,data_dir, hand_Detector, sign_Detector):
        self.data_directory = data_dir
        self.handDetector = loadClassifier(hand_Detector)
        self.signDetector = loadClassifier(sign_Detector)
        self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])


    def train(self, train_list):
        """
            train_list : list of users to use for training
            eg ["user_1", "user_2", "user_3"]
            The train function should train all your classifiers
            both binary and multiclass on the given list of users
        """
        print ("Train starts")
        # Load data for the binary (hand/not hand) classification task
        imageset, boundbox, hog_list, label_list = load_binary_data(train_list, self.data_directory)

        print ("Imageset, boundbox, hog_list,label_list Loaded!")

        # Load data for the multiclass classification task
        X_mul,Y_mul = get_data(train_list, imageset, self.data_directory)
        # print("-----------length required-------")
        # print(len(X_mul[0]), len(X_mul[1]))
        # print ("Multiclass data loaded")
       
        # encode the output value into numerical value
        Y_mul = self.label_encoder.fit_transform(Y_mul)

        if self.handDetector == None:
            # Build binary classifier for hand-nothand classification
            self.handDetector = improve_Classifier_using_HNM(hog_list, label_list, boundbox, imageset, threshold=40, max_iterations=35)

        print ("handDetector trained ")

        # Multiclass classification part to classify the various signs/hand gestures CHECK. TODO.
        
        if self.signDetector == None:
            svcmodel = SVC(kernel='linear', C=0.9, probability=True)
            self.signDetector = svcmodel.fit(X_mul, Y_mul)


        print ("sign Detector trained ")

        # dumpclassifier('handDetector.pkl', self.handDetector)
        
        # dumpclassifier('signDetector.pkl', self.signDetector)

        # dumpclassifier('label_encoder.pkl', self.label_encoder)


    def recognize_gesture(self, image):
        """
            image : a 320x240 pixel RGB image in the form of a numpy array

            This function should locate the hand and classify the gesture.
            returns : (position, label)

            position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
                x1,y1 is top left corner, x2,y2 is bottom right

            label : a single character. eg 'A' or 'B'
        """
        # print "In recognize_gesture"
        scales = [   1.25,
                 1.015625,
                 0.78125,
                 0.546875,
                 1.5625,
                 1.328125,
                 1.09375,
                 0.859375,
                 0.625,
                 1.40625,
                 1.171875,
                 0.9375,
                 0.703125,
                 1.71875,
                 1.484375
            ]

        detectedBoxes = [] ## [x,y,conf,scale]
        for sc in scales:
            detectedBoxes.append(image_pyramid_step(self.handDetector,image,scale=sc))
        # self.handDetector
        side = [0 for i in range(len(scales))]
        for i in range(len(scales)):
            side[i]= 128/scales[i]

        for i in range(len(detectedBoxes)):
            detectedBoxes[i][0]=detectedBoxes[i][0]/scales[i] #x
            detectedBoxes[i][1]=detectedBoxes[i][1]/scales[i] #y

        nms_lis = [] #[x1,x2,y1,y2]
        for i in range(len(detectedBoxes)):
            nms_lis.append([detectedBoxes[i][0],detectedBoxes[i][1],
                            detectedBoxes[i][0]+side[i],detectedBoxes[i][1]+side[i],detectedBoxes[i][2]])
        nms_lis = np.array(nms_lis)

        res = non_max_suppression_fast(nms_lis,0.4)

        output_det = res[0]
        x_top = output_det[0]
        y_top = output_det[1]
        side = output_det[2]-output_det[0]
        position = [x_top, y_top, x_top+side, y_top+side]
        
        croppedImage = crop(image, x_top, x_top+side, y_top, y_top+side)
        hogvec = convertToGrayToHOG(croppedImage)

        prediction = self.signDetector.predict_proba([hogvec])[0]

        zi = zip(self.signDetector.classes_, prediction)
        zilist = sorted(list(zi), key = lambda x:x[1],reverse = True)
        print("zi list: ", zilist)

        # To return the top 5 predictions
        final_prediction = []
        for i in range(5):
            ziarray = [zilist[i][0]]
            final_prediction.append(self.label_encoder.inverse_transform(ziarray))
        # print(position,final_prediction)

        return position,final_prediction
        
    def save_model(self, **params):

        """
            save your GestureRecognizer to disk.
        """

        self.version = params['version']
        self.author = params['author']

        file_name = params['name']

        pickle.dump(self, gzip.open(file_name, 'wb'))
        # We are using gzip to compress the file
        # If you feel compression is not needed, kindly take lite

    @staticmethod       # similar to static method in Java
    def load_model(**params):
        """
            Returns a saved instance of GestureRecognizer.

            load your trained GestureRecognizer from disk with provided params
            Read - http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-parameters
        """

        file_name = params['name']
        return pickle.load(gzip.open(file_name, 'rb'))

        # People using deep learning need to reinitalize model, load weights here etc.

def main():
    data_directory = '/Users/VickyYang/python3/GestureRecognition/datasetCode'
    userlist=['user_3','user_4','user_5','user_6','user_7','user_9','user_10']
    # train 2 user_* files 
    user_tr = userlist[:1]
    # user_te = userlist[-1:-7]
    imageset, boundbox, hog_list, label_list = load_binary_data(user_tr, data_directory)
    X_mul,Y_mul = get_data(user_tr, imageset, data_directory)
    Y_mul = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']).fit_transform(Y_mul)
    
    # -------------------------------back propogation-------------------------
    # Y_train = []
    # for y in Y_mul:
    #     y_value = [0]*24
    #     y_value[y] = 1
    #     Y_train.append(y_value)
    # trainingSets = list([[0]*len(X_mul[0]),[0]*24]*len(X_mul))
    # index = 0
    # for x in X_mul:
    #     trainingSets[index] = ([x, Y_train[index]])
    #     index = index+1
    # nn = NeuralNetwork(len(X_mul[0]),int(math.sqrt(len(X_mul[0]) + 1)+10), 24)
    # print("input neurons: ", len(X_mul[0]), ", hidden neurons: ", int(math.sqrt(len(X_mul[0]) + 1)+10), ", output neurons: ", 24)

    # training_inputs, training_outputs = trainingSets[1]
    # # print("training input: ", training_inputs, ", outputs: ", training_outputs)
    # nn.train(training_inputs, training_outputs)
    # training_intputs1, training_outputs1 = trainingSets[2]
    # nn.train(training_intputs1, training_outputs1)
    # image_num = 0
    # for training_inputs2, training_outputs2 in trainingSets:
    #     print('image no.', image_num, end=', ')
    #     nn.calculate_misclassification_error(training_inputs2, training_outputs2)
    #     image_num = image_num+1

    # for user_test in userlist:
    #     imageset, boundbox, hog_list, label_list = load_binary_data([user_test], data_directory)
    #     X_test,Y_test_mul = get_data([user_test], imageset, data_directory)
    #     Y_test_mul = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
    #     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']).fit_transform(Y_test_mul)
    #     Y_test = []
    #     for y in Y_test_mul:
    #         y_value = [-1]*24
    #         y_value[y] = y
    #         Y_test.append(y_value)
    #     trainingSets = list([[0]*len(X_test[0]),[-1]*24]*len(X_test))
    #     ytest_index = 0
    #     for x in X_test:
    #         trainingSets[ytest_index] = ([x, [Y_test[ytest_index]]])
    #         ytest_index = ytest_index+1
    #     nn.calculate_total_error(trainingSets)
    #     trainingSets.clear()
    #     imageset.clear()
    #     hog_list.clear()
    #     label_list.clear
    #     del boundbox
    #     X_test.clear()
    #     Y_test.clear()



    # ------------------------SVM-----------------------------------------

    #handDetector = loadClassifier('./handDetector.pkl')
    #signDetector = loadClassifier('./signDetector.pkl')
    # self,data_dir,hand_Detector,sign_Detector
    gs = GestureRecognizer(data_directory) 
    # , 'model_log_2.pkl', 'my_sign_detector.pkl'
    
    # need to train everytime
    gs.train(user_tr)
    # using two user_* files build a model classifier
    gs.save_model(name = "my_sign_detector.pkl.gz", version = "0.0.1", author = 'ss')

    # print ("The GestureRecognizer is saved to disk")
    
    # gs = GestureRecognizer.load_model(name = "my_sign_detector.pkl.gz") # automatic dict unpacking
    # print (new_gr.label_encoder)
    # print (new_gr.signDetector )

    # real testing
    # for userfolder in userlist:
    #     print('/Users/VickyYang/python3/GestureRecognition/datasetCode/'+userfolder)
    #     for filename in glob.glob('/Users/VickyYang/python3/GestureRecognition/datasetCode/'+userfolder+'/*.jpg'):
    #         im=Image.open(filename).convert('RGB')
    #         image_array = np.array(im)
    #         # gs.recognize_gesture(image_array)
    #         bpnn.test(image_array)

    # part testing
    # for filename in glob.glob('/Users/VickyYang/python3/GestureRecognition/datasetCode/user_4'+'/*.jpg'):
    #     im=Image.open(filename).convert('RGB')
    #     npim = np.array(im)
        # bpnn.test(npim)

    # filepath = ""
    # imagepath = ""
    # misclassified = {"user_3":0,"user_4":0,"user_5":0,"user_6":0,"user_7":0,"user_9":0,"user_10":0}
    # file = open("output.txt",'w')
    # for user in userlist:
    #     filepath = '/Users/VickyYang/python3/GestureRecognition/datasetCode/'+user+'/'
    #     onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    #     #print "Entering for",user
    #     for img in onlyfiles:
    #         predicted_correct = 0
    #         temp_img = img
    #         imagepath = filepath+img
    #         if ".jpg" in imagepath:
    #             #print imagepath
    #             img_h = Image.open(imagepath).convert('RGB')
    #             data = np.array(img_h)
    #             position,final_prediction = gs.recognize_gesture(data)
    #             for val1 in final_prediction:
    #                 class_label = val1[0]
    #                 class_label = class_label[0]
    #                 if class_label == temp_img[0]:
    #                     predicted_correct = 1
    #                     break
    #             if predicted_correct == 0:
    #                 misclassified[user] += 1
    # print ('misclassified', misclassified)
    
if __name__ == '__main__':
    main()