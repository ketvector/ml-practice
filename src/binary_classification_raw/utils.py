import numpy as np
import math

def calculateBasic(inputs, weights):
    return np.matmul(inputs,weights)

def calculateActivation(basic):
    def myActivation(x):
        e_x = math.exp(x[0]);
        return np.array([e_x / (e_x + 1)]) 
    return np.apply_along_axis(myActivation,axis=1,arr=basic)

def predict(activations):
    predictions = np.zeros(shape=(activations.shape))
    for index in range(activations.shape[0]):
        predictions[index] = np.array([1 if activations[index][0] > 0.5 else 0])
    return predictions


def loss(actual_y, predicted_a):
    curr = 0
    def single_item_loss(actual,predicted):
        if actual[0] == 1:
            return actual[0] - predicted[0]
        else:
            return predicted[0] - actual[0]
    
    for index in range(actual_y.shape[0]):
        curr = curr + single_item_loss(actual_y[index], predicted_a[index])

    return curr

def accuracy(actual_y, predicted_y):
    correct = 0
    total = actual_y.shape[0]
    for index in range(actual_y.shape[0]):
        now = 1 if actual_y[index][0] == predicted_y[index][0] else 0
        correct = correct + now
    return correct/total

def get_weight_updates(actual_y, input_matrix, activations, basic):
    updates = np.zeros(shape=(input_matrix.shape[1],1))
    for cIndex in range(input_matrix.shape[1]):
        updateAtIndex = 0
        for rIndex in range(input_matrix.shape[0]):
            add = (input_matrix[rIndex][cIndex] * activations[rIndex])/(1.0 + math.exp(basic[rIndex]))
            if actual_y[rIndex] == 0:
                add = -1 * add
            updateAtIndex = updateAtIndex + add
        updateAtIndex = updateAtIndex / input_matrix.shape[0]
        updates[cIndex] = np.array([updateAtIndex])
    return updates
    

