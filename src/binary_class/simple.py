import numpy as np
import utils
import matplotlib.pyplot as plt


raw_matrix = np.loadtxt('sonar.txt', delimiter=",", dtype="str")
num_rows = raw_matrix.shape[0]
num_cols = raw_matrix.shape[1]

input_matrix = np.zeros(shape=(num_rows,num_cols))
output_matrix = np.zeros(shape=(num_rows,1))

for index in range(num_rows):
    input_row = np.append(raw_matrix[index][:-1], np.array([1.0]))
    input_matrix[index] = input_row
    output = 1 if raw_matrix[index][-1] == 'M' else 0 
    output_matrix[index] = np.array([output])

weights = np.random.uniform(-0.5,0.5,(num_cols,1))

num_iter = 500
check_step = 100
accuracy_arr = np.zeros(shape=(int(num_iter/check_step + 1)))

alpha = 0.1

for iter in range(num_iter+1):
    basic = utils.calculateBasic(input_matrix, weights) 

    activations = utils.calculateActivation(basic)

    predictions = utils.predict(activations)

    loss = utils.loss(output_matrix, activations)
    if iter % check_step == 0:
        print("Loss")
        print(loss)
        
    
    accuracy = utils.accuracy(output_matrix,predictions)
    if iter % check_step == 0: 
        print("Accuracy")
        print(accuracy)
        accuracy_arr[int(iter/check_step)] = accuracy

    weight_update = utils.get_weight_updates(output_matrix, input_matrix, activations, basic)

    #print(weight_update)

    weights = np.add(weight_update * alpha, weights)

fix, ax = plt.subplots()
ax.plot(np.linspace(0,num_iter,int(num_iter/check_step + 1)), accuracy_arr)
plt.show()




