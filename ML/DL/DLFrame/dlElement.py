#####################################################################
##############  created by ZJZ   ####################################
#####################################################################

"""
Containing almost all of the deep neural network element here
"""

import numpy as np
import matplotlib.pyplot as plt

'''
some useful function
'''
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))

def tanh(x):
    return np.tanh(x) 

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis = 0)])  # ndim = 2
    
def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)



##########################################
#############test code###################
#print sigmoid(np.array([2,3,4]))
#print relu(np.array([2,-1]))
#########################################






def initialize_parameters(layer_dims, initialize_type):
    """
    Arguments:
    layer_dims -- python array containing the information of network structure
    
    Returns:
    parameters 
    """
    
    parameters = {}
    L = len(layer_dims)

    if(initialize_type == "random"):
    
        for l in range(1,L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l] , 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))   
        
    elif(initialize_type =="he"):

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]* np.sqrt(2./layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layer_dims[l] , 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))   

    return parameters

#####################################################
##################debug code #########################
#parameters = initialize_parameters([3,5,2])
#print parameters
####################################################



def linear_forward(A, W, b):
    """
    compute W*A
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    compute the activation of the forward propagation
    
    Arguments:
    A_prev --activations from previous layer
    W --weight matrix
    b -- bias
    
    Returns:
    A -- activation 
    cache -- dictionary containing W, b, Z of each layer
    
    example :
    A_prev =np.array([[2],[2],[3]])
    W = np.array([[1,2,3],[2,3,4]])
    b = 2.0
    print linear_activation_forward(A_prev, W, b, "sigmoid")
    
    return: (array([[ 0.99999996],
       [ 1.        ]]), ((array([[2],
       [2],
       [3]]), array([[1, 2, 3],
       [2, 3, 4]]), 2.0), array([[ 17.],
       [ 24.]])))

    """
    if activation =="sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
        activation_cache = (Z)
        
    elif activation =="relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = relu(Z)
        activation_cache = (Z)

    elif activation =="softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = softmax(Z)
        activation_cache = (Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache

##########################################
#############test code###################
#A_prev =np.array([[2],[2],[3]])
#W = np.array([[1,2,3],[2,3,4]])
#b = 2.0
#print linear_activation_forward(A_prev, W, b, "sigmoid")
##########################################

def L_layer_forward(X, parameters, construction):
    """
    implement forward propagation from start to the end
    
    Arguments:
    X -- input vector
    parameters -- neural network layout(numpy array)
    activation -- activation function of each layer 
                  "relu&sigmoid": relu*(L-1) and sigmoid*1
    
    Returns:
    A -- last activation value(output)
    caches -- dictionary containing W, b, Z of each layer
    
    Example:
    parameters = initialize_parameters([3,5,2])
    X = np.array([[2],[3],[4]])
    AL,caches = L_layer_forward(X, parameters, "relu&sigmoid")
    print ("AL",AL)
    print ("caches",caches)
    
    Return:
    ('AL', array([[ 0.4993538 ],
       [ 0.49992187]]))
    ('caches', [((array([[2],
       [3],
       [4]]), array([[-0.00217631,  0.00394372, -0.00784468],
       [ 0.01541106,  0.00394234, -0.01923715],
       [ 0.00249434,  0.00518348,  0.01985839],
       [ 0.00509152, -0.00168692,  0.00453093],
       [-0.00016401,  0.00041234,  0.0018633 ]]), array([[ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.]])), array([[-0.02390018],
       [-0.03429948],
       [ 0.09997265],
       [ 0.02324601],
       [ 0.0083622 ]]))])
    """
    caches = []
    A = X
    L = len(parameters) //2  #number of layers

    if(construction == "relu&sigmoid"):
        for l in range(1,L):
            A_prev = A
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            A, cache = linear_activation_forward(A_prev, W, b, "relu")
            caches.append(cache)

        W = parameters['W' + str(L)]
        b = parameters['b' + str(L)]
        AL, cache = linear_activation_forward(A, W, b, "sigmoid")
        caches.append(cache)
        assert(AL.shape == (parameters['W'+str(L)].shape[0], X.shape[1]))

    if(construction == "relu&softmax"):
        for l in range(1,L):
            A_prev = A
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            A, cache = linear_activation_forward(A_prev, W, b, "relu")
            caches.append(cache)

        W = parameters['W' + str(L)]
        b = parameters['b' + str(L)]
        AL, cache = linear_activation_forward(A, W, b, "softmax")
        caches.append(cache)
        assert(AL.shape == (parameters['W'+str(L)].shape[0], X.shape[1]))

    return AL, caches

###################################################
################test code ###########################
#parameters = initialize_parameters([3,5,2])
#X = np.array([[2],[3],[4]])
#AL,caches = L_layer_forward(X, parameters, "relu&sigmoid")
#print ("AL",AL)
#print ("caches",caches)
#######################################################



def compute_cost(AL, Y, functionType):
    """
    compute the cost
    
    Example:
    AL = np.array([[.2,.3]])
    Y = np.array([[1,0]])
    compute_cost(AL,Y, "sigmoid")
    
    return:
    array(0.9830564281864164)
    """
    cost = 0
    if(functionType == "sigmoid"):
        N = Y.shape[1] #number of dataset
        cost = -(np.dot(Y,np.log(AL).T)+np.dot(1-Y,np.log(1-AL).T))/N #cross entropy
        
        cost = np.squeeze(cost)
        
    elif(functionType == "softmax"):

        N = Y.shape[1] #number of dataset
        cost = -np.sum(Y* np.log(AL))/N #cross entropy
       
        cost = np.squeeze(cost)

    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    backward propagation for a single layer
    
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
   
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    Example:
    dZ = np.array([[ 1.62434536 ,-0.61175641]])
    cache = (np.array([[-0.52817175, -1.07296862],
           [ 0.86540763, -2.3015387 ],
           [ 1.74481176, -0.7612069 ]]), np.array([[ 0.3190391 , -0.24937038,  1.46210794]]), np.array([[-2.06014071]]))
    dA_prev, dW, db = linear_backward(dZ, cache)
    print dA_prev, dW, db
    
    Return:
    [[ 0.51822968 -0.19517421]
     [-0.40506362  0.15255393]
     [ 2.37496825 -0.8944539 ]] [[-0.10076895  1.40685096  1.64992504]] [[ 0.50629448]]
    """
    A_prev, W, b = cache
    N = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T)/N
    db = np.sum(dZ, axis = 1, keepdims = True)/N
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

###########################################
##############test code##################
#dZ = np.array([[ 1.62434536 ,-0.61175641]])
#cache = (np.array([[-0.52817175, -1.07296862],
#       [ 0.86540763, -2.3015387 ],
#       [ 1.74481176, -0.7612069 ]]), np.array([[ 0.3190391 , -0.24937038,  1.46210794]]), np.array([[-2.06014071]]))
#dA_prev, dW, db = linear_backward(dZ, cache)
#print dA_prev, dW, db
###########################################

def linear_activation_backward(dA, cache, activation):
    """
    backward propagation for a whole layer(Linear->Activation)
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    
    Example:
    AL = np.array([[-0.41675785 ,-0.05626683]])
    linear_activation_cache = ((np.array([[-2.1361961 ,  1.64027081],
           [-1.79343559, -0.84174737],
           [ 0.50288142, -1.24528809]]), np.array([[-1.05795222, -0.90900761,  0.55145404]]), np.array([[ 2.29220801]])), np.array([[ 0.04153939, -1.11792545]]))
    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    Return:
    sigmoid:
    dA_prev = [[ 0.11017994  0.0110534 ]
     [ 0.09466817  0.00949723]
     [-0.05743092 -0.00576155]]
    dW = [[ 0.10266786  0.09778551 -0.01968084]]
    db = [[-0.05729622]]

    relu:
    dA_prev = [[ 0.44090989  0.        ]
     [ 0.37883606  0.        ]
     [-0.2298228   0.        ]]
    dW = [[ 0.44513825  0.37371418 -0.10478989]]
    db = [[-0.20837892]]
    """

    def relu_backward(dA, activation_cache):
        Z = activation_cache
        dZ = dA * drelu(Z)
        return dZ

    def sigmoid_backward(dA, activation_cache):
        Z = activation_cache
        dZ = dA * dsigmoid(Z)
        return dZ

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation =="sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

###################################################
####################test code#####################
'''
AL = np.array([[-0.41675785 ,-0.05626683]])
linear_activation_cache = ((np.array([[-2.1361961 ,  1.64027081],
       [-1.79343559, -0.84174737],
       [ 0.50288142, -1.24528809]]), np.array([[-1.05795222, -0.90900761,  0.55145404]]), np.array([[ 2.29220801]])), np.array([[ 0.04153939, -1.11792545]]))
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''
##################################################

def L_layer_backward(AL, Y, caches, construction):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 

    Example:
    AL = np.array([[ 1.78862847  , 0.43650985]])
    Y = np.array([[1,0]])
    caches = (((np.array([[ 0.09649747, -1.8634927 ],
           [-0.2773882 , -0.35475898],
           [-0.08274148, -0.62700068],
           [-0.04381817, -0.47721803]]), np.array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
           [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
           [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]), np.array([[ 1.48614836],
           [ 0.23671627],
           [-1.02378514]])), np.array([[-0.7129932 ,  0.62524497],
           [-0.16051336, -0.76883635],
           [-0.23003072,  0.74505627]])), ((np.array([[ 1.97611078, -1.24412333],
           [-0.62641691, -0.80376609],
           [-2.41908317, -0.92379202]]), np.array([[-1.02387576,  1.12397796, -0.13191423]]), np.array([[-1.62328545]])), np.array([[ 0.64667545, -0.35627076]])))
    grads = L_layer_backward(AL, Y, caches)
    print(grads)
    
    Result:
    {'dW2': array([[-0.39202432, -0.13325855, -0.04601089]]), 'dW1': array([[ 0.41010002,  0.07807203,  0.13798444,  0.10502167],
       [ 0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.05283652,  0.01005865,  0.01777766,  0.0135308 ]]), 'dA1': array([[ 0.        ,  0.52257901],
       [ 0.        , -0.3269206 ],
       [ 0.        , -0.32070404],
       [ 0.        , -0.74079187]]), 'dA2': array([[ 0.12913162, -0.44014127],
       [-0.14175655,  0.48317296],
       [ 0.01663708, -0.05670697]]), 'db1': array([[-0.22007063],
       [ 0.        ],
       [-0.02835349]]), 'db2': array([[ 0.15187861]])}
    """
    if(construction == "relu&sigmoid"):
        grads ={}
        L = len(caches) #number of layers
        N = AL.shape[1] 
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    ### END CODE HERE ###

        current_cache = caches[L-1]

        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],current_cache,"relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        return grads

    elif(construction == "relu&softmax"):
        grads ={}
        L = len(caches) #number of layers
        N = AL.shape[1] 
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        current_cache = caches[L-1]

        linear_cache, activation_cache = current_cache
        Z = activation_cache
        dZ = -Y + AL
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache)
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],current_cache,"relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        return grads


def update_parameters(parameters, grads, learning_rate):
    
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2  #number of layers
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters



def L_layer_model(X, Y, layers_dims, initialize_type = "random", cost_type = "sigmoid", construction = "relu&sigmoid", learning_rate = 0.005, iterations = 1000, print_cost = False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = []
    
    parameters = initialize_parameters(layers_dims, initialize_type)
    

    for i in range(iterations):
        
        AL, caches = L_layer_forward(X, parameters, construction)
        
        cost = compute_cost(AL,Y, cost_type)
        
        grads = L_layer_backward(AL, Y, caches, construction)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def make_prediction(parameters, test_x, test_y, test_y2):
    prob, cache = L_layer_forward(test_x, parameters, "relu&softmax")
    cost = compute_cost(prob, test_y, "softmax")
    
    def predict(prob):
        return np.argmax(prob, axis = 0)
    predict = predict(prob)
    correct_num = 0
    for i in range(len(predict)):
        if predict[i] == test_y2[i]:
            correct_num+=1
    predict_rate = (correct_num+.0)/len(predict)
    return cost, predict_rate