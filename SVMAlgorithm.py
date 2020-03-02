# SVM Facial Recognition
#
# Author: Eric Walker


###########################################################################################################
# Import libraries and define static variables
###########################################################################################################

import os
import math
import copy
import imageio
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from collections import Counter


path = "ATT\\"
classes = 40
samples_per_category = 10
fold_1_files = [1, 2]; fold_2_files = [3, 4]; fold_3_files = [5, 6]; fold_4_files = [7, 8]; fold_5_files = [9, 10]


###########################################################################################################
# Class to store the 5 folds of image data and labels
###########################################################################################################
class folds:
    def __init__(self):
        self.fold_1_labels = []; self.fold_2_labels = []; self.fold_3_labels = []; self.fold_4_labels = []; self.fold_5_labels = []
        self.fold_1_images = []; self.fold_2_images = []; self.fold_3_images = []; self.fold_4_images = []; self.fold_5_images = []


###########################################################################################################
# Class to store Support Vector information for a class (labels, Lagrange multipliers, intercept)
###########################################################################################################
class svm_data:
    def __init__(self):
        self.labels = None
        self.a = None
        self.intercept = None


###########################################################################################################
# Function to load image files and create folds for cross verification
###########################################################################################################
def create_folds():

    print ("Loading Database Face Images...")
    set = folds()

    for i in range(1,classes+1):
        for j in range(1,samples_per_category+1):

            # Read the current image and reshape into column vector
            image = imageio.imread(path + str(i) + '_' + str(j) + '.png')
            image = image.reshape((image.shape[0]*image.shape[1], 1))    

            # Add image data to the proper fold
            if j in fold_1_files:
                if len(set.fold_1_images) == 0:
                    set.fold_1_images = image.copy()
                else:
                    set.fold_1_images = np.hstack((set.fold_1_images, np.copy(image)))
                set.fold_1_labels.append(i)

            elif j in fold_2_files:
                if len(set.fold_2_images) == 0:
                    set.fold_2_images = image.copy()
                else:
                    set.fold_2_images = np.hstack((set.fold_2_images, np.copy(image)))
                set.fold_2_labels.append(i)

            elif j in fold_3_files:
                if len(set.fold_3_images) == 0:
                    set.fold_3_images = image.copy()
                else:
                    set.fold_3_images = np.hstack((set.fold_3_images, np.copy(image)))
                set.fold_3_labels.append(i)

            elif j in fold_4_files:
                if len(set.fold_4_images) == 0:
                    set.fold_4_images = image.copy()
                else:
                    set.fold_4_images = np.hstack((set.fold_4_images, np.copy(image)))
                set.fold_4_labels.append(i)

            else:
                if len(set.fold_5_images) == 0:
                    set.fold_5_images = image.copy()
                else:
                    set.fold_5_images = np.hstack((set.fold_5_images, np.copy(image)))
                set.fold_5_labels.append(i)

    # Return an object containing fold data and labels
    return set


###########################################################################################################
# Function to train the SVM classifier with training data and training data labels
###########################################################################################################
def train(training_data, training_data_labels, c_val, margin_const, fxn):

    samples = training_data.shape[1]
    svm_class_data = {}
    subjects = Counter(training_data_labels)
    subjects = subjects.keys()

    # Iterate through all 40 subjects to create an svm_data() object for each
    for subject in subjects:
        one_v_rest = []

        # Set 1 or -1 labels for current training subject vs rest
        for label in training_data_labels:
            if label == subject:
                one_v_rest.append(1.0)
            else:
                one_v_rest.append(-1.0)
        
        # Obtain the kernel matrix values
        kernel_result = generate_kernel_matrix(training_data, fxn)

        # Create the inputs for the cvxopt quadratic programming solver 
        # Reference for configuring inputs to solvers.qp(): https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
        P = matrix(np.outer(one_v_rest,one_v_rest) * kernel_result)
        q = matrix(np.ones(samples) * -1.0)
        tmp1 = np.diag(np.ones(samples) * -1.0)
        tmp2 = np.identity(samples)
        G = matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(samples)
        tmp2 = np.ones(samples) * c_val
        h = matrix(np.hstack((tmp1, tmp2)))
        A = matrix(one_v_rest, (1,samples))
        b = matrix(0.0)
        # Run the quadratic programming solver
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        vectors = np.ravel(solution['x'])

        # Calculate the intercept value given the Lagrange multipliers and kernel matrix
        intercept = calculate_intercept(kernel_result, one_v_rest, vectors, margin_const)

        # Create an svm_data() object for the current subject and assign the Lagrange multipliers, intercept, and 1/-1 labels
        svm_class_data[subject] = svm_data()
        svm_class_data[subject].labels = list(one_v_rest)
        svm_class_data[subject].a = vectors.copy()
        svm_class_data[subject].intercept = intercept.copy()

    return svm_class_data


###########################################################################################################
# Function to test the SVM classifier with test images and their associated labels
###########################################################################################################
def test(training_data, testing_data, testing_data_labels, svm_objs, margin_const, fxn, output_file):
    
    # Create output results file
    if os.path.exists(output_file):
        os.remove(output_file)
    f = open(output_file,"w+")

    # Iterate through each test image
    correct = 0
    for i in range(len(testing_data_labels)):
        f_vals = []
        f_classes = []

        # Calculate the projection of the test image using svm_obj data from all 40 classes (itarate through each class)
        for subject in svm_objs:
            f_next = projection(testing_data[:,i], training_data, svm_objs[subject].labels, svm_objs[subject].intercept, svm_objs[subject].a, margin_const, fxn)
            f_vals.append(f_next)
            f_classes.append(subject)
        
        # Find the class with the positive projection value (or max value)
        index = f_vals.index(max(f_vals))

        # Verify if the predicted class matches the actual class
        if (f_classes[index] == testing_data_labels[i]):
            correct += 1
            outcome = "correct"
        else:
            outcome = "incorrect"

        # Write results to output file
        f.write("Test Person Label: " + str(testing_data_labels[i]) + "     Predicted Person Label: " + str(f_classes[index]) + "     Result: " + outcome + "\n")

    accuracy = (float(correct)/float(len(testing_data_labels)))*100.0

    # Write the accuracy rate to the output file for the fold
    f.write("Accuracy Rate: " + str(accuracy) + "%")
    f.close()
    return accuracy


###########################################################################################################
# Function to calculate kernel value matrix for the training images
###########################################################################################################
def generate_kernel_matrix(training_data, fxn):

    samples = training_data.shape[1]
    # Create the nxn kernel matrix (where n is the total number of training images, 320)
    kernel_result = matrix(np.zeros((samples,samples)))

    # Calculate the nxn kernel matrix where n is the number of training images
    for i in range(samples):
        for j in range(samples):
            kernel_result[i,j] = kernel_fxn_type(training_data[:,i],training_data[:,j], fxn)
    return kernel_result


###########################################################################################################
# Function to calculate individual kernel values depending on linear or 2nd order polynomial type
###########################################################################################################
def kernel_fxn_type(data1, data2, fxn):

    # Provide the option to select between linear or polynomial kernel function
    if (fxn == "Linear"):
        result = float(np.dot(data1,data2))
    elif (fxn == "Polynomial"):
        transposed = np.transpose(data1)
        result = math.pow((np.dot(transposed,data2) + 1.0), 2.0)

    return result
    

###########################################################################################################
# Function to calculate the intercept for the support vectors
###########################################################################################################
def calculate_intercept(kernel_result, one_v_rest, vectors, margin_const):

    # Keep the Lagrange multipliers greater than margin constant (aka 0)
    sv = vectors > margin_const
    arranged = np.arange(len(vectors))[sv]
    k_new = np.copy(kernel_result)
    sv_y = []

    # Create the sv_y vector with 1/-1 labels
    for i in arranged:
       sv_y.append(one_v_rest[i])

    chosen_vectors = vectors[sv]
    intercept = 0

    # Execute the formula to calculate intercept using the kernel function with the sorted Lagrange multipliers
    for i in range(len(chosen_vectors)):
        intercept += sv_y[i]
        for j in range(len(chosen_vectors)):
            intercept -= chosen_vectors[j] * sv_y[j] * k_new[arranged[i],arranged[j]]

    intercept = intercept / len(chosen_vectors)

    return intercept


###########################################################################################################
# Function to calculate the projection of the test image using the Lagrange multipliers and intercept
###########################################################################################################
def projection(test_image, training_data, training_data_labels, intercept, a, margin_const, fxn):

    samples = training_data.shape[1]

    # Build the new alpha array based on if the imported support vectors are greater than the margin value
    alpha = []
    for i in range(samples):
        if a[i] < margin_const:
            alpha.append(0.0)
        else:
            alpha.append(a[i])

    # Execute the formula to calculate f(x) for the given test image
    f_x = 0
    for i in range(samples):
        f_x += alpha[i] * training_data_labels[i] * kernel_fxn_type(training_data[:,i],test_image,fxn)

    f_x = f_x + intercept
    return f_x


###########################################################################################################
# Main Entry Point for Script
###########################################################################################################
if __name__ == '__main__':

    # Uncomment the following line to run linear SVM
    kernel_fxn = 'Linear'

    # Uncomment the following line to run polynomial SVM
    #kernel_fxn = 'Polynomial'

    # Read image data and store in folds object
    set = create_folds()

    margin_const = 1e-17
    c_val = 50.0
    total_accuracies = []

	
    # Fold 1
    output_file = "Fold_1_Results.txt"
    testing_data = set.fold_1_images
    testing_data_labels = set.fold_1_labels
    training_data = np.hstack((set.fold_2_images, set.fold_3_images, set.fold_4_images, set.fold_5_images))
    training_data_labels = set.fold_2_labels + set.fold_3_labels + set.fold_4_labels + set.fold_5_labels
    training_data = training_data.astype(np.float, copy = True)
    testing_data = testing_data.astype(np.float, copy = True)
    print ("Training SVM for Fold 1...")
    svm_objs = train(training_data, training_data_labels, c_val, margin_const, kernel_fxn)
    print ("Testing SVM for Fold 1...")
    accuracy = test(training_data, testing_data, testing_data_labels, svm_objs, margin_const, kernel_fxn, output_file)
    print ("Accuracy Rate for Fold 1: " + str(accuracy) + "%")
    total_accuracies.append(accuracy)


    # Fold 2
    del testing_data, testing_data_labels, training_data, training_data_labels, svm_objs
    output_file = "Fold_2_Results.txt"
    testing_data = set.fold_2_images
    testing_data_labels = set.fold_2_labels
    training_data = np.hstack((set.fold_1_images, set.fold_3_images, set.fold_4_images, set.fold_5_images))
    training_data_labels = set.fold_1_labels + set.fold_3_labels + set.fold_4_labels + set.fold_5_labels
    training_data = training_data.astype(np.float, copy = True)
    testing_data = testing_data.astype(np.float, copy = True)
    print ("Training SVM for Fold 2...")
    svm_objs = train(training_data, training_data_labels, c_val, margin_const, kernel_fxn)
    print ("Testing SVM for Fold 2...")
    accuracy = test(training_data, testing_data, testing_data_labels, svm_objs, margin_const, kernel_fxn, output_file)
    print ("Accuracy Rate for Fold 2: " + str(accuracy) + "%")
    total_accuracies.append(accuracy)


    # Fold 3
    del testing_data, testing_data_labels, training_data, training_data_labels, svm_objs
    output_file = "Fold_3_Results.txt"
    testing_data = set.fold_3_images
    testing_data_labels = set.fold_3_labels
    training_data = np.hstack((set.fold_1_images, set.fold_2_images, set.fold_4_images, set.fold_5_images))
    training_data_labels = set.fold_1_labels + set.fold_2_labels + set.fold_4_labels + set.fold_5_labels
    training_data = training_data.astype(np.float, copy = True)
    testing_data = testing_data.astype(np.float, copy = True)
    print ("Training SVM for Fold 3...")
    svm_objs = train(training_data, training_data_labels, c_val, margin_const, kernel_fxn)
    print ("Testing SVM for Fold 3...")
    accuracy = test(training_data, testing_data, testing_data_labels, svm_objs, margin_const, kernel_fxn, output_file)
    print ("Accuracy Rate for Fold 3: " + str(accuracy) + "%")
    total_accuracies.append(accuracy)


    # Fold 4
    del testing_data, testing_data_labels, training_data, training_data_labels, svm_objs
    output_file = "Fold_4_Results.txt"
    testing_data = set.fold_4_images
    testing_data_labels = set.fold_4_labels
    training_data = np.hstack((set.fold_1_images, set.fold_2_images, set.fold_3_images, set.fold_5_images))
    training_data_labels = set.fold_1_labels + set.fold_2_labels + set.fold_3_labels + set.fold_5_labels
    training_data = training_data.astype(np.float, copy = True)
    testing_data = testing_data.astype(np.float, copy = True)
    print ("Training SVM for Fold 4...")
    svm_objs = train(training_data, training_data_labels, c_val, margin_const, kernel_fxn)
    print ("Testing SVM for Fold 4...")
    accuracy = test(training_data, testing_data, testing_data_labels, svm_objs, margin_const, kernel_fxn, output_file)
    print ("Accuracy Rate for Fold 4: " + str(accuracy) + "%")
    total_accuracies.append(accuracy)


    # Fold 5
    del testing_data, testing_data_labels, training_data, training_data_labels, svm_objs
    output_file = "Fold_5_Results.txt"
    testing_data = set.fold_5_images
    testing_data_labels = set.fold_5_labels
    training_data = np.hstack((set.fold_1_images, set.fold_2_images, set.fold_3_images, set.fold_4_images))
    training_data_labels = set.fold_1_labels + set.fold_2_labels + set.fold_3_labels + set.fold_4_labels
    training_data = training_data.astype(np.float, copy = True)
    testing_data = testing_data.astype(np.float, copy = True)
    print ("Training SVM for Fold 5...")
    svm_objs = train(training_data, training_data_labels, c_val, margin_const, kernel_fxn)
    print ("Testing SVM for Fold 5...")
    accuracy = test(training_data, testing_data, testing_data_labels, svm_objs, margin_const, kernel_fxn, output_file)
    print ("Accuracy Rate for Fold 5: " + str(accuracy) + "%")
    total_accuracies.append(accuracy)


    print ("\n-------------------------------------------------------")
    print ("\nAverage Accuracy: " + str(np.mean(total_accuracies)) + "%")

    os.system("pause")