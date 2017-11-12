#!/home/joao/anaconda3/bin/python3.6
import numpy as np
import sys
from subprocess import PIPE, run

#
# load path location and class name of objects

objetos = np.empty([2100], dtype=object)
labels = np.empty([2100], dtype=object)
classes = np.array(["agricultural","airplane","baseballdiamond","beach","buildings","chaparral","denseresidential","forest","freeway","golfcourse","harbor","intersection","mediumresidential","mobilehomepark","overpass","parkinglot","river","runway","sparseresidential","storagetanks","tenniscourt"])
index=0
for y in range(0,21):
    for x in range(0,100):
        objetos[index] = "UCMerced_LandUse/Images/"+classes[y]+"/"+classes[y]+str(x).zfill(2) +".fv"
        labels[index] = classes[y]
        index += 1
   
#print(labels[299])

np.random.seed(int(sys.argv[1]))
indices = np.random.permutation(len(objetos))
n_training_samples = 5
learnset_data = objetos[indices[:-n_training_samples]]
learnset_labels = labels[indices[:-n_training_samples]]
testset_data = objetos[indices[-n_training_samples:]]
testset_labels = labels[indices[-n_training_samples:]]
#print(learnset_data[:4], learnset_labels[:4])
#print(testset_data[:4], testset_labels[:4])

# following line is only necessary, if you use ipython notebook!!!
#%matplotlib inline 

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#colours = ("r", "b")
#X = []
#for iclass in range(3):
#    X.append([[], [], []])
#    for i in range(len(learnset_data)):
#        if learnset_labels[i] == iclass:
#            X[iclass][0].append(learnset_data[i][0])
#            X[iclass][1].append(learnset_data[i][1])
#            X[iclass][2].append(sum(learnset_data[i][2:]))
#colours = ("r", "g", "y")
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for iclass in range(3):
#       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colour [iclass])
#plt.show()
def distance(instance1, instance2):
    command = ["bic/source/bin/bic_distance",instance1,instance2]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    
    return result.stdout
#print(distance(learnset_data[1], learnset_data[2]))
#print(distance(learnset_data[3], learnset_data[44]))

def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k, 
                  distance=distance):
    """
    get_neighbors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with  
    (index, dist, label)
    where 
    index    is the index from the training_set, 
    dist     is the distance between the test_instance and the 
             instance training_set[index]
    distance is a reference to a function used to calculate the 
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

#test with 5 firsts item from testset
#for i in range(5):
#    neighbors = get_neighbors(learnset_data, 
#                              learnset_labels, 
#                              testset_data[i], 
#                              4, 
#                              distance=distance)
#    #print(i,testset_data[i],testset_labels[i],neighbors,'\n')

from collections import Counter

def vote_prob(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if (votes4winner/sum(votes) > 0.1):
        return winner, votes4winner/sum(votes)
    else:
        return 'unknow', votes4winner/sum(votes)

for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, 
                              learnset_labels, 
                              testset_data[i], 
                              int(sys.argv[2]), 
                              distance=distance)
    print("index: ", i, 
          ", vote_prob: ", vote_prob(neighbors), 
          ", label: ", testset_labels[i], 
          ", data: ", testset_data[i])


