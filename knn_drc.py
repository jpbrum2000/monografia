#!/home/joao/anaconda3/bin/python3.6
import numpy as np
from collections import Counter
import sys
from subprocess import PIPE, run
distance_matrix = np.load('distance_matrix.npy')
#--
# Passo 1
# Carregar classes diretorio
#----------
objetos = np.empty([2100], dtype=object)
labels = np.empty([2100], dtype=object)
classes = np.array(["agricultural","airplane","baseballdiamond","beach","buildings","chaparral","denseresidential","forest","freeway","golfcourse","harbor","intersection","mediumresidential","mobilehomepark","overpass","parkinglot","river","runway","sparseresidential","storagetanks","tenniscourt"])
index=0
for y in range(0,21):
    for x in range(0,100):
        objetos[index] = "UCMerced_LandUse/Images/"+classes[y]+"/"+classes[y]+str(x).zfill(2) +".fv"
        labels[index] = classes[y]
        index += 1
		
#--
# Passo 2
# Definir funcao de calculo de distancia distance
#----------
def distance(instance1, instance2):
    command = ["bic/source/bin/bic_distance",instance1,instance2]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return float(result.stdout)
def distance2(instance1, instance2):
    d = float(distance_matrix[np.where(objetos==instance1)[0][0]][np.where(objetos==instance2)[0][0]])
    return d
#print(distance(learnset_data[1], learnset_data[2]))
#print(distance(learnset_data[3], learnset_data[44]))

#--
# Passo 6
# Definir funcao de busca dos k vizinhos mais proximos e ordena por numero de votos
#----------
def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  K, 
                  distance=distance):
    """
    get_neighbors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with  
    (index, dist, label) where 
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
    class_counter = Counter()
    ocorr_counter = Counter()
    for d in distances:
        class_counter[d[2]] += d[1]
        ocorr_counter[d[2]] += 1

    neighbors = []

    for e in list(class_counter):
    	neighbors.append((e,class_counter[e]/ocorr_counter[e],e))
    neighbors.sort(key=lambda x: x[1])
    return neighbors

#--
# Passo 7
# Escolher a Classe ganhadora
#----------
def vote_prob(neighbors,T):
    n1 = neighbors[0]
    for x in neighbors[1:]:
        if(x[2] != n1[2]):
            n2=x
            break
    if((n1[1]/n2[1])<=T):
        return n1[2], n1[1]/n2[1]
    else:
        return 'unknow', n1[1]/n2[1]

#--
# Passo 8
# Preencher confusion Matriz
#----------
def confusion_matrix(n_classes_conhecidas, T, treino, treino_labels, test, test_labels):
	confusion_matrix = np.zeros([n_classes_conhecidas+1, n_classes_conhecidas+1])
	classes=np.append(np.unique(treino_labels),'unknow')
	for i in range(len(test)):
		neighbors = get_neighbors(treino, 
								  treino_labels, 
								  test[i], 
								  0, 
								  distance=distance2)
		#print("index: ",i,", vote_prob: ", vote_prob(neighbors,T),", label: ", test_labels[i],", data: ", test[i])
		#print(np.where(classes==test_labels[i])[0][0])
		#print(np.where(classes==vote_prob(neighbors)[0])[0][0])
		#print(confusion_matrix)
		if(any(classes==test_labels[i])):
			confusion_matrix[np.where(classes==test_labels[i])[0][0]][np.where(classes==vote_prob(neighbors,T)[0])[0][0]] += 1
		else:
			confusion_matrix[np.where(classes=='unknow')[0][0]][np.where(classes==vote_prob(neighbors,T)[0])[0][0]] += 1
	return confusion_matrix
		
