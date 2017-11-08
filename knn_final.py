#--
# Passo 1
# Carregar classes
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
# Definir Classes Conhecidas e das Classes Desconhecidas
#----------
n_classes_conhecidas = 20
labels_conhecidos = labels[np.where(np.in1d(labels,classes[:n_classes_conhecidas]))]
objetos_conhecidos = objetos[np.where(np.in1d(labels,classes[:n_classes_conhecidas]))]
labels_desconhecidos = labels[np.where(np.in1d(labels,classes[n_classes_conhecidas:]))]
objetos_desconhecidos = objetos[np.where(np.in1d(labels,classes[n_classes_conhecidas:]))]

#--
# Passo 3
# Dividir Treino e Teste
#treino 50% dos conhecido 
#test = 50% dos conhecido e desconhecidos
#----------
porcentagem_do_treino = 50.0/100.0
treino = objetos_conhecidos[:int(len(labels_conhecidos)*porcentagem_do_treino)]
treino_labels = labels_conhecidos[:int(len(labels_conhecidos)*porcentagem_do_treino)]
test = np.append(objetos_conhecidos[int(len(labels_conhecidos)*porcentagem_do_treino):],objetos_desconhecidos)
test_labels = np.append(labels_conhecidos[int(len(labels_conhecidos)*porcentagem_do_treino):],objetos_desconhecidos)

#--
# Passo 4
# Definir função de calculo de distancia distance
#----------
def distance(instance1, instance2):
    command = ["bic/source/bin/bic_distance",instance1,instance2]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    
    return result.stdout
#print(distance(learnset_data[1], learnset_data[2]))
#print(distance(learnset_data[3], learnset_data[44]))

#--
# Passo 5
# Definir função de busca dos k vizinhos mais proximos
#----------
def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k, 
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
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

#--
# Passo 6
# Preencher confusion Matriz
#----------

#--
# Passo 6
# Calcular F-measure da Confusion Matriz
#----------

