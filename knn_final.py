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
