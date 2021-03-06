import numpy as np
import knn as libknn
import sys
#--
# Passo 1
# Carregar classes diretorio
# Se usar a matrix de distancia 'distance_matrix.npy', não alterar a ordem das classes abaixo
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
# return labels_conhecidos, objetos_conhecidos, labels_desconhecidos, objetos_desconhecidos
#----------
def divide_know_unknow(n_classes_conhecidas,labels,objetos):
    classes = np.unique(labels)
    np.random.seed(1)
    classes = np.random.permutation(classes)
    labels_conhecidos = labels[np.where(np.in1d(labels,classes[:n_classes_conhecidas]))]
    objetos_conhecidos = objetos[np.where(np.in1d(labels,classes[:n_classes_conhecidas]))]
    labels_desconhecidos = labels[np.where(np.in1d(labels,classes[n_classes_conhecidas:]))]
    objetos_desconhecidos = objetos[np.where(np.in1d(labels,classes[n_classes_conhecidas:]))]
    return labels_conhecidos, objetos_conhecidos, labels_desconhecidos, objetos_desconhecidos

#--
# Passo 3
# Dividir Treino e Teste
#treino 75% dos conhecido 
#test = 25% dos conhecido e desconhecidos
#return treino, treino_labels, test, test_labels
#----------
def divide_training_test(FOLD, labels_conhecidos, objetos_conhecidos, labels_desconhecidos, objetos_desconhecidos):
	treino = []
	treino_labels = []
	test = []
	test_labels = []
	for c in np.unique(labels_conhecidos):
		lc = labels_conhecidos[np.where(labels_conhecidos==c)]
		oc = objetos_conhecidos[np.where(labels_conhecidos==c)]
		treino = np.append(treino,np.concatenate(np.delete(np.array_split(oc,2),FOLD,axis=0)))
		treino_labels = np.append(treino_labels,np.concatenate(np.delete(np.array_split(lc,2),FOLD,axis=0)))
		test = np.append(test,np.array_split(oc,2)[FOLD])
		test_labels = np.append(test_labels,np.array_split(lc,2)[FOLD])
	test = np.append(test,objetos_desconhecidos)
	test_labels = np.append(test_labels,labels_desconhecidos)
	return treino, treino_labels, test, test_labels

#--
# Passo 4
# Calcular Accuracy e F-measure da Confusion Matriz
# return accuracy, f_measure_macro, f_measure_micro
#----------
def evaluated(n_classes_conhecidas, confusion_matrix):
	accuracy_knows = 0
	accuracy_unknows = 0
	precision_macro = 0
	recall_macro = 0
	f_measure_macro = 0
	precision_micro = 0
	recall_micro = 0
	f_measure_micro = 0
	
	#Calcular accuracy_knows
	diag=0
	total=0
	for x in range(n_classes_conhecidas):
		for y in range(n_classes_conhecidas+1):
			if(x==y):
				diag += confusion_matrix[x][y]
			total += confusion_matrix[x][y]
	accuracy_knows = diag/total
	
	#Calcular accuracy_unknows
	diag=0
	total=0
	x = n_classes_conhecidas
	for y in range(n_classes_conhecidas+1):
		if(x==y):
			diag += confusion_matrix[x][y]
		total += confusion_matrix[x][y]
	if(diag!=0):
		accuracy_unknows = diag/total	
	else:
		accuracy_unknows = 0
	
	#Calcula precision_macro
	precision_macro=0.0
	for x in range(n_classes_conhecidas):
		false_positive=0
		for y in range(n_classes_conhecidas+1):
			if (x!=y):
				false_positive += confusion_matrix[y][x]
		if(confusion_matrix[x][x]!=0):
			precision_macro += confusion_matrix[x][x]/(confusion_matrix[x][x]+false_positive)
	precision_macro = precision_macro/n_classes_conhecidas
	#Calcula recall_macro
	recall_macro=0.0
	for x in range(n_classes_conhecidas):
		false_negative=0
		for y in range(n_classes_conhecidas+1):
			if (x!=y):
				false_negative += confusion_matrix[x][y]
		if(confusion_matrix[x][x]!=0):
			recall_macro += confusion_matrix[x][x]/(confusion_matrix[x][x]+false_negative)
	recall_macro = recall_macro/n_classes_conhecidas
	#Calcula f_measure_macro
	f_measure_macro = (2*precision_macro*recall_macro)/(precision_macro+recall_macro)
	
	#Calcula precision_micro
	sum_true_positive = 0
	sum_precision_divisor = 0
	for x in range(n_classes_conhecidas):
		sum_false_positive = 0
		for y in range(n_classes_conhecidas+1):
			if (x!=y):
				sum_false_positive += confusion_matrix[y][x]
		sum_precision_divisor += confusion_matrix[x][x] + sum_false_positive
		sum_true_positive += confusion_matrix[x][x]
	precision_micro = sum_true_positive/sum_precision_divisor
	#Calcula recall_micro
	sum_true_positive = 0
	sum_recall_divisor = 0
	for x in range(n_classes_conhecidas):
		sum_false_negative = 0
		for y in range(n_classes_conhecidas+1):
			if (x!=y):
				sum_false_negative += confusion_matrix[x][y]
		sum_recall_divisor += confusion_matrix[x][x] + sum_false_negative
		sum_true_positive += confusion_matrix[x][x]
	recall_micro = sum_true_positive/sum_recall_divisor
	
	#Calcula f_measure_micro
	f_measure_micro = (2*precision_micro*recall_micro)/(precision_micro+recall_micro)
	
	return accuracy_knows, accuracy_unknows, f_measure_macro, f_measure_micro

def grid_search(n_classes_conhecidas):
	accuracies = []
	for K in [1,3,5,7,9,11,13,15,17,19,21]:
		accuracy_aux = 0
		for FOLD in range(2):
			treino, treino_labels, test, test_labels = divide_training_test(0,*divide_know_unknow(n_classes_conhecidas,labels,objetos))
			#confusion_matrix(n_classes_conhecidas, K, treino, treino_labels, test, test_labels)
			confusion_matrix = libknn.confusion_matrix(n_classes_conhecidas, K, *divide_training_test(FOLD,*divide_know_unknow(n_classes_conhecidas,treino_labels,treino)))
			accuracy_aux += evaluated(n_classes_conhecidas,confusion_matrix)[0]
		#print(K,accuracy_aux/2)
		accuracies.append((K,accuracy_aux/2))
	accuracies.sort(key=lambda x: x[1],reverse=True)
	print(accuracies)
	return accuracies

def knn(K,n_classes_conhecidas):
	confusion_matrix = libknn.confusion_matrix(n_classes_conhecidas,K,*divide_training_test(0,*divide_know_unknow(n_classes_conhecidas,labels,objetos)))
	return 'accuracy_knows,accuracy_unknows,f_measure_macro,f_measure_micro'+str(evaluated(n_classes_conhecidas,confusion_matrix))

n_classes = int(sys.argv[1])
f = open('testes\knn\knn_result_'+str(n_classes), 'a')
for K in grid_search(n_classes)[:1]:
    result = str('Num Classes,'+str(n_classes)+',knn,'+str(K)+' '+knn(K[0],n_classes)+'\n')
    f.write(result)
f.close()
