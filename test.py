import subprocess
#--
# Testa com numero de classes conhecidas 3,6,9,12,15,21
#----------
print(3)
p1 = subprocess.Popen(["python","grid_search.py",str(3)])
print(6)
p2 = subprocess.Popen(["python","grid_search.py",str(6)])
print(9)
p3 = subprocess.Popen(["python","grid_search.py",str(9)])
p1.wait()
print(12)
p4 = subprocess.Popen(["python","grid_search.py",str(12)])
p2.wait()
print(15)
p5 = subprocess.Popen(["python","grid_search.py",str(15)])
p3.wait()
print(21)
p6 = subprocess.Popen(["python","grid_search.py",str(21)])