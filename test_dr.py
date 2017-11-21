import subprocess
for i in [3,6,9,12,15,17,19,21]:
        print(i)
        subprocess.Popen(["python","grid_search_dr.py",str(i)])
