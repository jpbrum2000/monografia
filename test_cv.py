import subprocess
for i in [3,6,9,12,15,21]:
        print(i)
        subprocess.Popen(["python","grid_search_cv.py",str(i)])
