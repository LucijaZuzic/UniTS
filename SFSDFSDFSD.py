import os

for file in os.listdir():
    if ("__MS.txt") in file:
        os.rename(file, file.replace("__MS", "_MS"))