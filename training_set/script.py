import os

count= 0
for filename in os.listdir("."):
    if filename == "scrrpt.py":
        continue
    print(filename)
    if (count < 10):
        os.rename(filename, 'img046-0000%d.png' % count)
    elif(count >= 10 and count < 100): 
        os.rename(filename, 'img046-000%d.png' % count)

    count = count + 1