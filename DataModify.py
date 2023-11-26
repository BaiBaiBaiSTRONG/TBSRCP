import random


catogaory_pred = []
with open("./category_pred.txt", "r") as f:
     lines = f.readlines()
     lines = [line.strip() for line in lines]
     catogaory_pred.append(lines)

# random change some of the prediction result to see the change of confusion matrix
for i in range(100):
    catogaory_pred[0][random.randint(0, len(catogaory_pred[0])-1)] = str(random.randint(1, 25))
