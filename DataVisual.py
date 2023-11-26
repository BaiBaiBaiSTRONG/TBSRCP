import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
classe = ["Crayon","Doraemon","Elephant","MashiMaro","Mermaid","Minions","PeppaPig","Pikachu","Smurf","SpongeBob","bicycle","butterfly","car","cup","dinosaur","dolphin","house","kartell","mickey","plane","keyboard","tree","umbrella","watch","Snowman","donald","Garfield","Twilight"]
classes = ["Crayon","Doraemon","Elephant","MashiMaro","Mermaid","Minions","PeppaPig","Pikachu","Smurf","SpongeBob","bicycle","butterfly","car","cup","dinosaur","dolphin","house","kartell","mickey","plane","keyboard","tree","umbrella","watch","Snowman"]
str2int = {"Crayon": 1, "Doraemon": 2, "Elephant": 3, "MashiMaro": 4, "Mermaid": 5, "Minions": 6, "PeppaPig": 7, "Pikachu": 8, "Smurf": 9, "SpongeBob": 10, "bicycle": 11, "butterfly": 12, "car": 13, "cup": 14, "dinosaur": 15, "dolphin": 16, "house": 17, "kartell": 18, "mickey": 19, "pigeon": 20, "plane": 21, "tree": 22, "umbrella": 23, "watch": 24, "Snowman":25, "donald":26, "Garfield":27, "Twilight":28}



sns.set()
f, ax = plt.subplots()
labels = ["".join("c" + str(i)) for i in range(0, 28)]

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)
# y_true = [[0.37, 0, 0.7], [0.5, 1, 0.9]]
# y_pred = [[1, 0, 0], [0, 1, 1]]

# # 转换y_true为bool类型,选取最大的为1，其余为0
# y_true_array = np.array(y_true)
# y_pred_array = np.array(y_pred)
# # print(y_true)
# y_pred_array = (y_pred_array == y_pred_array.max(axis=1)[:, None]).astype(int)

# c2 = multilabel_confusion_matrix(y_pred_array, y_pred_array)
c3 = [[[17722,10],[6,17]],[[17024, 28],[0, 703]], [[16941, 60],[74, 680]],[[17010, 151], [56, 38]], [[16979, 6], [27, 743]], [[17011, 52], [6, 686]], [[17026, 12], [37, 680]], [[17056, 5], [7, 687]], [[16985, 25], [81, 664]], [[17036, 120], [5, 594]], [[17024, 33],[30, 668]], [[17084, 64],[4, 603]], [[17045, 32], [2, 676]], [[17011, 53], [25, 666]], [[17017, 20], [22, 696]], [[16726, 6], [342, 681]], [[17007, 55], [6, 687]], [[17003, 5], [4, 743]], [[17155, 8], [29, 563]], [[16970, 102], [38, 645]], [[17006, 3], [15, 731]], [[17028, 42], [30, 655]], [[16945, 29], [114, 667]], [[17021, 16], [11, 707]], [[16961, 16], [66, 712]], [[17003, 62], [15, 675]], [[17755, 0], [0, 0]], [[17755, 0], [0, 0]]]
# calculate the confusion matrix for each class according to c3 and display the result as a table

for i in range(0, 28):
    tmpTp = c3[i][0][0]
    tmpFn = c3[i][0][1]
    tmpFp = c3[i][1][0]
    tmpTn = c3[i][1][1]
    tmpF1 = 2 * tmpTp / (2 * tmpTp + tmpFn + tmpFp)
    tmpAcc = (tmpTp + tmpTn) / (tmpTp + tmpFn + tmpFp + tmpTn)
    tmpRecall = tmpTp / (tmpTp + tmpFn)
    tmpPrecision = tmpTp / (tmpTp + tmpFp)
    # print 




# Visualize confusion matrix as heatmap
# fig, ax = plt.subplots(7, 4, figsize=(20, 20))
# for axes, c in zip(ax.flatten(), c3):
#     print_confusion_matrix(c, axes, labels[0], ["0", "1"])
#     labels.pop(0)
# plt.tight_layout()
# plt.show()

# Calculate Accuracy, Precision, Recall, F1-score and MCC according to confusion matrix for each class, display the results as a table
# 废物了，不用看了
# disp = ConfusionMatrixDisplay(confusion_matrix=np.array(c3) , display_labels=classes)
# disp.plot(
#     include_values=True,            # 混淆矩阵每个单元格上显示具体数值
#     cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
#     ax=None,                        # 同上
#     xticks_rotation="horizontal",   # 同上
#     values_format="d"               # 显示的数值格式
# )
# plt.show()

# read the catagory prediction result from the file
catogaory_pred = []
catogaory_true = []
with open("./category_pred.txt", "r") as f:
     lines = f.readlines()
     lines = [line.strip() for line in lines]
     catogaory_pred.append(lines)
with open("./category_true.txt", "r") as f:
     lines = f.readlines()
     lines = [line.strip() for line in lines]
     catogaory_true.append(lines)

# random change some of the prediction result to see the change of confusion matrix
for i in range(4000):
    catogaory_pred[0][random.randint(0, len(catogaory_pred[0])-1)] = str(random.randint(1, 25))

# print(np.array(catogaory_pred).shape)
# print(np.array(catogaory_true).shape)
b=[]
for i in range(len(catogaory_true)):
    b.append(np.array(catogaory_true[i]))
c=np.array(b)
true = c
b=[]
for i in range(len(catogaory_pred)):
    b.append(np.array(catogaory_pred[i]))
c=np.array(b)
pred = c
true = true.reshape(-1)
pred = pred.reshape(-1)


confusion_matrix = confusion_matrix(true, pred)

# normalize the confusion matrix
# confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
disp.plot(
    include_values=False,            # 混淆矩阵每个单元格上显示具体数值
    cmap="rocket_r",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
    ax=None,colorbar=True,                        # 同上
    xticks_rotation="horizontal",   # 同上
    values_format="456"               # 显示的数值格式

)
plt.show()

