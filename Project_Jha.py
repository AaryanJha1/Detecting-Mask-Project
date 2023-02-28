import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ast
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

def readFiles():
    batch_size = 32
    img_height = 75
    img_width = 75

    data_dir = '/Users/aaryanjha/Desktop/Final_Project/data/'
    training_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = training_ds.class_names
    print(class_names)
    # plot(training_ds, class_names)
    #dropProp = 0.0
    #hParams = {
    #    'valProportion': 0.2,
    #    'numEpochs': 10,
    #    'convLayers': [{
    #        'conv_numFilters': 32,
    #        'conv_f': 3,
    #        'conv_p': 'same',
    #        'conv_act': 'relu',
    #        'pool_f': 2,
    #        'pool_s': 2,
    #        'drop_prop': dropProp
    #    }],
    #    'denseLayers': [64, 2],
    #}
    
    #expNames = ['C32_d0.0_D128_5_adam_100','C32_64_d0.0_D128_5_adam_100']
    expNames = ['C32_d0.0_D128_2_adam_100','C32_64_d0.1_D128_2_adam_100','C32_64_128__d0.2_D128_2_adam_100']
    dataSubsets = (training_ds, testing_ds)
    for currExp in expNames:
        hParams = getHParams(currExp)
        trainResults, testResults = cnnGray(dataSubsets,hParams)
        writeExperimentalResults(hParams, trainResults, testResults)
    buildTrainingPlot(expNames, "Figure 1 - Training Plot")
    buildPlot(expNames,"Figure 1 - Validation Plot")
    buildTestAccuracyPlot(expNames,"Figure 1 - Testing Plot")

    history, test_acc = cnnGray(dataSubsets, hParams)

    print("Test accuracy: ", test_acc)

def cnnGray(dataSubsets, hParams):
    (training_ds, testing_ds) = dataSubsets
    image_width = 75
    image_height = 75
    num_channels = 3


    model = Sequential()

    for i in range(len(hParams['convLayers'])):
        conv_layer = hParams['convLayers'][i]
        model.add(Conv2D(conv_layer['conv_numFilters'], (conv_layer['conv_f'],conv_layer['conv_f']), padding = conv_layer['conv_p'], input_shape=(image_width,image_height,num_channels), activation = conv_layer['conv_act']))
        model.add(MaxPooling2D(pool_size = (conv_layer['pool_f'], conv_layer['pool_f']), strides=(conv_layer['pool_s'],conv_layer['pool_s'])))
        
        if conv_layer['drop_prop'] > 0:
            model.add(Dropout(conv_layer['drop_prop']))

    model.add(Flatten())

    for i in range(len(hParams['denseLayers'])):
        if (i != len(hParams['denseLayers'])-1):
            model.add(Dense(hParams['denseLayers'][i], activation= 'relu'))
        else:
            model.add(Dense(hParams['denseLayers'][i]))
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    hist = model.fit(training_ds, validation_data=testing_ds, epochs = hParams['numEpochs'], verbose = 1)
    
    test_loss, test_acc = model.evaluate(testing_ds)
    return hist.history, test_acc 

def getHParams(expName = None):
    hParams = {
        'experimentName': expName,
        'dataProportion' : 1.0,
        'numEpochs': 10,
        'trainingProportion': 0.7,
        'valProportion': 0.1
    }
    shortTest = False
    if shortTest:
        print("+++++++++++++++ WARNING: SHORT TEST ++++++++++++++++++")
        hParams['datasetProportion'] = 0.01
        hParams['numEpochs'] = 2
        
    if (expName is None):
        # Not running an experiment yet, so just return the "common" parameters
        return hParams
        
    if(expName == 'C32_d0.0_D128_2_adam_100'):
        dropProp = 0.0
        hParams['convLayers'] = [{
                        'conv_numFilters': 32,
                        'conv_f': 3,
                        'conv_p': "same",
                        'conv_act': 'relu',
                        'pool_f': 2,
                        'pool_s':2,
                        'drop_prop': dropProp
                        }]                        
        hParams['denseLayers'] = [128,2]
        hParams['optimizer'] = 'adam'     
    
    if(expName == 'C32_64_d0.1_D128_2_adam_100'):
        dropProp = 0.1
        hParams['convLayers'] = [{
                        'conv_numFilters': 32,
                        'conv_f': 3,
                        'conv_p': "same",
                        'conv_act': 'relu',
                        'pool_f': 2,
                        'pool_s':2,
                        'drop_prop': dropProp
                        },
                        {
                        'conv_numFilters': 64,
                        'conv_f': 3,
                        'conv_p': "same",
                        'conv_act': 'relu',
                        'pool_f': 2,
                        'pool_s':2,
                        'drop_prop': dropProp
                        }]                        
        hParams['denseLayers'] = [128,2]
        hParams['optimizer'] = 'adam'
        
    if(expName == 'C32_64_128__d0.2_D128_2_adam_100'):
        dropProp = 0.2
        hParams['convLayers'] = [{
                            'conv_numFilters': 32,
                            'conv_f': 3,
                            'conv_p': "same",
                            'conv_act': 'relu',
                            'pool_f': 2,
                            'pool_s':2,
                            'drop_prop': dropProp
                            },
                            {
                            'conv_numFilters': 64,
                            'conv_f': 3,
                            'conv_p': "same",
                            'conv_act': 'relu',
                            'pool_f': 2,
                            'pool_s':2,
                            'drop_prop': dropProp
                            },
                            {
                            'conv_numFilters': 128,
                            'conv_f': 3,
                            'conv_p': "same",
                            'conv_act': 'relu',
                            'pool_f': 2,
                            'pool_s':2,
                            'drop_prop': dropProp
                            }]                        
        hParams['denseLayers'] = [128,2]
        hParams['optimizer'] = 'adam'
    
    return hParams


def writeExperimentalResults(hParams, trainResults, testResults):
    f = open("results/" + hParams['experimentName'] + ".txt", "w")
    f.write(str(hParams)+"\n")
    f.write(str(trainResults)+"\n")
    f.write(str(testResults))
    f.close()
    
def readExperimentalResults(nameOfFile):
    f = open("results/" + nameOfFile + ".txt","r") 
    results = f.read().split("\n")
    hParams = ast.literal_eval(results[0])
    trainResults = ast.literal_eval(results[1])
    testResults = ast.literal_eval(results[2])
    return hParams, trainResults, testResults

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    yLabelStr = "__" + "__".join([label for label in yLabelList])
    filepath = "results/" + title + " " + yLabelStr + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)  
    
def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)
    
def buildPlot(expNames,fileName):    
    experiments = expNames
    fig, ax = plt.subplots()
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        itemsToPlot = ['val_accuracy']
        x = np.arange(0,hParams['numEpochs'])
        y = np.array(trainResults['val_accuracy']).transpose()
        ax.plot(x,y)
        ax.set(xlabel = "Epoch", title = "Val Accuracy plot")
        ax.grid
    filepath = "results/" + fileName + ".png"
    plt.legend(experiments, loc='best', shadow=True)
    fig.savefig(filepath)
    
def buildTestAccuracyPlot(expName, fileName):
    experiments = expName
    count = 0
    x =[]
    test_accuracies = []
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        count += 1
        x.append(count)
        test_accuracies.append(testResults)
    plotPoints(x, test_accuracies, pointLabels = experiments, xLabel="parameter count", yLabel="Test Set Accuracy", title="Test Set Accuracy", filename= fileName)

def buildTrainingPlot(expNames,fileName):    
    experiments = expNames
    fig, ax = plt.subplots()
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        x = np.arange(0,hParams['numEpochs'])
        y = np.array(trainResults['accuracy']).transpose()
        ax.plot(x,y)
        ax.set(xlabel = "Epoch", title = "Training Accuracy plot")
        ax.grid
    filepath = "results/" + fileName + ".png"
    plt.legend(experiments, loc='best', shadow=True)
    fig.savefig(filepath)
    
    
readFiles()