import os
import json
from data.config import emptyArch
from os import listdir
import numpy as np
from feature.make_dir import makeDir
import sys
from data.config import featureMap, PRIMITIVES, folder
import copy
# this file just use to plot figure that shows alphas' variation during training
def printToDecodeFile(decodeJson, kth):
    filePath = "./decode/{}th_decode.json".format(str(kth))
    f = setStdoutToFile(filePath)
    print(json.dumps(decodeJson, indent=4))
    setStdoutToDefault(f)
    
def setStdoutToFile(filePath):
    print("std output to ", filePath)
    f = open(filePath, 'w')
    sys.stdout = f
    return f

def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__
    
def pickSecondMax(input):
    # set the max to zero to and find the max again 
    # to get the index of second large value of original input
    alphas = np.copy(input)
    alphasMaxIndex = np.argmax(alphas, -1)

    for i in range(len(alphas)):
        alphas[i, alphasMaxIndex[i]] = 0
    alphasSecondMaxIndex = np.argmax(alphas, -1)

    return alphasSecondMaxIndex

def loadAllAlphas():
    listAlphas = []
    epoch = 45
    for epoch in range(epoch):
        tmp = np.load("./alpha_pdart_nodrop/alpha_prob_0_{}.npy".format(epoch))
        listAlphas.append(tmp)
    return listAlphas

def loadAlphasAtEpoch(kth, epoch):
    return  np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(str(kth), str(epoch)))
def decodeAlphas(kth):
    #* get index of innercell which hase greatest alphas value
    genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                        'genotype_' + str(kth))
    lastEpoch = 44
    lastAlphas = loadAlphasAtEpoch(kth, lastEpoch)
    maxAlphasIndex = np.argmax(lastAlphas, axis=-1)
    
    # choose top two alphas
    # print("lastAlphas", lastAlphas)
    indexOfSortAlphas = np.argsort(lastAlphas, axis=-1)
    # twoLargestIndex = indexOfSortAlphas[:, 1:, 5:]
    oneLargestIndex = indexOfSortAlphas[:, :, -1]
    # twoLargestIndex = np.reshape([0, 0, 0, 0, 0], twoLargestIndex.shape)
    # oneLargestIndex = np.reshape([4, 1, 0, 0, 0], oneLargestIndex.shape)
    # print(oneLargestIndex.shape)
    
    # print("indexOfSortAlphas", indexOfSortAlphas)
    
    # print("twoLargestIndex", twoLargestIndex)
    oneLargestIndex = np.reshape(oneLargestIndex, (5, 1, 1))
    # oneLargestIndex = np.reshape([4, 1, 0, 0, 0], (5, 1, 1))
    # twoLargestIndex = np.reshape(twoLargestIndex, (5, 2))
    # print("oneLargestIndex", oneLargestIndex.shape)
    np.save(genotype_filename, oneLargestIndex)
    
    # print("finish decode and save genotype:", maxAlphasIndex)
    return oneLargestIndex
    
def manualAssign(kth):

    makeDir("./weights_pdarts_nodrop/")
    genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                    'genotype_' + str(kth))
        
    arch = np.reshape([4, 1, 0, 0, 0], (5, 1, 1))
    np.save(genotype_filename, arch)
    return arch
def decodeOperation(allAlphas):
    takeNumOfOp = 1
    finalAlpha = allAlphas[-1] #* take the last epoch

    sortAlphaIndex = np.argsort(finalAlpha) #* from least to largest
    sortAlphaIndex = sortAlphaIndex[::-1] #* reverse ndarray
    res = np.full_like(finalAlpha, 0, dtype=np.int32)
    for i in range(takeNumOfOp):
        res[sortAlphaIndex[i]] = 1
    return res.tolist() #* make ndarray to list
def decodeAllOperation(kth, pickedLayerList=None):
    fileNameList = []
    decodeDict = {}

    #info decode operations of all layer
    if pickedLayerList==None:
        #* split file according to different kth
        for fileName in sorted(listdir(folder["alpha_pdart_nodrop"])):
            if fileName.split("th")[0]==str(kth):
                fileNameList.append(fileName)
        for fileName in fileNameList:
            #* load alpha npy file
            filePath = os.path.join(folder["alpha_pdart_nodrop"], fileName)
            allAlphas = np.load(filePath)
            key = fileName.split(".")[0]
            key = key.split("th_")[1] # eg:layer0_1
            decodeDict[key] = decodeOperation(allAlphas)
    else:
        #info create fileNameList based on decode Layer
        for fileName in sorted(listdir(folder["alpha_pdart_nodrop"])):
            for pickedLayerName in pickedLayerList:
                if (fileName.split("th")[0]==str(kth)) and (pickedLayerName in fileName):
                    fileNameList.append(fileName)
        
        for fileName in fileNameList:
            #* load alpha npy file
            filePath = os.path.join(folder["alpha_pdart_nodrop"], fileName)
            alphaPerLayer = np.load(filePath)
            key = fileName.split(".")[0]
            key = key.split("th_")[1] # eg:layer0_1
            decodeDict[key] = decodeOperation(alphaPerLayer)
        
        #info create complete decode Dict
        toSaveDict = copy.deepcopy(emptyArch) # it need to be deep copied
        for layerName in decodeDict:
            toSaveDict[layerName] = decodeDict[layerName]
        # print("toSaveDict", toSaveDict)

    return toSaveDict

def getCompareLayerList(fileNameList, basedLayerNo):
    compareLayerList = []
    for fileName in fileNameList:
        if "_{}.npy".format(str(basedLayerNo)) in fileName:
            compareLayerList.append(fileName)
    return compareLayerList
def decodeLayer(compareLayerList):
    finalBetaDict = {}
    #info get last beta in compareLayerList and make it a finalBetaDict
    for fileName in compareLayerList:
        filePath = os.path.join(folder["betaLog"], fileName)
        layerName = filePath.split(".")[-2]
        finalBetaDict[layerName] = np.load(filePath)[-1]
    #info find largest beta in each finalBetaDict
    sort_orders = sorted(finalBetaDict.items(), key=lambda x: x[1], reverse=True) #*sort dict by value
    return [sort_orders[0][0]]

    
# todo topological sort to decode beta
def decodeAllLayer(kth):
    fileNameList = []
    
    #info split file according to different kth
    for fileName in sorted(listdir(folder["betaLog"])):
        if str(kth)+"th" in fileName:
            fileNameList.append(fileName)
    #info back trace
    startFromLayer = 5
    pickerLayerList = []
    for i in range(startFromLayer, -1, -1):
        # print("startFromLayer", startFromLayer)
        compareLayerList = getCompareLayerList(fileNameList, startFromLayer)
        pickedLayerList = decodeLayer(compareLayerList)
        toLayerNo = pickedLayerList[0].split("_")[-2]
        startFromLayer = toLayerNo
        pickerLayerList.append(pickedLayerList[0])
        if toLayerNo=="0":
            #info reach First layer
            break
    return pickerLayerList
            
if __name__ == '__main__': 
    # filePath = "./decode/decode.json"
    # setStdoutToFile(filePath)
    # decodeAllInnerCell()
    # f = open("./decode/0th_decode.json")
    # # returns JSON object as 
    # # a dictionary
    # data = json.load(f)
    # print(data)
    # for key in data:
    #     print(key, data[key])
    # exit()
    for kth in range(0, 3):
        pickerLayerList = decodeAllLayer(kth)
        print("pickerLayerList", pickerLayerList)
        toSaveDict = decodeAllOperation(kth, pickerLayerList)
        printToDecodeFile(toSaveDict, kth)
        # break
    # for kth in range(3):
        # print(decodeAlphas(kth))
        # print(manualAssign(kth))

    exit()
