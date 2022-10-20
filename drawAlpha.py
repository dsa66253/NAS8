import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from data.config import PRIMITIVES, folder
from os import listdir
from data.config import cfg_nasmodel as cfg

numOfEpoch=cfg["epoch"]
# numOfEpoch = 25
kth=0
alphaFolder = "alpha_pdart_nodrop"
desFolder = "plot"
def loadAllAlphas(kth=0):
    listAlphas = []
    for epoch in range(numOfEpoch):
        tmp = np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(kth, epoch))
        listAlphas.append(tmp)
    return listAlphas


#* create x axis labels 0, 5
#* labels also means build rectangle at which epoch is
def plot(kth, atLayer=0, atInnerCell=0):
    #info prepare needed data
    #* name:alphas_layer_innercell_op
    allAlphas = loadAllAlphas(kth)
    opList = ["alphas{}_{}_3".format(atLayer, atInnerCell),
            "alphas{}_{}_5".format(atLayer, atInnerCell),
            "alphas{}_{}_7".format(atLayer, atInnerCell),
            "alphas{}_{}_9".format(atLayer, atInnerCell),
            "alphas{}_{}_11".format(atLayer, atInnerCell),
            "alphas{}_{}_skip".format(atLayer, atInnerCell)
            ]
    opDic = {}
    for op in opList:
        opDic[op] = []
    for epoch in range(len(allAlphas)):
        for op in range(len(opList)):
            # print("handle epoch{} atinnercell{} op{}".format(epoch, atInnerCell, op))
            # print("allAlphas[epoch]", allAlphas)
            # print("allAlphas[epoch]", allAlphas[epoch])
            # print("allAlphas[epoch][atLayer]", allAlphas[epoch][atLayer])
            # print("allAlphas[epoch][atLayer][atInnerCell]", allAlphas[epoch][atLayer][atInnerCell])
            opDic[opList[op]].append( allAlphas[epoch][atLayer][atInnerCell][op] )

            
            
    #info put data into figure
    fig, ax = plt.subplots(figsize=(20,4))
    width = 0.13
    x = np.arange(len(allAlphas))  # the label locations
    for i in range(len(opList)):
        ax.bar(x + i*width, opDic[opList[i]], width, label=opList[i], align='edge')
    
    
    ax.set_ylabel('alphas probability')
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel('epoch')
    ax.set_xticks(range(0, numOfEpoch, 1))
    ax.set_title('layer{} innercell{}'.format(atLayer, atInnerCell))
    # ax.set_xticks(x, labels)
    ax.legend(loc=2)
    # ax.grid(True);
    plt.savefig('./plot/' + '{}th_layer{}_innercell{}'.format(kth, atLayer, atInnerCell) + '.png')


    # plt.savefig('./weights_pdarts_nodrop/' + 'layer' + str(currentLayer) + '.png')

def loadAlphasAtEpoch(kth, epoch):
    return  np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(str(kth), str(epoch)))
    
def drawBar():
    allAlphas = loadAllAlphas(kth)

    numOfEpoch = len(allAlphas)
    numOfLayer = len(allAlphas[0])
    numOfInnerCell = len(allAlphas[0][0])
    numOfOp = len(allAlphas[0][0][0])

    print("alphas architecture:numOfEpoch {}, numOfLayer {}, numOfInnerCell {}, numOfOp {}".format(numOfEpoch, numOfLayer, numOfInnerCell, numOfOp))
    for kth in range(3):
        for layer in range(numOfLayer):
            for innerCell in range(numOfInnerCell):
                plot(kth, layer, atInnerCell=innerCell)
def plot_line_chart_layer_innercell(kth, atLayer=0, atInnerCell=0):
    #info prepare needed data
    print("draw {}th layer{} innercell{}".format(kth, atLayer, atInnerCell))
    #* name:alphas_layer_innercell_op
    allAlphas = np.load("./alpha_pdart_nodrop/allAlphas_{}.npy".format(kth))
    opList = []
    for i in range(len(PRIMITIVES)):
        opList.append(PRIMITIVES[i]+"_{}_{}".format(atLayer, atInnerCell))
    opDic = {}
    for op in opList:
        opDic[op] = []
    for iteraion in range(len(allAlphas)):
        for op in range(len(opList)):
            opDic[opList[op]].append( allAlphas[iteraion][atLayer][atInnerCell][op] )
            
            
    #info put data into figure
    numOfIteration = len(allAlphas)
    numOfRow = 3
    numOfCol = 1
    fig, axs = plt.subplots(numOfRow, numOfCol, figsize=(20, 6), constrained_layout=True)
    x = np.arange(len(allAlphas))  # the label locations
    for i in range(len(opList)):
        # ax.bar(x + i*width, opDic[opList[i]], width, label=opList[i], align='edge')
        splits = np.array_split(opDic[opList[i]], numOfRow)
        for row in range(len(splits)):
            axs[row].plot(splits[row], label=opList[i])
            
            axs[row].set_ylabel('alphas probability')
            # axs[row].set_yticks([0.1, 0.2, 0.3, 0.4])
            # axs[row].set_yticks(np.arange(0, 0.01, 0.001))
            axs[row].set_title('layer{} innercell{}'.format(atLayer, atInnerCell))
            
            #info data need to transfer iteration to epoch
            baseTick = row * numOfIteration//len(splits)
            numOfTick = numOfIteration//len(splits)
            iterPerEpoch = numOfIteration//numOfEpoch
            axs[row].set_xticks( range( 0, numOfTick, iterPerEpoch) )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+numOfTick, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xlabel('epoch')
            axs[row].legend(loc=2)
    # plt.show()
    plt.savefig('./plot/' + 'lineChart_{}th_layer{}_innercell{}'.format(kth, atLayer, atInnerCell) + '.png')
def plot_line_chart_all(kth):
    
    allAlphas = np.load("./alpha_pdart_nodrop/allAlphas_{}.npy".format(kth))
    numOfIteration = len(allAlphas)
    numOfLayer = len(allAlphas[0])
    numOfInnerCell = len(allAlphas[0][0])
    numOfOp = len(allAlphas[0][0][0])
    print("alphas architecture:numOfIteration {}, numOfLayer {}, numOfInnerCell {}, numOfOp {}".format(numOfIteration, numOfLayer, numOfInnerCell, numOfOp))
    for layer in range(numOfLayer):
        for innerCell in range(numOfInnerCell):
            plot_line_chart_layer_innercell(kth, layer, atInnerCell=innerCell)
def plot_line_chart_all_file():
    fileNameList = []

    for fileName in sorted(listdir(folder["alpha_pdart_nodrop"])):
        #* load alpha npy file
        filePath = os.path.join(folder["alpha_pdart_nodrop"], fileName)
        fileNameList.append(filePath)
        allAlphas = np.load(filePath)
        plot_line_chart_innercell(allAlphas, fileName=fileName.split(".")[0])
        # allAlphas = [[0.2, 0.2, 0.2, 0.2, 0.2]
        #         ,[0.2, 0.2, 0.2, 0.2, 0.2]]

        # plot_line_char_innercell(allAlphas, fileName)
        # break

    # allAlphas = [[0.5, 0.2, 0.2, 0.3, 0.1]
    #             ,[0.2, 0.2, 0.2, 0.2, 0.2]]
    # tmp = np.array(allAlphas)
    # for i in range(100):
    #     tmp = np.append(tmp, allAlphas, axis=0)
    # allAlphas = np.array(tmp)

    # plot_line_char_innercell(allAlphas, fileName="tmp")

    

def plot_line_chart_innercell(allAlphas, fileName=""):
    alphaDict = {}
    for key in PRIMITIVES:
        alphaDict[key] = []
    for index in range(len(PRIMITIVES)):
        for iteration in range(len(allAlphas)):
            alphaDict[PRIMITIVES[index]].append( allAlphas[iteration][index] )
    # allAlphas = [[0.2 0.2 0.2 0.2 0.2]
    #         ,[0.2 0.2 0.2 0.2 0.2]]
    numOfIteration = len(allAlphas)
    numOfRow = 3
    numOfCol = 1
    fig, axs = plt.subplots(numOfRow, numOfCol, figsize=(20, 6), constrained_layout=True)
    x = np.arange(len(allAlphas))  # the label locations
    totalTickRow = numOfIteration // numOfRow
    totalEpochRow = numOfEpoch // numOfRow
    iterPerEpoch = numOfIteration//numOfEpoch
    for i in range(len(PRIMITIVES)):
        #* print each conv line
        # splits = np.array_split(alphaDict[PRIMITIVES[i]], numOfRow)
        # #! split didn't match tick labels
        # axs[0].plot(alphaDict[PRIMITIVES[i]], label=PRIMITIVES[i])
        for row in range(numOfRow):

            splits = alphaDict[PRIMITIVES[i]][row*totalEpochRow*iterPerEpoch:(row+1)*totalEpochRow*iterPerEpoch]
            axs[row].plot(splits, label=PRIMITIVES[i])
            
            axs[row].set_ylabel('alphas value')
            # axs[row].set_yticks([0.1, 0.2, 0.3, 0.4])
            # axs[row].set_yticks(np.arange(0, 0.01, 0.001))
            axs[row].set_title(fileName)
            
            #info data need to transfer iteration to epoch

            # print("numOfIteration", numOfIteration)
            # print("numOfEpoch", numOfEpoch)
            # print("iterPerEpoch", iterPerEpoch)
            baseTick = row * numOfIteration//numOfRow
            axs[row].set_xticks( range( 0, totalTickRow, iterPerEpoch) )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+totalTickRow, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+totalTickRow, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xlabel('epoch')
            axs[row].legend(loc=2)
    # plt.show()
    print("save to ", os.path.join(desFolder, fileName)+ '.png')
    plt.savefig(os.path.join(desFolder, fileName)+ '.png')
    plt.close()
def getBetaDict():
    #info load all beta files
    fileNameList = []
    for fileName in sorted(listdir(folder["betaLog"])):
        if "beta." in fileName:
            # beta's file name starts with beta.
            fileNameList.append(fileName)
    #info create betaDict for each layer per kth
    betaDict = {}
    for fileName in fileNameList:
        #* load alpha npy file
        filePath = os.path.join(folder["betaLog"], fileName)
        # print("filePath", filePath)
        beta = np.load(filePath)
        layerName = fileName.split(".")[1] + "." + fileName.split(".")[2]
        betaDict[layerName] = beta
    return betaDict
def plot_all_beta_line_chart():
    betaDict = getBetaDict()
    #info compare what all beta in one figure
    for kth in range(3):
        toPrintBetaDict = {}
        for layerName in betaDict:
            if str(kth)+"th" in layerName:
                toPrintBetaDict[layerName] = betaDict[layerName]
        plot_beta_line_chart(toPrintBetaDict, fileName="beta.{}th".format(str(kth)))
        
        
def plot_beta_line_chart(toPrintBetaDict, fileName):
    # toPrintBetaDict: print this dict in one file; key is label
    # this figure will save as fileName
    numOfLayer = len(toPrintBetaDict)
    betaDict = {}
    betaDict[fileName] = toPrintBetaDict
    numOfIteration = 0
    for layerName in toPrintBetaDict:
        numOfIteration = len(toPrintBetaDict[layerName])
    numOfRow = 3
    numOfCol = 1
    
    x = np.arange(len(toPrintBetaDict))  # the label locations
    totalTickRow = numOfIteration // numOfRow
    totalEpochRow = numOfEpoch // numOfRow
    iterPerEpoch = numOfIteration//numOfEpoch
    # print("iterPerEpoch", iterPerEpoch)
    fig, axs = plt.subplots(numOfRow, numOfCol, figsize=(20, 8), constrained_layout=True)
    # print(np.linspace(0, 1, numOfLayer))

    
    for row in range(numOfRow):
        cm = plt.get_cmap('gist_rainbow')
        # axs[row].set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
        for layerName in toPrintBetaDict:

            splits = toPrintBetaDict[layerName][row*totalEpochRow*iterPerEpoch:(row+1)*totalEpochRow*iterPerEpoch]
            axs[row].plot(splits, label=layerName)
            #info set legend and label
            axs[row].set_ylabel('beta value')
            axs[row].set_title(fileName)
            baseTick = row * numOfIteration//numOfRow
            axs[row].set_xticks( range( 0, totalTickRow, iterPerEpoch) )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+totalTickRow, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+totalTickRow, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xlabel('epoch')
    axs[2].legend(loc='lower center', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.75))
    print("save to ", os.path.join(desFolder, fileName)+ '.png')
    plt.savefig(os.path.join(desFolder, fileName)+ '.png')
    plt.close()
def plot_betaGrad_line_chart():
    #info load all beta files
    fileNameList = []
    for fileName in sorted(listdir(folder["betaLog"])):
        if "betaGrad." in fileName:
            # beta's file name starts with beta.
            fileNameList.append(fileName)
    #info create betaDict for each layer per kth
    betaGradDict = {}
    for fileName in fileNameList:
        #* load alpha npy file
        filePath = os.path.join(folder["betaLog"], fileName)
        # print("filePath", filePath)
        beta = np.load(filePath)
        layerName = fileName.split(".")[1] + "_" + fileName.split(".")[2]
        betaGradDict[layerName] = beta
    # for k in betaGradDict:
    #     print(k, betaGradDict[k])
    #     for i in range(len(betaGradDict[k])):
            
    #         if i>800 and i<1200:
    #             print(i, betaGradDict[k][i])
    for kth in range(3):
        toPrintBetaDict = {}
        for layerName in betaGradDict:
            if str(kth)+"th" in layerName:
                toPrintBetaDict[layerName] = betaGradDict[layerName]
        for k in toPrintBetaDict:
            print(k)
        plot_beta_line_chart(toPrintBetaDict, fileName="betaGrad.{}th".format(str(kth)))
def plot_beta_line_chart_by_same_dest():
    betaDict = getBetaDict()
    for kth in range(0, 3):
        #info get kth dict
        kthBetaDict = {}
        for layerName in betaDict:
            if str(kth)+"th" in layerName:
                kthBetaDict[layerName] = betaDict[layerName]
        # print("dest layer: ", desLayerIndex)
        # print("======================")
        # for k in kthBetaDict:
        #     print(k)
        #info get toPrintDict by destination layer
        for desLayerIndex in range(1, 6):
            toPrintDict = {}
            
            for layerName in kthBetaDict:
                if layerName.split("_")[-1] == str(desLayerIndex):
                    toPrintDict[layerName] = kthBetaDict[layerName]

            plot_beta_line_chart(toPrintDict, fileName="{}th.betaByDest{}".format(str(kth), str(desLayerIndex)))
                
                
if __name__=="__main__":
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    plot_beta_line_chart_by_same_dest()
    plot_line_chart_all_file()
    # plot_betaGrad_line_chart()
    # plot_all_beta_line_chart()
    # x = np.linspace(0, 1, 10)
    # fig, ax = plt.subplots()
    # ax.set_prop_cycle(color=['red', 'green', 'blue'])
    # for i in range(1, 6):
    #     plt.plot(x, i * x + i, label='$y = {i}x + {i}$'.format(i=i))
    # plt.legend(loc='best')

    # fig.savefig('./moreColors.png')
    # plt.savefig(os.path.join(desFolder, fileName)+ '.png')
    # plt.close()
    # NUM_COLORS = 20

    # cm = plt.get_cmap('gist_rainbow')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    # for i in range(NUM_COLORS):
    #     ax.plot(np.arange(10)*(i+1))

    # fig.savefig('moreColors.png')

    # for kth in range(3):
    #     plot_line_chart_all(kth)


    