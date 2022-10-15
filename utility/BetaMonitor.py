import torch
from data.config import folder, cfg_nasmodel
import numpy as np
import os
import numpy as np
class BetaMonitor():
    def __init__(self):
        self.allBeta = 0
        # self.allAlphasGrad = 0
        self.betaLogDict = None
        self.betaGradLogDict = None
    def logBetaDictPerIter(self, net, iteration):
        betaDict = net.getBetaDict()
        if self.betaLogDict==None:
            self.betaLogDict = {}
            for layerName in betaDict:
                self.betaLogDict[layerName] = torch.FloatTensor(betaDict[layerName]).numpy()
        else:
            for layerName in betaDict:
                self.betaLogDict[layerName] = np.append(self.betaLogDict[layerName], torch.FloatTensor(betaDict[layerName]).numpy(), axis=0)
    def logBetaGradDictPerIter(self, net, iteration, epoch):
        betaDict = net.getBetaDict()
        # print("logBetaGradDictPerIter")
        # print("betaDict", betaDict)
        if self.betaGradLogDict==None:
            self.betaGradLogDict = {}
            for layerName in betaDict:
                if betaDict[layerName][0][0].grad==None:
                    self.betaGradLogDict[layerName] = torch.FloatTensor([0.0]).numpy()
                else:
                    # print(betaDict[layerName])
                    # print("betaDict[layerName][0][0].grad", betaDict[layerName][0].grad)
                    self.betaGradLogDict[layerName] = torch.FloatTensor(betaDict[layerName][0].grad).numpy()
                    # print("self.betaLogDict[layerName]", self.betaLogDict[layerName])
        else:
            for layerName in betaDict:
                # if betaDict[layerName][0][0].grad==None:
                if epoch >= cfg_nasmodel["start_train_nas_epoch"]:
                    grad = betaDict[layerName][0].grad.detach().clone().cpu()
                else:
                    grad = torch.FloatTensor([0.0])
                # print("epoch: {}, layerName:{}, grad: {}".format(str(epoch), layerName, str(grad)))
                self.betaGradLogDict[layerName] = np.append(self.betaGradLogDict[layerName], grad.numpy(), axis=0)
                    


    def logAlphasPerIteration(self, net, iteration):
        # not used 
        if iteration==0:
            tmp = net.getAlphasTensor()
            self.allAlphas = tmp.reshape((1, *tmp.size()))
        else:
            tmp = net.getAlphasTensor()
            self.allAlphas = torch.cat((self.allAlphas, tmp.reshape((1, *tmp.size()))))
        print("self.allAlphas", self.allAlphas)
    def saveAllBeta(self, kth):
        for layerName in self.betaLogDict:
            try:
                np.save(os.path.join( folder["betaLog"], "beta.{}th.{}".format(kth, layerName) ), self.betaLogDict[layerName])
                #todo change to np.save(os.path.join( folder["betaLog"], "beta.{}th_{}".format(kth, layerName) ), self.betaLogDict[layerName])
            except Exception as e:
                print("cannot save betaPerIteration", e)
    def saveAllBetaGrad(self, kth):
        for layerName in self.betaGradLogDict:
            try:
                np.save(os.path.join( folder["betaLog"], "betaGrad.{}th.{}".format(kth, layerName) ), self.betaGradLogDict[layerName])
                #todo change to np.save(os.path.join( folder["betaLog"], "beta.{}th_{}".format(kth, layerName) ), self.betaLogDict[layerName])
            except Exception as e:
                print("cannot save betaPerIteration", e)
                exit()
    def logAlphasGradPerIteration(self, net, iteration):
        if iteration==0:
            
            tmp = net.alphas.grad
            print("tmp = net.alphas.grad", net.alphas.grad)
            self.allAlphasGrad = tmp.reshape((1, *tmp.size()))
        else:
            tmp = net.alphas.grad
            self.allAlphasGrad = torch.cat((self.allAlphasGrad, tmp.reshape((1, *tmp.size()))))
    def saveAllAlphasGrad(self, kth):
        try:
            np.save(os.path.join( folder["alpha_pdart_nodrop"], "allAlphasGrad_{}".format(kth) ), self.allAlphasGrad.cpu().detach().numpy())
        except Exception as e:
            print("cannot save alphasGradPerIteration", e)
            