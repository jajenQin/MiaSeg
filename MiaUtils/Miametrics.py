from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from matplotlib import pyplot as plt
import numpy as np
import skimage
import skimage.io
from skimage.measure import regionprops
from scipy import ndimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.segmentation import slic
#from medpy.metric.binary import hd,asd,recall,obj_tpr,obj_fpr
import logging
import cv2
import vtk
import SimpleITK as itk
class MiaMetrics(object):
    """
    This class for text or images or volume data writer into a specific file
    """
    def __init__(self,logger):
        self.logger=logger
    def SPRegionArribute(self,superpixel,img,label):
        """
        superpixel:The superpixel extracted from image data
        img: orginal image data for statistics
        :return:R
         SPAllmean:mean of deviation of the different regions labeled by the superpixels in index.
         SPdevAll:deviation of deviation of the different regions labeled by the superpixels in index.
        """

        region = regionprops(superpixel,img)#get the all regions information
       # Objectsuper=np.zeros(superpixel.shape)
        #Objectsuper[label!=0]=superpixel[label!=0]#only object region will be caculated to see the deviation
        Objectsuper=superpixel
        splabel=np.unique(Objectsuper[Objectsuper!=0])#removed the repeatabel number and get unique label
        # print('The superpixels label:')
        # print(splabel)
        # SPdeviation=ndimage.standard_deviation(img, Objectsuper)
        # print(SPdeviation)
        SPdeviation_each=ndimage.standard_deviation(img, Objectsuper, index=splabel)

        #print(SPdeviation_each)

        SPdevAll=ndimage.standard_deviation(SPdeviation_each)
        #print('The deviation at whole image superpixels region:%f'%SPdevAll)

        SPAllmean=ndimage.mean(SPdeviation_each)
        #print('The mean deviationat whole image superpixels region:%f'%SPAllmean)
        return SPdevAll,SPAllmean

    def SPSelecttest(self,slice_colors,PatchNorm,label,numsuperpixels):

       SPDev=[]
       SPMean=[]
       spcount=0
       for nsp in numsuperpixels:
           superpixel = slic(slice_colors, n_segments=nsp, compactness=5, sigma=1.0)#superpixel caculation wiht slic
           spdev,spmean=self.SPRegionArribute(superpixel,PatchNorm,label)
           SPDev.append(spdev)
           SPMean.append(spmean)
           slice_entro = entropy(PatchNorm, disk(5))#get the whole image entropy
           region = regionprops(superpixel,slice_entro)#get the all regions information
           spcount+=1
           print('The Grounp No.%d/%d superpixel have been generated'%(spcount,len(numsuperpixels)))

      #******************************************************************************#
       #display the superpixel results
      #*******************************************************************************#
           # #
           fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
                                    subplot_kw={'adjustable': 'box-forced'})
           ax = axes.ravel()
           ax[0].imshow(slice_colors)
           from skimage.segmentation import mark_boundaries
           ax[1].imshow(mark_boundaries(slice_colors, superpixel),cmap='jet')
           ax[1].set_title('The %d superpixels'%nsp)
           ax[1].imshow(label, cmap='hot', alpha=0.0)
           ax[2].imshow(slice_entro)
           ax[3].imshow(superpixel)

       return SPDev,SPMean
      # plt.figure()
      # plt.plot(range(100,2000,50),np.array(SPDev))
    def DSCMetric(self,pred,gt):
        """
        compute the dice similarity coefficient dice(A,B)=2|A ∩)B|/|A+B|
        pred:the ndarray prediction from network,datatype must be bool
        gt:ndaray Ground truth of data,datatype must be bool
        :return: dice coefficient
        """
        eps = 1e-5
        #intersection = np.sum(pred[gt.nonzero()])
        intersection=np.sum(np.bitwise_and(pred,gt))
        union =  eps + np.sum(pred) + np.sum(gt)
        dsc = 2 * intersection/ (union)
        return dsc
    def VOEMetric(self,pred,gt):
        """
        compute the volume overlap error eg.Jaccard coefficient
        VOE(A,b)=1-|A∩B|/|A∪B|
        :param pred:the ndarray prediction from network,datatype must be bool
        :param:gt:ndaray Ground truth of data,datatype must be bool

        :return: VOE
        """
        intersection=np.bitwise_and(pred,gt)
        union=np.bitwise_or(pred,gt)
        voe=1-np.sum(intersection)/np.sum(union)
        return voe

    def RVDMetric(self,pred,gt):
        """
        compute the asymmetic metric
        RVD(A,B)=(|B|-|A|)/|A|
        :param pred:the ndarray prediction from network,datatype must be bool
        :param:gt:ndaray Ground truth of data,datatype must be bool
        :return: RVD
        """
        rvd=(np.sum(pred)-np.sum(gt))/np.sum(gt)
        return rvd
    def ASDMetric(self,pred,gt):
        """
        compute the avarage symmetric surface distance metric
        S(A) :the set of surface voxels of A
        d(v,S(A))=min||v-SA|| the shortest distance of an arbitrary voxel vo to S(A)
        ASD(A,B)=1/(S(A)+S(B))(∑d(SA,S(B))+∑d(SB,S(A))
        :param pred:the ndarray prediction from network,datatype must be bool
        :param:gt:ndaray Ground truth of data,datatype must be bool
        :return: RVD
        """
        asd=(np.sum(pred)-np.sum(gt))/np.sum(gt)
        return asd

    def QuantativePlot(self,QRlist, outputpath, filename):
        """
        This function is for ROC and PR curve result plot for kfold validation
        :param QRlist: dict type with  a list Quantative result
        :return:
        """
        tprs = []
        aucs = []
        prs = []
        rcs = []
        aps = []
        mean_fpr = np.linspace(0, 1, QRlist[0]["gtlabel"].shape[0])
        i = 0
        fig1 = plt.figure(figsize=(9, 5))
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure(figsize=(9, 5))
        ax2 = fig2.add_subplot(111)

        for QuatativeResult in QRlist:
            gtlabel = QuatativeResult["gtlabel"]
            prediction = QuatativeResult["predlabel"]
            predproblist = QuatativeResult["premap"]
            # Acc = QuatativeResult["Accuracy"]
            # precision = QuatativeResult["Precision"]
            from sklearn.metrics import roc_curve, auc
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(gtlabel, predproblist[:, 1])
            from sklearn.metrics import average_precision_score
            average_precision = average_precision_score(gtlabel, predproblist[:, 1])

            prs.append(precision)
            aps.append(average_precision)
            rcs.append(recall)

            fpr, tpr, thresholds = roc_curve(gtlabel, predproblist[:, 1])
            from scipy import interp
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来

            ax1.plot(fpr, tpr, lw=1, label='ROC fold {0} (area ={1:0.2f})'''.format(i, roc_auc),
                     color='deeppink', linestyle=':', linewidth=4)


            ax2.step(recall, precision, alpha=0.2, where='post',
                     label='PR fold {0}(AP={1})'''.format(i, average_precision),
                     color='navy', linestyle=':', linewidth=4)
            # ax2.fill_between(recall, precision, step='post', alpha=0.2,color='b')

            # 画对角线
        ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0  #

        mean_auc = auc(mean_fpr, mean_tpr)  # AUC mean calculation
        std_auc = np.std(aucs)

        ax1.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=4, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver operating characteristic')
        ax1.legend(loc="lower right")

        mean_presicion = np.mean(prs, axis=0)
        mean_recall = np.mean(rcs, axis=0)
        mean_ap = np.mean(aps)
        ax2.step(mean_presicion, mean_recall, color='y', alpha=0.2, where='post',
                 label=r'Mean PR (Mean AP= {0}'''.format(mean_ap), linewidth=4)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlim([0.0, 1.0])
        ax2.set_title('Precision-Recall curve')
        ax2.legend(loc="lower right")
        fig1.savefig(os.path.join(outputpath, 'roc' + filename), facecolor='black')
        fig2.savefig(os.path.join(outputpath, 'pr' + filename), facecolor='black')
        plt.show()

    def helper(self,b1, b2, h, w, thres):
        cnt = 0
        for i in range(h):
            for j in range(w):
                if b1[i][j]:
                    lower_x = max(0, i - thres)
                    upper_x = min(h - 1, i + thres)
                    lower_y = max(0, j - thres)
                    upper_y = min(w - 1, j + thres)
                    matrix_rows = b2[lower_x: upper_x + 1, :]
                    matrix = matrix_rows[:, lower_y: upper_y + 1]
                    if matrix.sum() > 0:
                        cnt = cnt + 1
        total = b1.sum()
        print(cnt)
        print(total)
        return cnt / total


    def eval_bound(self,segmask1, gtmask2, thres):
        '''Evaluate precision for boundary detection'''
        s1 = segmask1.shape
        s2 = gtmask2.shape

        if s1 != s2:
            print('shape not match')
            return -1, -1
        if len(s1) == 3:
            b1 = segmask1.reshape(s1[0], s1[1]) == 0
            b2 = gtmask2.reshape(s2[0], s2[1]) == 0
        else:
            b1 = segmask1 == 0
            b2 = gtmask2 == 0

        h = s1[0]
        w = s1[1]
        precision = self.helper(b1, b2, h, w, thres)
        recall = self.helper(b2, b1, h, w, thres)
        return precision, recall

    def find_bound(self,label):
        height = label.shape[0]
        width = label.shape[1]
        ret = np.zeros([height,width, 1], dtype=bool)
        div = label.reshape([height,width,1])
        df0 = np.diff(div,axis=0)
        df1 = np.diff(div,axis=1)
        mask0 = df0 != 0
        mask1 = df1 != 0
        ret[0:height - 1, :, :] = np.logical_or(ret[0:height - 1, :, :], mask0)
        ret[1:height, :, :] = np.logical_or(ret[1:height, :,:], mask0)
        ret[:,  0:width-1, :] = np.logical_or(ret[:, 0:width-1,:],mask1)
        ret[:, 1:width, :] = np.logical_or(ret[:, 1:width,:], mask1)

        ret2 = np.ones([height,width,1], dtype="uint8")
        ret2 = ret2*255 - ret * 255
        # plt.imshow(ret)
        # plt.show()
        return ret2

