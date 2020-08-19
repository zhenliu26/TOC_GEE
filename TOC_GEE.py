import ee
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib import colors as matpltColors

ee.Initialize()


def TOC_Image_coor(input_binary: ee.Image, input_index: ee.Image, thresholdList: ee.List, noDataValue: int, exportCoor: str = False,
                   exportVariable: str = False) -> ee.List:
    """
    This function is to export the coordinates of the TOC curve. The input format is the ee.Image.
    :param input_binary: (ee.Image) Reference image. presence is 1, absence is 0
    :param input_index: (ee.Image) Index image.
    :param thresholdList: (ee.List/list) The list of thresholds. (from high to low or from low to high)
    :param noDataValue: (number) always be -1
    :param exportCoor: (string / default is False) The default parameter is False (not export the coordinates), string should be output path of coordinates. (extension should be txt)
    :param exportVariable: (string / default is False) The default parameter is False (not export the coordinates), string should be output of variables. (extension should be txt)
    :return: (ee.list) coordinates of the TOC curve. The first column is x, the second is y, the third is threshold of the point.
    """
    # calculate the mask to map out the nodata value
    mask = maskImage(input_binary, noDataValue);
    # calculate the binary image (mapping out noDataValue)
    mask_Binary = mask.multiply(input_binary);
    # calculate the potential presence
    presenceInY = sumImage(mask_Binary)
    # calculate the total pixels in the study area
    totalNumber = sumImage(mask)
    # get the sequence of threshold: descending is 0; ascending is 1.
    sequence = ee.Algorithms.If(ee.Number(thresholdList.get(0)).lt(thresholdList.get(-1)), 1, 0);
    def coordinatelist(x):
        index1 = compareIndex(input_index, x, sequence).multiply(mask);
        index2 = ee.Image((index1.add(mask_Binary).gte(2)));
        X_value = sumImage(index1);
        Y_value = sumImage(index2);
        x_1 = ee.Number(X_value);
        y_1 = ee.Number(Y_value);
        coor = ee.List([x_1, y_1,ee.Number(x)]);
        return coor;
    result = thresholdList.map(coordinatelist)
    result_output = result
    if(exportCoor):
        result = np.array(result.getInfo()).transpose()
        exportCoorFunc(exportCoor, np.array([result[0].tolist()]), np.array([result[1].tolist()]), result[2].tolist())
    if(exportVariable):
        pInY = presenceInY.getInfo()
        tNumber = totalNumber.getInfo()
        AUC = calculate_AUC(np.array([result[0].tolist()]), np.array([result[1].tolist()]),pInY,tNumber)
        ccorner = correctCorner(np.array([result[0].tolist()]), np.array([result[1].tolist()]),pInY)
        exportVariableFunc(exportVariable, tNumber, pInY, ccorner, AUC)
    return result_output
def TOC_Image(img: ee.Image, QCbandname: str, IndexbandnameList: ee.List, thresholdList: ee.List, noDataValue: int, nameList: list,
              boolcorrectcorner: bool = False,
              booluniformline: bool = False, unit: str = 'observations') -> None:
    '''
    This is the function to show the TOC curve.
    :param img: (ee.Image) the image which contains the reference band and index bands.
    :param QCbandname: (String) the band name of QC band
    :param IndexbandnameList: (ee.List/list) the list of index band names
    :param thresholdList: (ee.List/list) the list of thresholds, format like [[],[],[],[],[]] (the same order as the name list)
    :param noDataValue: (number) always be -1
    :param nameList: (list) The list of band names
    :param boolcorrectcorner: (bool) whether to show the correct corners on the diagram
    :param booluniformline: (bool) whether to show uniform line on the diagram
    :param unit: (str) the unit of each record
    :return: no return
    '''
    # change the format to ee format
    if (type(IndexbandnameList) is not ee.List):
        IndexbandnameList = ee.List(IndexbandnameList)
    if (type(QCbandname) is not str):
        QCbandname = QCbandname.getInfo()
    if (type(thresholdList) is not ee.List):
        thresholdList = ee.List(thresholdList)
    # calculate the mask to map out the nodata value
    mask = maskImage(img.select(QCbandname), noDataValue);
    # calculate the binary image (mapping out noDataValue)
    mask_Binary = mask.multiply(img.select(QCbandname));
    # calculate the potential presence
    presenceInY = sumImage(mask_Binary)
    # calculate the total pixels in the study area
    totalNumber = sumImage(mask)
    def coordinateList(x):
        x = ee.String(x)
        band1 = img.select(x);
        num = IndexbandnameList.indexOf(x)
        QC = img.select(QCbandname)
        threshold = ee.List(thresholdList.get(num))
        result1 = TOC_Image_coor(QC, band1, threshold, noDataValue)
        return result1
    coordinates = IndexbandnameList.map(coordinateList)
    coordinates = coordinates.getInfo()
    Xlist=[]
    Ylist=[]
    for item in coordinates:
        xyList = np.array(item).transpose()
        Xlist.append(xyList[0])
        Ylist.append(xyList[1])
        # Xlist.insert(-1, xyList[0])
        # Ylist.insert(-1, xyList[1])
    Xlist = np.array(Xlist)
    Ylist = np.array(Ylist)
    curveNum = len(Xlist)

    painter = painter_Generator(Xlist, Ylist, unit)
    painter.paintInit(boolUniform=booluniformline, boolCorrectcorner=boolcorrectcorner)
    for i in range(len(nameList)):
        if (boolcorrectcorner):
            painter.correctCorner(i)
        painter.paintOne(i, nameList[i], '^')
    painter.show()
def TOC_Feature_coor(featurecollection_input: ee.FeatureCollection, QCname: str, Indexname: str, thresholdList: ee.List, Classname: str = None,
                     ClassList: dict = None,
                     exportCoor: str = False,
                     exportVariable: str = False) -> None:
    '''
    This function is to export the coordinates of the TOC curve. The input format is the ee.FeatureCollection.
    :param featurecollection_input: (ee.FeatureCollection) featurecollection that contains reference and index properties
    :param QCname: (string) The name of reference property
    :param Indexname: (string) The name of index property
    :param thresholdList: (ee.List/list) The list of thresholds. (from high to low or from low to high)
    :param Classname: (string) The name of class property when applying stratified sampling
    :param ClassList: (disctionary) The size of stratums {classname: classsize, classname: classsize ...}
    :param exportCoor: (string / default is False) The default parameter is False (not export the coordinates), string should be output path of coordinates. (extension should be txt)
    :param exportVariable: (string / default is False) The default parameter is False (not export the coordinates), string should be output of variables. (extension should be txt)
    :return: (ee.list) coordinates of the TOC curve. The first column is x, the second is y, the third is threshold of the point.
    '''

    featurecollection_list = featurecollection_input.toList(featurecollection_input.size())
    def getPropertyIndex(x):
        return ee.Feature(x).getNumber(Indexname)
    def getPropertyQC(x):
        return ee.Feature(x).getNumber(QCname)

    Index_1 = featurecollection_list.map(getPropertyIndex)
    Index_2 = ee.Array(Index_1)
    QC_1 = featurecollection_list.map(getPropertyQC)
    QC_2 = ee.Array(QC_1)

    if((Classname is not None) and (ClassList is not None)):
        if(type(Classname) is not str):
            Classname = Classname.getInfo()
        if(type(ClassList) is not ee.Dictionary):
            ClassList = ee.Dictionary(ClassList)
        classProperty = featurecollection_input.aggregate_histogram(Classname)
        def classWeight(key,value):
            presenceInClass = classProperty.get(key)
            return ee.Number(value).divide(presenceInClass)

        ClassWeights = ClassList.map(classWeight)
        print(ClassWeights.getInfo())
        def setWeight(x):
            x = ee.Feature(x)
            return ClassWeights.get(x.get(Classname))
        weight = ee.Array(featurecollection_list.map(setWeight))
    else:
        weight = ee.Array(ee.List.repeat(1,QC_1.size()))

    presenceInY = sumArray(QC_2.multiply(weight));
    totalNumber = sumArray(weight);
    sequence = ee.Algorithms.If(ee.Number(thresholdList.get(0)).lt(thresholdList.get(-1)), 1, 0);
    def coordinatelist(x):
        index1 = compareIndexArray(Index_2, x, sequence);
        index1_weight = index1.multiply(weight)
        index2 = ee.Array((index1.add(QC_2).gte(2)));
        index2_weight = index2.multiply(weight)
        x_1 = sumArray(index1_weight);
        y_1 = sumArray(index2_weight);
        coor = ee.List([x_1, y_1, ee.Number(x)]);
        return coor;

    result = thresholdList.map(coordinatelist)
    result_output = result
    if (exportCoor):
        result = np.array(result.getInfo()).transpose()
        exportCoorFunc(exportCoor, np.array([result[0].tolist()]), np.array([result[1].tolist()]), result[2].tolist())
    if (exportVariable):
        pInY = presenceInY.getInfo()
        tNumber = totalNumber.getInfo()
        AUC = calculate_AUC(np.array([result[0].tolist()]), np.array([result[1].tolist()]), pInY, tNumber)
        ccorner = correctCorner(np.array([result[0].tolist()]), np.array([result[1].tolist()]), pInY)
        exportVariableFunc(exportVariable, tNumber, pInY, ccorner, AUC)
    return result_output
def TOC_FeatureCollection(FC: ee.featurecollection, QCName: str, IndexnameList: ee.List, thresholdList: ee.List, nameList: list, boolcorrectcorner: bool = False,
                          booluniformline: bool = False,
                          Classname: str = None,
                          ClassList: dict = None, unit: str = 'observations') -> None:
    '''
    This is the function to show the TOC curves from the FeatureCollection.
    :param FC: (ee.FeatureCollection) featurecollection that contains reference and index properties
    :param QCName: (string) The name of reference property
    :param IndexnameList: (ee.List) The list of index property names
    :param thresholdList: (ee.List/list) the list of thresholds, format like [[],[],[],[],[]] (the same order as the name list)
    :param nameList: (list) the list of index names
    :param boolcorrectcorner: (bool) whether to show the correct corners on the diagram
    :param booluniformline: (bool) whether to show uniform line on the diagram
    :param Classname: (string) The name of class property when applying stratified sampling
    :param ClassList: (disctionary) The size of stratums {classname: classsize, classname: classsize ...}
    :param unit: (str) the unit of each record
    :return: No return
    '''

    # change the format to ee format
    if (type(IndexnameList) is not ee.List):
        IndexbandnameList = ee.List(IndexnameList)
    if (type(QCName) is not str):
        QCbandname = QCName.getInfo()
    if (type(thresholdList) is not ee.List):
        thresholdList = ee.List(thresholdList)
    def coordinateList(x):
        x = ee.String(x)
        num = IndexbandnameList.indexOf(x)
        threshold = ee.List(thresholdList.get(num))
        result1 = TOC_Feature_coor(FC, QCName, x, threshold,Classname=Classname,ClassList=ClassList)
        return result1
    coordinates = IndexbandnameList.map(coordinateList)
    coordinates = coordinates.getInfo()
    Xlist = []
    Ylist = []
    for item in coordinates:
        xyList = np.array(item).transpose()
        Xlist.append(xyList[0])
        Ylist.append(xyList[1])
        # Xlist.insert(-1, xyList[0])
        # Ylist.insert(-1, xyList[1])
    Xlist = np.array(Xlist)
    Ylist = np.array(Ylist)
    curveNum = len(Xlist)

    painter = painter_Generator(Xlist, Ylist, unit)
    painter.paintInit(boolUniform=booluniformline, boolCorrectcorner=boolcorrectcorner)
    for i in range(len(nameList)):
        if (boolcorrectcorner):
            painter.correctCorner(i)
        painter.paintOne(i, nameList[i], '^')
    painter.show()


# ---------- This part is painter class----------
#  digits rule: 1. maximum three significant digits; 2. the same decimal places
class labelTranslator:
    def __init__(self, digits,decimalPlaces):
        self.digits = digits
        self.decimalPlaces = decimalPlaces

    def threeDigits(self, temp, position):
        if(temp==0):
            return '0'
        elif(self.decimalPlaces==0):
            return '%.0f' % (temp / 10 ** self.digits)
        elif (self.decimalPlaces == 1):
            return '%.1f' % (temp / 10 ** self.digits)
        elif (self.decimalPlaces == 2):
            return '%.2f' % (temp / 10 ** self.digits)
        elif (self.decimalPlaces == 3):
            return '%.3f' % (temp / 10 ** self.digits)
class painter_Generator:
    def __init__(self, Xlist, Ylist, unit):
        self.Xlist = Xlist
        self.Ylist = Ylist
        self.totalNum = Xlist[0][-1]
        self.presenceInY = Ylist[0][-1]
        self.unit = unit
        self.TOCNum = len(Xlist)
        self.clickIndex = 0
        self.labelList = []
        self.fig = 0

    def axisRange(self,Xmax):
        numDigit = len(str(int(Xmax))) - 1
        resultRange = np.arange(0, Xmax, Xmax / 5)
        # if ((Xmax - resultRange[-1]) < 10 ** numDigit / 5.0):
        #     resultRange = np.delete(resultRange, -1)
        resultRange = np.append(resultRange, Xmax)
        return resultRange


    def paintInit(self, boolUniform=False,boolCorrectcorner=False):
        Xmax = self.totalNum
        Ymax = self.presenceInY
        plt.close()

        self.fig = plt.figure()

        #  maximum, minimum and uniform line
        # l3 = plt.plot([0, Ymax, Xmax], [0, Ymax, Ymax], 'r--', color="blue",
        #               label='Maximum')
        if(boolCorrectcorner):
            plt.text(0, Ymax * 1.01, 'The    marks where Misses equals False Alarms.', color="black", fontsize=8)
            plt.text(0.062 * Xmax, Ymax * 1.01, '★', color="red", fontsize=8)
        if(boolUniform):
            l2 = plt.plot([0, Xmax], [0, Ymax], 'r:', color="violet", label='Uniform')

        # l4 = plt.plot([0, Xmax - Ymax, Xmax], [0, 0, Ymax], 'r--',
        #               color="purple", label='Minimum')
        #  Grey area in outer TOC area
        cmap = matpltColors.ListedColormap('#e0e0e0')
        plt.tripcolor([Xmax-Ymax, Xmax, Xmax], [0, 0, Ymax], [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0, cmap=cmap)
        plt.tripcolor([0, 0, Ymax], [0, Ymax, Ymax], [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0,
                      cmap=cmap)

        # make the coordinates square
        plt.axis('square')
        # plt.axis('auto')
        plt.axis([Xmax * -0.005, Xmax * 1.005, Ymax * -0.005, Ymax * 1.005])
        self.ticksOption()

        # plt.axis([0, Xmax, 0, Ymax])
        plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    def show(self):
        plt.show()

    def paintOne(self, index, Name, marker):
        plt.plot(self.Xlist[index], self.Ylist[index], marker + '-', label=Name)
        handles, labels = plt.gca().get_legend_handles_labels()
        numCurve = len(handles)
        # change the order of maximum and minimum
        # in handles the order will always uniform, x1, x2
        if(labels[0]=='Uniform'):
            order = list(range(1,numCurve))
            order.extend([0])
        else:
            order = list(range(0, numCurve))
        # put the maximum line first, uniform and minimum at last
        # order.insert(0,0)
        # order.extend([1,2])
        plt.gca().legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='center left', bbox_to_anchor=(1, 0.5))
    def correctCorner(self,index):
        correctCornerY = correctCorner(np.array([self.Xlist[index]]), np.array([self.Ylist[index]]),self.presenceInY)
        plt.plot(self.presenceInY, correctCornerY, 'r*')

    def correctCornerLabel(self, index):
        correctCornerY = correctCorner(np.array([self.Xlist[index]]), np.array([self.Ylist[index]]), self.presenceInY)
        plt.text(self.presenceInY, correctCornerY, str((self.presenceInY, round(correctCornerY,2))),color="red")



    # set up for a ticks labels
    def ticksOption(self):
        # ticks for x and y axis
        Xmax = self.totalNum
        Ymax = self.presenceInY
        plt.xticks(self.axisRange(Xmax))
        plt.yticks(self.axisRange(Ymax))

        reference = {0:'',3:'thousand ',6:'million ', 9:'billion ', 12:'trillion '}


        Digits_X_3 = (int(len(str(int(Xmax))) - 1) // 3) * 3
        X_decimal = decimalPlace(Xmax, Digits_X_3)
        plt.xlabel('Hits+False Alarms' + ' (' + reference[Digits_X_3] + self.unit + ')')
        LTranslator = labelTranslator(Digits_X_3,X_decimal)
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(LTranslator.threeDigits))
        # plt.text(Xmax * 1.02, Ymax * 0.000, 'E' + str(Digits_X_3), fontsize=10)

        Digits_Y_3 = (int(len(str(int(Ymax))) - 1) // 3) * 3
        Y_decimal = decimalPlace(Ymax, Digits_Y_3)
        plt.ylabel('Hits' + ' (' + reference[Digits_Y_3] + self.unit + ')')
        LTranslator = labelTranslator(Digits_Y_3,Y_decimal)
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(LTranslator.threeDigits))
        # plt.text(Xmax * 0.00, Ymax * 1.02, 'E' + str(Digits_Y_3), fontsize=10)
    def setLabelList(self, labelList):
        self.labelList = labelList
    # show labels (can change properties later)
    def showLabel(self,list_label):
        for item in list_label:
            plt.text(float(item[0]), float(item[1]), item[2])
    def ClickReact(self, thresholdTable, indexCombo, thresholdDigits):
        fig=self.fig
        self.clickIndex = indexCombo.currentIndex()
        x_for_label = self.Xlist[self.clickIndex]
        y_for_label = self.Ylist[self.clickIndex]
        label_dis = self.labelList[self.clickIndex]

        def onpick(event):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            # points = tuple(zip(xdata[ind], ydata[ind]))
            index_label = -1
            # print(ind)
            # print(xdata.shape)☺
            if(np.array_equal(x_for_label,xdata)):
                index_label=np.array([ind])[0,0]
            # print(index_label)
            # print(x_for_label)

            # if(xdata==x_for_label):
            #     index_label = ind[0]
            #     print(index_label)
            if(index_label>=0):
                # print(ind)
                #
                # print(index_label)
                # print('hello')
                threshold_digits = int(thresholdDigits.text())
                row = thresholdTable.rowCount()
                formatThreshold='{0:.'+str(threshold_digits)+'f}'
                thresholdTable.insertRow(thresholdTable.rowCount())
                thresholdTable.setItem(row, 0, QTableWidgetItem(str(xdata[ind][0])))
                thresholdTable.setItem(row, 1, QTableWidgetItem(str(ydata[ind][0])))
                if (label_dis[index_label]=='origin' or label_dis[index_label]=='end'):
                    thresholdTable.setItem(row, 2, QTableWidgetItem(label_dis[index_label]))
                else:
                    thresholdTable.setItem(row, 2, QTableWidgetItem(formatThreshold.format((float(label_dis[index_label])))))

        fig.canvas.mpl_connect('pick_event', onpick)

# ---------- This part is the relevant functions for painter----------
def decimalPlace(maximum, tenPower):
    resultRange = np.arange(0, maximum, maximum / 5)
    decimalDigits = 0
    for i in resultRange:
        str1="{:.3g}".format(i/(10 ** tenPower))
        if('.' in str1):
            if (len(str1.split('.')[1]) > decimalDigits):
                decimalDigits = len(str1.split('.')[1])
    return decimalDigits
def correctCorner(TOCX,TOCY,presenceInY):
    Boolx1 = (TOCX <= presenceInY).astype(int)
    index1 = Boolx1.sum() - 1
    # print((presenceInY - TOCX[0, index1]))
    if (index1 == TOCY.shape[1] - 1):
        y_res = TOCY[0, -1]
    else:
        y_res = (TOCY[0, index1 + 1] - TOCY[0, index1]) * 1.0 / (TOCX[0, index1 + 1] - TOCX[0, index1]) * (presenceInY - TOCX[0, index1]) + TOCY[0, index1]*1.0
    # problem located
    return y_res


# This part is for output
def calculate_AUC(TOCX,TOCY,presenceInY,totalNum):
    Area01 = areaUnderCurve(TOCX,TOCY)
    Area02 = areaUnderCurve(np.array([[0, presenceInY]]),np.array([[0, presenceInY]]))
    Area03 = areaUnderCurve(np.array([[0, totalNum]]),np.array([[0, presenceInY]]))*2
    AUC = (Area01-Area02)/(Area03-2*Area02)
    return AUC
def AUCfromResult(result):
    if(type(result) != list):
        result = result.getInfo()
    AUC = calculate_AUC(np.array([np.array(result)[:, 0]]), np.array([np.array(result)[:, 1]]), result[-1][1], result[-1][0])
    return AUC[0]
def areaUnderCurve(listx, listy):  # compute the area under the curve
    if(listx[0,-1]>65536):
        listx=np.int64(listx)
        listy=np.int64(listy)
        Llistx = np.delete(listx, listx.shape[1] - 1)
        Hlistx = np.delete(listx, 0)
        Llisty = np.delete(listy, listy.shape[1] - 1)
        Hlisty = np.delete(listy, 0)
        Areas = np.sum(np.int64([(Hlistx - Llistx) * (Hlisty + Llisty)]), axis=1) / 2.0
    else:
        Llistx = np.delete(listx, listx.shape[1] - 1)
        Hlistx = np.delete(listx, 0)
        Llisty = np.delete(listy, listy.shape[1] - 1)
        Hlisty = np.delete(listy, 0)
        Areas = np.sum(np.array([(Hlistx - Llistx) * (Hlisty + Llisty)]), axis=1) / 2.0

    return Areas
def exportCoorFunc(Exportpath,TOCX,TOCY,thresholdList):
    with open(Exportpath, 'w') as f:
        for i in range(TOCX.shape[1]):
            # f.writelines('hello')
            f.write(str(TOCX[0, i]) + " " + str(TOCY[0, i]) + " " + str(thresholdList[i]) + "\n")

def exportVariableFunc(Exportpath,totalNum,presenceInY,correctcorner,AUC):
    with open(Exportpath, 'w') as f:
        f.write('AUC: ' + str('%.3f' % AUC[0]) + "\n")
        f.write('presence in Y: ' + str(presenceInY) + "\n")
        f.write('extent: ' + str(totalNum) + "\n")
        f.write('correct corner: ' + '(' + str(presenceInY)+','+str(correctcorner) + ')' +"\n")
# image function
def sumImage(img):
    sumImg = img.reduceRegion(**{'reducer': ee.Reducer.sum(),'geometry':img.geometry()});
    value = sumImg.get(sumImg.keys().get(0));
    return value;
def maskImage(BiImage, noDataValue):
    mask = BiImage.neq(noDataValue);
    return mask;

def compareIndex(indexArray, number, sequence):
  Bi_index = ee.Algorithms.If(ee.Number(sequence).eq(0), indexArray.gte(ee.Number(number)), indexArray.lte(ee.Number(number)));
  return ee.Image(Bi_index);

# array function
def sumArray(arr1):
    sumArr = arr1.reduce(**{'reducer': ee.Reducer.sum(),'axes': [0]})
    value = sumArr.get([0])
    return ee.Number(value)
def compareIndexArray(indexArray, number, sequence):
    Bi_index = ee.Algorithms.If(ee.Number(sequence).eq(0), indexArray.gte(ee.Number(number)), indexArray.lte(ee.Number(number)));
    return ee.Array(Bi_index);
