# TOC_GEE -- Generate TOC curves from data in Google Earth Engine
## Preparation

There are several required libraries before the start:
- Google Earth Engine
- Matplotlib
- Numpy
> Those liberary can be installed by pip
```shell
$ pip install earthengine-api --upgrade
$ pip install -U matplotlib
$ pip install numpy
```
## Features
- **Save Time. Files and calculation are all on the Google Earth Engine, while drawing the chart is on the local computer**
- **Provide function to export coordinates to be operated on the TOC program**

## Data
There are two types of data format in Google Earth Engine -- **ee.Image** and **ee.FeatureCollection**.

If the data source is the image where all pixels have reference information, **TOC_Image** and **TOC_Image_coor** are the functions to generate TOC curves. If the dataset is featurecollection that stores the sample records, **TOC_FeatureCollection** and **TOC_Feature_coor** are used to generate TOC curves. For the image, the image should contains the band of reference information and the bands of index variable (RS indices, band values, possibilities). 
- In the reference information, the presence should be 1, the absence should be 0, the No Data should be a number (except 0 or 1, always -1). 
- For the threshold squence, it is determined by the index variable. If the higher index variable means the higher possibility of presence, the sequence of the thresholds should be from high to low. For example, because the higher MNDWI means the higher possiblity of water, so the thresholds for MNDWI should be from 1 to -1. If the lower index variable means the higher possibility of presence, the sequence of the thresholds should be from low to high. For example, because the lower NDVI means the higher possiblity of water, so the thresholds for NDVI should be from -1 to 1.

## Functions

> Tips: Don't forget to authorize the Google Earth Engine account by ee.Authenticate().

Before calling the TOC functions, move the TOC_GEE.py to the project folder. And, import the library in the working script.
```python
import TOC_GEE
```

### TOC_Image

This function will generate the TOC curve from the ee.Image (the data format in the Google Earth Engine) and display it. The parameters in the TOC_Image function are:
- img: (ee.Image) the image which contains the reference band and index bands.
- QCbandname: (String) the band name of QC band
- IndexbandnameList: (ee.List/list) the list of index band names
- thresholdList: (ee.List/list) the list of thresholds, format like [[],[],[],[],[]] (the same order as the name list)
- noDataValue: (number) always be -1
- nameList: (list) The list of band names
- boolcorrectcorner: (bool) whether to show the correct corners on the diagram
- booluniformline: (bool) whether to show uniform line on the diagram
- unit:(String) the unit name

The sample code is like:
```python
TOC_GEE.TOC_Image(img,'QC',['mndwi','ndvi'],[ee.List.sequence(-1,1,0.1,None).reverse(),ee.List.sequence(-1,1,0.1,None)],-1,['mndwi','ndvi'],unit='pixels')
```

### TOC_Image_coor

This function will calculate the coordinates of the TOC curve from the ee.Image (the data format in the Google Earth Engine) and export it. (It can handle one index variable once) The parameters in the TOC_Image_coor function are:
- input_binary: (ee.Image) Reference image. presence is 1, absence is 0
- input_index: (ee.Image) Index image.
- thresholdList: (ee.List/list) The list of thresholds. (from high to low or from low to high)
- noDataValue: (number) always be -1
- exportVariable: (string / default is False) The default parameter is False (not export the coordinates), string should be output of variables. (extension should be txt)

The sample code is like:
```python
TOC_GEE.TOC_Image_coor(QC_Image ,MNDWI_Image, ee.List.sequence(-1,1,0.1,None).reverse(),-1,exportCoor='coordinates1.txt',exportVariable='v1.txt')
```

### TOC_FeatureCollection

This function will generate the TOC curve from the ee.FeatureCollection (the data format in the Google Earth Engine) and display it. The parameters in the TOC_FeatureCollection function are:
- FC: (ee.FeatureCollection) featurecollection that contains reference and index properties
- QCName: (string) The name of reference property
- IndexnameList: (ee.List) The list of index property names
- thresholdList: (ee.List/list) the list of thresholds, format like [[],[],[],[],[]] (the same order as the name list)
- nameList: (list) the list of index names
- boolcorrectcorner: (bool) whether to show the correct corners on the diagram
- booluniformline: (bool) whether to show uniform line on the diagram
- Classname: (string) The name of class property when applying stratified sampling
- ClassList: (disctionary) The size of stratums {classname: classsize, classname: classsize ...}
- unit: (str) the unit name

The sample code is like:
```python
TOC_GEE.TOC_FeatureCollection(ftc,'QC',['mndwi','ndvi'],[ee.List.sequence(-1,1,0.1,None).reverse(),ee.List.sequence(-1,1,0.1,None)],['mndwi','ndvi'],boolcorrectcorner=True,Classnmae='class',ClassList={'valley':20,'plain':40,'mountain':40})
```

### TOC_Feature_coor

This function will calculate the coordinates of the TOC curve from the ee.FeatureCollection (the data format in the Google Earth Engine) and export it. (It can handle one index variable once) The parameters in the TOC_Feature_coor function are:
- featurecollection_input: (ee.FeatureCollection) featurecollection that contains reference and index properties
- QCname: (string) The name of reference property
- Indexname: (string) The name of index property
- thresholdList: (ee.List/list) The list of thresholds. (from high to low or from low to high)
- Classname: (string) The name of class property when applying stratified sampling
- ClassList: (disctionary) The size of stratums {classname: classsize, classname: classsize ...}
- exportCoor: (string / default is False) The default parameter is False (not export the coordinates), string should be output path of coordinates. (extension should be txt)
- exportVariable: (string / default is False) The default parameter is False (not export the coordinates), string should be output of variables. (extension should be txt)

The sample code is like:
```python
TOC_GEE.TOC_Feature_coor(ftc,'QC','mndwi',ee.List([0,0.5,1]),boolcorrectcorner=True,Classnmae='class',ClassList={'valley':20,'plain':40,'mountain':40},exportCoor='coordinates1.txt',exportVariable='v1.txt')
```

