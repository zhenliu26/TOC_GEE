# TOC_GEE
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
- **Provide function to export coordinates to be operated on the TOC program*

## Functions

> Tips: Don't forget to authorize the Google Earth Engine account by ee.Authenticate().

### TOC_Image

The parameters in the TOC_Image function are:
- img: (ee.Image) the image which contains the reference band and index bands.
- QCbandname: (String) the band name of QC band
- IndexbandnameList: (ee.List/list) the list of index band names
- thresholdList: (ee.List/list) the list of thresholds, format like [[],[],[],[],[]] (the same order as the name list)
- noDataValue: (number) always be -1
- nameList: (list) The list of band names
- boolcorrectcorner: (bool) whether to show the correct corners on the diagram
- booluniformline: (bool) whether to show uniform line on the diagram
- unit:(String) the unit name

```python
outProperty = ['idNum','ran','startDate']
sample = ee.FeatureCollection('users/BAI_debug/sampleMidWest')
saveAddress = "data\source_test.csv"
```

### Step 2: data preparation

Change date name, ID name in both csv and GEE, and the source name
```python
# import mainUI
ee.Initialize()
# date property name in csv
dateName = "startDate"
# ID property name in csv
IDName = 'idNum'
# ID property name in GEE
IDGEE = 'idString'
# date property name in GEE
dateGEE = 'startDate'
sourceName = 'data/source_test.csv'
#  the dataframe to store the sample data. (it will be updated everytime you click enter)
df = pd.read_csv(sourceName)
sample = ee.FeatureCollection('users/BAI_debug/sampleMidWest')
classCode = ['NotAssessed','water','non-water']
```

Because the imagecollection can't be stored in Google Earth Engine Asset, so that the final result (which is the imagecollection in Google Earth Engine) should be calculated in the data preparation module.
```python
# data processing
finalResult = ee.ImageCollection(image)
```
### Step 3: hand label

When you run the code, the interface will be shown like below.
[![UI](https://raw.githubusercontent.com/zhenliu26/Images/master/sampleUI.jpg)]()

You can select the date and feature. The layers controller will help to change the background layer. When you are certain about the class of the target point, change the class and **Click the Enter**. The record will be updated automatically.

### Step 4: upload back to Google Earth Engine
