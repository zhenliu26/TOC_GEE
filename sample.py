import TOC_GEE
import ee

ee.Initialize()

# data preparation
def getwater(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
    mndwi = image.normalizedDifference(['B3', 'B12']).rename('mndwi');
    image = image.addBands(ndvi).addBands(mndwi);
    return image;
img1 = getwater(ee.Image('projects/cloudtostreet/ML/coincident_S1S2_chipped/USA_348639'))
MNDWI = img1.select('mndwi')
NDVI = img1.select('ndvi')
QC = img1.select('QC')
ftc = ee.FeatureCollection('projects/cloudtostreet/TOC/Midwest2019/midwestData')


def normalized(num1, num2):
    num1 = ee.Number(num1)
    num2 = ee.Number(num2)
    result = num1.subtract(num2).divide(num1.add(num2))
    return result
def getindices(x):
    x = ee.Feature(x)
    mndwi = normalized(x.get('B3'),x.get('B12'))
    ndvi = normalized(x.get('B8'),x.get('B4'))
    x = x.set('mndwi',mndwi).set('ndvi',ndvi)
    return x

ftc = ftc.map(getindices)

# ftc = ftc.filterMetadata('startDate', 'equals', '2019-03-26')
ftc_QC = ftc


# Call function
# # TOC_Image
# TOC_GEE.TOC_Image(img1,'QC',['mndwi','ndvi'],[ee.List.sequence(-1,1,0.1,None).reverse(),ee.List.sequence(-1,1,0.1,None)],-1,['mndwi','ndvi'],unit='pixels')

# #TOC_FeatureCollection
# TOC_GEE.TOC_FeatureCollection(ftc_QC,'QC',['mndwi','ndvi'],[ee.List.sequence(-1,1,0.1,None).reverse(),ee.List.sequence(-1,1,0.1,None)],['mndwi','ndvi'],boolcorrectcorner=True)

# # TOC_Image_coor
# a= TOC_GEE.TOC_Image_coor(QC,MNDWI,ee.List.sequence(-1,1,0.1,None).reverse(),-1,exportCoor='coordinates4.txt',exportVariable='v1.txt')
# AUC_ftc = TOC_GEE.AUCfromResult(a)
# print(AUC_ftc)

# # TOC_Feature_coor
# a = TOC_GEE.TOC_Feature_coor(ftc_QC,'QC','mndwi',ee.List([0,0.5,1]))
