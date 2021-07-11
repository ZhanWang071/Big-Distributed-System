# 基于Flask框架的Web界面开发
# Author: 王湛
# Date: 2021/07/01

from flask import Flask, request, render_template, jsonify
import json
import os
import geopandas as gpd
from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.sql import SparkSession, SQLContext, functions
from pyspark.sql.functions import array_contains,count,countDistinct,col, udf
from sedona.utils.adapter import Adapter
from sedona.core.geom.envelope import Envelope
from sedona.core.spatialOperator import RangeQueryRaw
from sedona.core.spatialOperator import KNNQuery
from tqdm import tqdm
from geopy.distance import geodesic

def df2html(df, n=5):
    df.show(n)
    df_html = df.limit(n).toPandas().to_html()
    return df_html

def extract_tag(tags,keyword):
    for line in tags:
        if type(line.key)==str:
            if line.key==keyword:
                return line.value  
        else:
            if line.key.decode('utf-8')==keyword:
                return line.value.decode('utf-8') 
    return "EMPTY"

spark = SparkSession. \
    builder. \
    appName('appName'). \
    config("spark.serializer", KryoSerializer.getName). \
    config("spark.kryo.registrator", SedonaKryoRegistrator.getName). \
    config('spark.jars.packages',
        'org.apache.sedona:sedona-python-adapter-3.0_2.12:1.0.0-incubating,org.datasyslab:geotools-wrapper:geotools-24.0'). \
    getOrCreate()

SedonaRegistrator.registerAll(spark)

sc = spark.sparkContext
sqlContext = SQLContext(sc)

def loadOsmWay(city):
    osmWay = sqlContext.read.parquet("osmdata/"+city+".osm.pbf.way.parquet")
    osmWay = osmWay.select('id','tags','nodes')
    osmWay.createOrReplaceTempView("osmWay")
    return osmWay

def loadOsmNode(city):
    osmNode = sqlContext.read.parquet("osmdata/"+city+".osm.pbf.node.parquet")
    osmNode = osmNode.select('id','tags', 'latitude','longitude')
    osmNode.createOrReplaceTempView("osmNode")
    noderdd = osmNode.rdd.filter(lambda x:x.tags!=[])
    osmNode = spark.createDataFrame(noderdd)
    osmNode.createOrReplaceTempView("osmNode")
    osmNode = spark.sql("SELECT osmNode.id,osmNode.tags, \
                        ST_Point(cast(osmNode.latitude as Decimal(24,20)),cast(osmNode.longitude as Decimal(24,20))) \
                        as geom from osmNode")
    osmNode.createOrReplaceTempView("osmNode")
    return osmNode

def loadOsmNodeAll(city):
    osmNode_all = sqlContext.read.parquet("osmdata/"+city+".osm.pbf.node.parquet")
    osmNode_all = osmNode_all.select('id','tags', 'latitude','longitude')
    osmNode_all.createOrReplaceTempView("osmNode_all")
    osmNode_all = spark.sql("SELECT osmNode_all.id,osmNode_all.tags,ST_Point(cast(osmNode_all.latitude as Decimal(24,20)),cast(osmNode_all.longitude as Decimal(24,20))) as geom from osmNode_all")
    osmNode_all.createOrReplaceTempView("osmNode_all")
    return osmNode_all

def df2csv(df,path,key,value=None,write_csv=True):
    rdd=df.rdd.map(lambda x:(extract_tag(x.tags,"name"),extract_tag(x.tags,key),str(x.geom)))
    osmNode_before=spark.createDataFrame(rdd).withColumnRenamed("_1",'name').withColumnRenamed("_2", 'one_tag').withColumnRenamed("_3",'geom')
    if value:
        osmNode_output=osmNode_before.filter(osmNode_before.one_tag==value).select(osmNode_before.name,osmNode_before.geom)
        if write_csv:
            osmNode_output.write.option("header", "true").csv(path+"/"+value+".csv")

    else:
        osmNode_output=osmNode_before.filter.select(osmNode.name,osmNode_before.geom)
        if write_csv:
            osmNode_output.write.option("header", "true").csv(path+"/"+key+".csv")
    return osmNode_output

def KNN(df,k,my_coordinate,k_v_pair,write_csv=False,path=None):
    #extract the tag and calculate the distance
    key,value=k_v_pair
    rdd=df.rdd.map(lambda x:(extract_tag(x.tags,"name"),extract_tag(x.tags,key),\
                             x.geom,geodesic(my_coordinate,(x.geom.x,x.geom.y)).km,x.id))
    osmNode_before=spark.createDataFrame(rdd).withColumnRenamed("_1",'name')\
                            .withColumnRenamed("_2", 'one_tag').withColumnRenamed("_3",'geom')\
                            .withColumnRenamed("_4",'distance').withColumnRenamed("_5",'id')
    
    #select some columns
    if value:
        osmNode_output=osmNode_before.filter(osmNode_before.one_tag==value).select(osmNode_before.name,osmNode_before.distance,osmNode_before.id)
    else:
        osmNode_output=osmNode_before.select(osmNode_before.name,osmNode_before.distance,osmNode_before.id)
     
    #sort and find the k nearest point
    osmNode_output.createOrReplaceTempView("osmNode_output")
    knn_output=spark.sql("select * from osmNode_output order by distance")
    knn_output=knn_output.limit(k)
    
    #if we set write_csv as True, we should choose one path to save data
    if write_csv==True:
        if value:
            knn_output.write.option("header", "true").csv(path+"/knn_"+value+".csv")
        else:
            knn_output.write.option("header", "true").csv(path+"/knn_"+key+".csv")
            
    return knn_output

def distance(df,key1,key2,method="name"):#method="name" or "id"
    if method=="name":
        searchrdd=df.rdd.map(lambda x:(extract_tag(x.tags,"name"),x.geom))
        geom1=searchrdd.filter(lambda x:x[0]==key1).collect()[0][1]
        geom2=searchrdd.filter(lambda x:x[0]==key2).collect()[0][1]
    elif method=="id":
        searchrdd=df.rdd.map(lambda x:(x.id,x.geom))
        geom1=searchrdd.filter(lambda x:x[0]==key1).collect()[0][1]
        geom2=searchrdd.filter(lambda x:x[0]==key2).collect()[0][1]
    c1=(geom1.x,geom1.y)
    c2=(geom2.x,geom2.y)
    dist=geodesic(c1,c2).km
    return dist

app = Flask(__name__, template_folder='.', static_folder=".")

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/step0', methods=['GET','POST'])
def step0():
    message = request.get_json()
    print(message['message'])
    city = message['message']
    sqlContext.setConf("spark.sql.parquet.binaryAsString","true")

    global osmWay
    osmWay = loadOsmWay(city)

    global osmNode
    osmNode = loadOsmNode(city)

    global osmNode_all
    osmNode_all = loadOsmNodeAll(city)

    osmWayHtml = df2html(osmWay)
    osmNodeHtml = df2html(osmNode)
    return jsonify({
        'way': osmWayHtml,
        'node': osmNodeHtml
    })

@app.route('/step1', methods=['GET','POST'])
def step1():
    message = request.get_json()
    df = osmNode if message['data'] == 'node' else osmWay
    path = message['folder'] if message['folder'] else "."
    key = message['key']
    value = message['value'] if message['value'] else None
    write_csv = True if message['write'] == 'Yes' else False

    osmNode_output = df2csv(df,path,key,value,write_csv)
    resultHtml = df2html(osmNode_output)
    return jsonify({
        'result': resultHtml
    })

@app.route('/step2', methods=['GET','POST'])
def step2():
    message = request.get_json()
    df = osmNode if message['data'] == 'node' else osmWay
    path = str(message['folder']) if message['folder'] else "."
    key = message['key']
    value = message['value'] if message['value'] else None
    k_v_pair = (key, value)
    write_csv = True if message['write'] == 'Yes' else False
    k = int(message['num'])
    my_coordinate = (float(message['lat']), float(message['lng']))
    print(k,my_coordinate,k_v_pair,write_csv,path)

    result = KNN(df,k,my_coordinate,k_v_pair,write_csv,path)
    resultHtml = df2html(result)
    return jsonify({
        'result': resultHtml
    })

@app.route('/step3', methods=['GET','POST'])
def step3():
    message = request.get_json()
    df = osmNode if message['data'] == 'node' else osmWay
    key1 = message['key1']
    key2 = message['key2']
    method = message['method']
    if method == 'id':
        key1, key2 = int(key1), int(key2)

    result = distance(df,key1,key2,method)
    return jsonify({
        'result': result
    })

def spatial_range_query(df,window,k_v_pair,write_csv=False,path=None):
    key,value=k_v_pair
    #spatial range query in spatialRDD
    osmNode_rdd = Adapter.toSpatialRdd(df,"geom")
    osmNode_rdd.analyze()
    range_query_window = Envelope(window[0],window[1],window[2],window[3])
    consider_boundary_intersection = False
    using_index = False
    query_result = RangeQueryRaw.SpatialRangeQuery(\
                   osmNode_rdd, range_query_window, consider_boundary_intersection, using_index)
    osmNode_output = Adapter.toDf(query_result, spark,["id","tags"])
    return osmNode_output

@app.route('/step4', methods=['GET','POST'])
def step4():
    message = request.get_json()
    df = osmNode if message['data'] == 'node' else osmWay
    window = (float(message['num1']), float(message['num2']), \
              float(message['num3']),float(message['num4']))
    key = message['key']
    value = message['value'] if message['value'] else None
    k_v_pair = (key, value)
    path = str(message['folder']) if message['folder'] else "."
    write_csv = True if message['write'] == 'Yes' else False

    result = spatial_range_query(df,window,k_v_pair,write_csv,path)
    resultHtml = df2html(result)
    return jsonify({
        'result': resultHtml
    })


def get_polygon(nodes):
    lis=[]
    for node in nodes:
        node_id=node[1]
        point=osmNode_all.filter(osmNode_all.id==node_id).select("geom").collect()[0][0]
        lis.append(point)
    polygon=Polygon(lis)
    return polygon

def name2polygon(df,name,write_csv=False,path=None):
    rdd=df.rdd.map(lambda x:(extract_tag(x.tags,"name"),x.nodes)).filter(lambda x:x[0]==name)
    try:
        nodes=rdd.collect()[0][1]
    except:
        raise Exception("The place does not exist.")
    geom=get_polygon(nodes)
    
    if write_csv==True:
        data=[(name,str(geom))]
        write_df=spark.createDataFrame(data, schema = ['name','geom'])
        write_df.write.option("header", "true").csv(path+"/"+name+".csv")
    return geom

def locate_point(df,my_point,write_csv=False,path=None):
    rdd=df.rdd.map(lambda x:(x.geom,extract_tag(x.tags,"name"))).filter(lambda x:x[1]==my_point)
    try:
        geom=rdd.collect()[0][0]
    except:
        raise Exception("The place does not exist.")
        
    if write_csv:
        data=[(my_point,str(geom))]
        write_df=spark.createDataFrame(data, schema = ['pointname','geom'])
        write_df.write.option("header", "true").csv(path+"/"+my_point+"loc_point.csv")
    return geom

def searchWay(nodes,target_id):
    for i,node_id in nodes:
        if node_id==target_id:
            return True
    return False

def locate(df_node,df_way,my_point,write_csv=False,path=None):
    #find the node id
    rdd=df_node.rdd.map(lambda x:(x.id,extract_tag(x.tags,"name"))).filter(lambda x:x[1]==my_point)
    try:
        node_id=rdd.collect()[0][0]
    except:
        raise Exception("The place does not exist.")
    
    way_rdd=df_way.rdd.map(lambda x:(x.id,extract_tag(x.tags,"name"),x.nodes,searchWay(x.nodes,node_id)))
    way_df=spark.createDataFrame(way_rdd).withColumnRenamed("_1",'id')\
                            .withColumnRenamed("_2", 'name').withColumnRenamed("_3",'nodes')
    way_df=way_df.filter(way_df._4==True)
    pathname=way_df.collect()[0].name
    
    geom=name2polygon(osmWay,pathname,write_csv=False,path=None)
    
    if write_csv:
        data=[(pathname,str(geom))]
        write_df=spark.createDataFrame(data, schema = ['pathname','geom'])
        write_df.write.option("header", "true").csv(path+"/"+my_point+"_loc_way.csv")
    
    #search in osmWay
    return pathname,geom

@app.route('/step5_1', methods=['GET','POST'])
def step5_1():
    message = request.get_json()
    df = osmNode if message['data'] == 'node' else osmWay
    path = message['folder'] if message['folder'] else "."
    name = message['name']
    write_csv = True if message['write'] == 'Yes' else False

    result = locate_point(df,name,write_csv,path)
    print(result)
    return jsonify({
        'result': str(result)
    })

@app.route('/step5_2', methods=['GET','POST'])
def step5_2():
    message = request.get_json()
    path = message['folder'] if message['folder'] else "."
    name = message['name']
    write_csv = True if message['write'] == 'Yes' else False

    pathname,geom = locate(osmNode, osmWay, name, write_csv, path)
    print(pathname, geom)
    return jsonify({
        'pathname': str(pathname),
        'geom': str(geom)
    })

@app.route('/step6', methods=['GET','POST'])
def step6():
    message = request.get_json()
    df = osmNode if message['data'] == 'node' else osmWay
    path = message['folder'] if message['folder'] else "."
    name = message['name']
    write_csv = True if message['write'] == 'Yes' else False

    result = name2polygon(df,name,write_csv,path)
    print(result)
    return jsonify({
        'result': str(result)
    })

if __name__ == '__main__':
    app.run()

