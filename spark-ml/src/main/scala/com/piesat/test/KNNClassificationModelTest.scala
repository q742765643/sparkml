package com.piesat.test

import org.apache.spark.ml.classification.KNNClassificationModel
import org.apache.spark.ml.feature.{IndexToString, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object KNNClassificationModelTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("11").master("local").getOrCreate()
    var pc = spark.sparkContext;
    /** *****=======1.加载csv============= *******/
    var df: DataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("D:\\gitcangku\\sparkml\\spark-ml\\data\\regression\\bikesharing\\hour.csv")
    var headers = df.columns
    var lable = headers(headers.length - 1);


    /** *****=======2.最后一列定为标签转为索引============= *******/
    val lableIndex = new StringIndexer().setInputCol(lable).setOutputCol("lable").fit(df);
     var df1=lableIndex.transform(df);

    /** *****=======3.将各个数值型的特征转换成一个特征向量，使用的是VectorAssembler============= *******/
    val featuresVector: VectorAssembler = new VectorAssembler().
      setInputCols(headers.slice(0, headers.length - 1)) //
      .setOutputCol("features")
    //featuresVector.transform(df).show()
    var df2=featuresVector.transform(df1)
    /** *****=======4.将索引转回字符============= *******/
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(lableIndex.labels)
    /** *****=======5.数据归一化============= *******/
    val standardScaler = new StandardScaler()
      .setInputCol(featuresVector.getOutputCol)
      .setOutputCol("scaledFeatures")
      .setWithStd(true)//是否将数据缩放到单位标准差。
      .setWithMean(false).fit(df2)//否在缩放前使用平均值对数据进行居中。
    var df4=standardScaler.transform(df2)
    /*var knn=KNNClassificationModel.load("D:/spark.knnmodel")
    df4.show()
    var df3=labelConverter.transform( knn.transform(df4))
    df3.select("prediction","cnt","predictedLabel").show()
    System.out.println(knn.extractParamMap().toString())*/
  }
}
