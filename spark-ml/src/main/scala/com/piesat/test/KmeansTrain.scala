package com.piesat.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object KmeansTrain {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("DecisionTreeRegression").master("local").getOrCreate()
    var pc = spark.sparkContext;
    /** *****=======1.加载csv============= *******/
    var df: DataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("D:\\gitcangku\\sparkml\\spark-ml\\data\\clustering\\gps.csv")
    var headers = df.columns


    /** *****=======2.将各个数值型的特征转换成一个特征向量，使用的是VectorAssembler============= *******/
    val featuresVector: VectorAssembler = new VectorAssembler().
      setInputCols(headers.slice(0, headers.length - 1)) //
      .setOutputCol("features")
    //featuresVector.transform(df).show()

    //标准化（归一化）
    /** *****=======3.数据标准化============= *******/
    val standardScaler = new StandardScaler()
      .setInputCol(featuresVector.getOutputCol)
      .setOutputCol("scaledFeatures")
      .setWithStd(true) //是否将数据缩放到单位标准差。
      .setWithMean(false) //是否在缩放前使用平均值对数据进行居中。


    /** *****=======4.通过KMeans建立模型============= *******/
    val rfc = new KMeans()
      .setFeaturesCol(featuresVector.getOutputCol)
      .setPredictionCol("prediction")

    /** *****=======5.通过Pipeline实现管道，通过SetStages来设置各个阶段的处理项============= *******/
    val pipeline: Pipeline = new Pipeline().setStages(Array(featuresVector, standardScaler, rfc))

    /** *****=======6.过网格搜索完成超参数的验证============= *******/
    val paramMap = new ParamGridBuilder()
      .addGrid(rfc.getParam("maxIter"), Array(10,20,50,100))
      .addGrid(rfc.getParam("k"), Array(5,6,7,8,9,10,11))
      .addGrid(rfc.getParam("seed"), Array(10l, 20l, 30l))
      .build

    /** *****=======7.聚类评估器============= *******/
    val clusteringEvaluator = new ClusteringEvaluator()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMetricName("silhouette")

    /** *****=======8.交叉验证评估器============= *******/
    val cv: CrossValidator = new CrossValidator().setEstimator(pipeline).setEvaluator(clusteringEvaluator).setEstimatorParamMaps(paramMap).setNumFolds(5)

    /** *****=======9.拆分训练集和测试集============= *******/
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    /** *****=======10.训练模型============= *******/
    val cvModel: CrossValidatorModel = cv.fit(trainingData)
    df.show()

    /** *****=======11.获取最好的模型============= *******/
    val bestModel: PipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    /** *****=======12.模型保存============= *******/
    val rfcBestModel: KMeansModel = bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[KMeansModel]
    rfcBestModel.save("D:/spark.kmodel")
    //加载模型
    //PipelineModel.load("D:/spark.model")
    /** *****=======13.打印重要特征============= *******/
    rfcBestModel.clusterCenters.foreach(println)

    /** *****=======14.打印参数============= *******/
    System.out.println(rfcBestModel.extractParamMap().toString())
    //System.out.println(rfcBestModel.params.toList.mkString("(", ",", ")"))
    //System.out.println(rfcBestModel.explainParams())
    /** *****=======15.使用最后模型预测============= *******/
    val predictionResultDF = bestModel.transform(testData)
    predictionResultDF.show(20)
    /** *****=======16.轮廓的平方欧氏距离============= *******/
    val silhouette = clusteringEvaluator.evaluate(predictionResultDF)
    println("评估结果 轮廓的平方欧氏距离 silhouette==============" + silhouette)
    spark.stop()
  }
}
