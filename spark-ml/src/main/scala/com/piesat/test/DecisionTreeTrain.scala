package com.piesat.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 回归算法，决策树回归
 */
object DecisionTreeTrain {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("DecisionTreeRegression").master("local").getOrCreate()
    var pc = spark.sparkContext;
    /** *****=======1.加载csv============= *******/
    var df: DataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("D:\\gitcangku\\sparkml\\spark-ml\\data\\regression\\bikesharing\\hour.csv")
    var headers = df.columns
    var lable = headers(headers.length - 1);


    /** *****=======2.最后一列定为标签转为索引============= *******/
    val lableIndex = new StringIndexer().setInputCol(lable).setOutputCol("lable").fit(df);
    //lableIndex.fit(df).transform(df).show();

    /** *****=======3.将各个数值型的特征转换成一个特征向量，使用的是VectorAssembler============= *******/
    val featuresVector: VectorAssembler = new VectorAssembler().
      setInputCols(headers.slice(0, headers.length - 1)) //
      .setOutputCol("features")
    //featuresVector.transform(df).show()

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
      .setWithMean(false)//否在缩放前使用平均值对数据进行居中。

    /** *****=======6.通过决策树建立模型============= *******/
    val rfc: DecisionTreeClassifier = new DecisionTreeClassifier().setLabelCol("lable").setFeaturesCol("features")

    /** *****=======7.通过Pipeline实现管道，通过SetStages来设置各个阶段的处理项============= *******/
    val pipeline: Pipeline = new Pipeline().setStages(Array(lableIndex, featuresVector, rfc, labelConverter,standardScaler))

    /** *****=======8.过网格搜索完成超参数的验证============= *******/
    val paramMap: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(rfc.maxDepth, Array(3, 5, 10, 20, 25))
      .addGrid(rfc.maxBins, Array(30, 50, 100, 200))
      .addGrid(rfc.minInstancesPerNode, Array(1, 3, 5, 10, 20))
      .addGrid(rfc.minInfoGain, Array(0.0, 0.3, 0.5))
      .build()

    /** *****=======9.多标签分类的评估器============= *******/
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator().setLabelCol("lable").setPredictionCol("prediction").setMetricName("accuracy")
    //二分类验证器适应只有两分类
    //val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("lable").setMetricName("areaUnderROC")

    /** *****=======10.交叉验证评估器============= *******/
    val cv: CrossValidator = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramMap).setNumFolds(3)

    /** *****=======11.拆分训练集和测试集============= *******/
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    /** *****=======12.训练模型============= *******/
    val cvModel: CrossValidatorModel = cv.fit(trainingData)
    df.show()

    /** *****=======13.获取最好的模型============= *******/
    val bestModel: PipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    /** *****=======14.模型保存============= *******/
    val rfcBestModel: DecisionTreeClassificationModel = bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[DecisionTreeClassificationModel]
    rfcBestModel.save("D:/spark.model")
    //加载模型
    //PipelineModel.load("D:/spark.model")
    /** *****=======15.打印重要特征============= *******/
    val importancesVector: ml.linalg.Vector = rfcBestModel.featureImportances
    println(importancesVector)

    /** *****=======16.打印参数============= *******/
    System.out.println(rfcBestModel.extractParamMap().toString())
    //System.out.println(rfcBestModel.params.toList.mkString("(", ",", ")"))
    //System.out.println(rfcBestModel.explainParams())
    /** *****=======17.使用最后模型预测============= *******/
    val predictionResultDF = bestModel.transform(testData)
    predictionResultDF.show(20)
    /** *****=======18.获取错误率============= *******/
    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
    println("准确率 = " + ( predictionAccuracy))
    spark.stop()

  }

}
