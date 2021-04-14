package com.piesat.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{Bucketizer, IndexToString, StandardScaler, StringIndexer, VectorAssembler, VectorIndexer, Word2Vec}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.loss.SquaredError
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

/**
 * 回归算法，决策树回归
 */
object DecisionTreeRegression {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("DecisionTreeRegression").master("local").getOrCreate()
    //todo 加载数据
    var pc=spark.sparkContext;
    val data = pc.textFile("D:\\gitcangku\\sparkml\\spark-ml\\data\\regression\\bikesharing\\hour.csv")
    val head = data.first();
    var a=data.take(2);
    System.out.println(data.take(1))

    val rawRdd: RDD[Array[String]] = data
      .filter(!_.equals(head))
      .map(_.split(","))

    val rdd1 = rawRdd.map(x=>{
      (x(x.length-1),x.slice(0,x.length-1))
    }
    )


/*    val rdd = rawRdd.map(x=>
      LabeledPoint(x(x.length-1).toDouble
        ,Vectors.dense(x.slice(0,x.length-1).map(_.toDouble)))
    )*/


    //将rdd进行持久化
    //rdd.cache()
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text");
    documentDF.show()
    val word2vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)

    val model = word2vec.fit(documentDF)
    val result = model.transform(documentDF)
    result.show()




    val df = spark.createDataFrame(Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))).toDF("label","features")
    df.show()

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)
    labelIndexer.transform(df).show();



    // Automatically identify categorical features, and index them.
/*    val featureIndexer = new Word2Vec()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(df)*/
   // featureIndexer.transform(df).show()

    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))


    /* val head = csvData.first();
        val rawRdd: RDD[Array[String]] = csvData
          .filter(!_.equals(head))
          .map(_.split(","))

        var rdd=rawRdd.map(x=> (x(x.length-1),x.slice(0,x.length-1)))
        val data = spark.createDataFrame(rdd).toDF("label","features")
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(data)

        val featureIndexer = new Word2Vec()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setVectorSize(1)
          .setMinCount(0).fit(data)

        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


        // Train a DecisionTree model.
        val dt = new DecisionTreeClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)

        // Chain indexers and tree in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainingData)

        // Make predictions.
        val predictions = model.transform(testData)

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5)

        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test Error = ${(1.0 - accuracy)}")

        val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
        println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    */
    spark.stop()

  }

}
