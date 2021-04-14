package com.piesat.decisiontree

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, KNNClassificationModel}
import org.apache.spark.ml.feature.{IndexToString, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
object DecisionTreePrediction {
  def main(args: Array[String]): Unit = {
    args.foreach(println)
    var infilePath=args.apply(0);
    var outfilePath=args.apply(1);
    var modelPath=args.apply(2);
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("DecisionTreePrediction").getOrCreate()

    /** *****=======1.加载csv============= *******/
    var df: DataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(infilePath)
    var headers = df.columns
    for (colName <- headers) {
      val featuresIndex = new StringIndexer().setInputCol(colName).setOutputCol(colName+"Index").fit(df);
      df=featuresIndex.transform(df);
    }
    var pip:PipelineModel=PipelineModel.load(modelPath);
    var out=pip.transform(df)
    headers=headers:+"predictedLabel"
    out.select(headers.map(col(_)): _*).show()
    out.select(headers.map(col(_)): _*).coalesce(1).write.option("header", "true").csv(outfilePath)

  }
}
