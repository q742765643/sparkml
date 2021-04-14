package com.piesat.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, Word2Vec}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * 回归算法，决策树回归
 */
object DecisionTreeRegression1 {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("DecisionTreeRegression").master("local").getOrCreate()
    //todo 加载数据
    var pc=spark.sparkContext;
    val data=Seq("i",
      Vectors.dense(-1,3,-1,-9,88),
      Vectors.dense(0,5,1,10,96),
      Vectors.dense(0,5,1,11,589))
    val df=spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(3)
    val indexerModel = indexer.fit(df)
    indexerModel.transform(df).show()



    spark.stop()

  }

}
