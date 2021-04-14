package com.piesat

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext

object Test {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("DecisionTreeRegression").master("local").getOrCreate()
    var sc = spark.sparkContext;
    var ssc = new SQLContext(sc);
    val obj1 = new TClass(
      List(Array('1', '2', '3'), null),
      Map("123" -> Array(1, 2, 3),
        "nil" -> null),
      new TInnerClass(new java.util.Date))
    val obj2 = new TClass(
      List(Array('1', '2', '3'), null),
      Map("empty" -> Array(),
        "90" -> Array(9, 0)),
      new TInnerClass(null))
    val tClazz = classOf[TClass]
    val rdd = sc.makeRDD(Seq(obj1, obj2))
    val rowRDD = rdd.flatMap(DataFrameReflectUtil.getRow(tClazz, _))
    DataFrameReflectUtil.getStructType(tClazz) match {
      case Some(scheme) =>
        val df = ssc.createDataFrame(rowRDD, scheme)
        df.registerTempTable("df")
        df.printSchema
        ssc.sql("select list1, map1, obj1 from df").show(false)
        ssc.sql("select map1['90'], map1['90'][0], date_add(obj1.date1, 1) from df").show(false)
      case None =>
        println("getStructType failed")
    }
  }

}
class TClass(
              val list1: List[Array[Char]],
              val map1: Map[String, Array[Int]],
              val obj1: TInnerClass) extends Serializable

class TInnerClass(
                   val date1: java.util.Date) extends Serializable

