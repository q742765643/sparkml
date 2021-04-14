package com.piesat.test
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


// 下面这个UDF函数的含义是将测试集的数据和训练集的数据结合起来，选择年龄这个字段，来求年龄字段的平均值，取出一个年龄的平均值转换称Double类型返回。
object titanicAnalusis {

  def getUserAvgAge(trainDf:DataFrame,testDf:DataFrame) ={
    // .head()  Returns the first row. 返回第一行,返回一行中的位置是0的数以原始双精度值返回。
    trainDf.union(testDf).agg(avg("Age")).head().getDouble(0)
  }

  // 平均票价
  def getFareAvgAge(trainDf:DataFrame,testDf:DataFrame) ={
    // .head()  Returns the first row. 返回第一行,返回一行中的位置是0的数以原始双精度值返回。
    trainDf.union(testDf).agg(avg("Fare")).head().getDouble(0)
  }

  def main(args: Array[String]): Unit = {
    // 1. 准备环境
    // 1.1 配置SparkConf
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("titanicAnalusisModel")
    // 1.2 构建SparkSession
    val sparkSession: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    // 1.3 获取SparkContext
    val sc: SparkContext = sparkSession.sparkContext
    sc.setLogLevel("WARN")
    // 2. 准备数据，读取测试集数据和训练集数据
    val trainPath = "train.csv"
    val testPath = "test.csv"


    // 2.1 从csv中读取数据
    var trainDf: DataFrame = sparkSession.read.format("csv").option("header","true").option("inferSchema","true").load(trainPath)
    var testDf: DataFrame = sparkSession.read.format("csv").option("header","true").option("inferSchema","true").load(trainPath)
    //    trainDf.show(false)
    //    |PassengerId|Survived|Pclass|Name                                                   |Sex   |Age |SibSp|Parch|Ticket          |Fare   |Cabin|Embarked|
    //    +-----------+--------+------+-------------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
    //    |1          |0       |3     |Braund, Mr. Owen Harris                                |male  |22.0|1    |0    |A/5 21171       |7.25   |null |S       |
    //      |2          |1       |1     |Cumings, Mrs. John Bradley (Florence Briggs Thayer)    |female|38.0|1    |0    |PC 17599        |71.2833|C85  |C       |
    //    root
    //    |-- PassengerId: integer (nullable = true)
    //    |-- Survived: integer (nullable = true)
    //    |-- Pclass: integer (nullable = true)
    //    |-- Name: string (nullable = true)
    //    |-- Sex: string (nullable = true)
    //    |-- Age: double (nullable = true)
    //    |-- SibSp: integer (nullable = true)
    //    |-- Parch: integer (nullable = true)
    //    |-- Ticket: string (nullable = true)
    //    |-- Fare: double (nullable = true)
    //    |-- Cabin: string (nullable = true)
    //    |-- Embarked: string (nullable = true)

    trainDf.printSchema()
    //    testDf.show(false)

    // 3-数据的清洗与处理，使用SparkSql来完成相应的操作。
    // 3-1 处理年龄缺失值的方式，比如使用平均值来代替缺失值，下面定义了一个UDF函数
    //其实就像C语言中的API，通过API对参数进行相关的操作后返回

    val avgAge: Double = getUserAvgAge(trainDf,testDf)
    // 3-2 打印一下所有乘客年龄的平均值
    println(avgAge)

    // 3-3 对年龄字段中的缺失值填充平均值，na方法的含义是返回一个DataFrameNaFunctions对象来处理缺失值
    trainDf = trainDf.na.fill(avgAge,Array("Age"))
    testDf = testDf.na.fill(avgAge,Array("Age"))
    //    trainDf.show(false)
    //    testDf.show(false)

    // Returns a [[DataFrameNaFunctions]] for working with missing data.
    // 3-4 处理上船地址是缺失值的数据，填充为S
    // trainDf = trainDf.na.fill("S",Array("Embarked"))
    trainDf = trainDf.na.fill("S",Array("Embarked"))
    testDf = testDf.na.fill("S",Array("Embarked"))
    // 3-5 乘船票价缺失的项填充为车票的平均值。
    val avgFare: Double = getFareAvgAge(trainDf,testDf)
    println(avgFare)
    trainDf = trainDf.na.fill(avgFare, Array("Fare"))
    testDf = testDf.na.fill(avgFare,Array("Fare"))

    // 3-6 定义一个udf函数，从name字段中得出头衔
    // 3-6-1自定义UDF函数
    val getTitleFromName = (name:String) =>{
      if (name.contains("Miss") )
        "Miss"
      else if (name.contains("Mr") )
        "Mr"
      else if (name.contains("Mrs") )
        "Mrs"
      else "RareTitle"

    }
    // 3-6-2注册UDF函数
    sparkSession.udf.register("getTitleFromName", getTitleFromName)

    // 3-7 使用UDF函数，获取每一行中的Name列，进行处理后得到Title列，如果title存在，则替换，如果不存在则新增
    // 返回一个新的DataSet并添加一个新的列名，如果列名已经存在则进行替换。
    // 第二个参数表示的Column,列的实体，头衔这个列是通过性别的列转换而来。
    trainDf = trainDf.withColumn("title", callUDF("getTitleFromName",trainDf("Name")))
    testDf = testDf.withColumn("title", callUDF("getTitleFromName",testDf("Name")))
    trainDf.show(false)
    //    +-----------+--------+------+-------------------------------------------------------+------+------------------+-----+-----+----------------+-------+-----+--------+---------+
    //    |PassengerId|Survived|Pclass|Name                                                   |Sex   |Age               |SibSp|Parch|Ticket          |Fare   |Cabin|Embarked|title    |
    //    +-----------+--------+------+-------------------------------------------------------+------+------------------+-----+-----+----------------+-------+-----+--------+---------+
    //    |1          |0       |3     |Braund, Mr. Owen Harris                                |male  |22.0              |1    |0    |A/5 21171       |7.25   |null |S       |Mr       |
    // 3.8 如果性别为女，年龄大于18岁，并且有子女或父母，认为这个人是一个母亲，这个是规则，规则跟最终的评判结果还是相关的，那如何找规则，这是一个难点。
    // Parses the expression string into the column that it represents, similar to
    // when(expr(),value).otherwise(other value)将表达式字符串解析到它表达的列中，里面的字符串使用单引号括起来,符合字符串中的规则则返回一个值，不符合则返回另外一个值
    // 将这个Column生成规则放入到withColumn中的第二个参数上
    // 第二个参数表示的Column,列的实体，头衔这个列是通过性别的列转换而来。DSL中的等于号是两个=
    val motherColumn: Column = when(expr("Age > 18 and Parch > 0 and Sex == 'female'"),1).otherwise(0)
    trainDf = trainDf.withColumn("mother",motherColumn)
    testDf = testDf.withColumn("mother",motherColumn)
    //    trainDf.show(false)
    //    testDf.show(false)
    //    +-----------+--------+------+--------------------+------+-----------------+-----+-----+----------------+-------+-----+--------+---------+------+
    //    |PassengerId|Survived|Pclass|                Name|   Sex|              Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|    title|mother|
    //    +-----------+--------+------+--------------------+------+-----------------+-----+-----+----------------+-------+-----+--------+---------+------+
    //    |          1|       0|     3|Braund, Mr. Owen ...|  male|             22.0|    1|    0|       A/5 21171|   7.25| null|       S|       Mr|     0|
    // 4-数据的特征工程
    // 4.1.1 StringIndexer,Sex列进行转换，字符串转换成索引
    val sexIndex: StringIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    // 4.1.2 上船位置列从字符串转换成索引
    val emberkedIndex: StringIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")
    // 4.1.3 头衔列字符串转换成索引
    val titleIndex: StringIndexer = new StringIndexer().setInputCol("title").setOutputCol("titleIndex")
    // 4.1.4 是否幸存标签列的字符串转换成索引
    val survivedIndex: StringIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("SurvivedIndex")

    // 4.1.5 判断将要做特征的列中是否有缺失值，有的话构建模型会失败
    //    println("null column dataFrame ==============")
    //    val nullEmbarked:DataFrame = trainDf.filter(trainDf("Embarked").isNull)
    //    println(nullEmbarked.count())
    //    val nullTitleIndex: DataFrame = trainDf.filter(trainDf("title").isNull)
    //    println(nullTitleIndex.count())
    //    val nullMother: DataFrame = trainDf.filter(trainDf("mother").isNull)
    //    println(nullMother.count())
    //    val nullPclass: DataFrame = trainDf.filter(trainDf("Pclass").isNull)
    //    println(nullPclass.count())
    //    val nullAge: DataFrame = trainDf.filter(trainDf("Age").isNull)
    //    println(nullAge.count())
    //    val nullFare: DataFrame = trainDf.filter(trainDf("Fare").isNull)
    //    println(nullFare.count())
    //    val nullSibSp: DataFrame = trainDf.filter(trainDf("SibSp").isNull)
    //    println(nullSibSp.count())
    //    val nullParch: DataFrame = trainDf.filter(trainDf("Parch").isNull)
    //    println(nullParch.count())

    // 4.2 将各个数值型的特征转换成一个特征向量，使用的是VectorAssembler
    val featuresVector: VectorAssembler = new VectorAssembler().
      setInputCols(Array("SexIndex", "EmbarkedIndex", "titleIndex", "mother", "Pclass", "Age", "Fare", "SibSp", "Parch"))//
      .setOutputCol("Features")
    // 5. 通过随机森林算法建立分类模型
    val rfc: RandomForestClassifier = new RandomForestClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("Features")
    // 6. 通过Pipeline实现管道，通过SetStages来设置各个阶段的处理项
    val pipeline: Pipeline = new Pipeline().setStages(Array(sexIndex,emberkedIndex,titleIndex,survivedIndex,featuresVector,rfc))
    // 7-通过网格搜索完成超参数的验证 添加具有多个值的参数，即一个参数有多个候选值，通过交叉验证选择出模型性能最好的参数
    val paramMap: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(rfc.maxDepth, Array(5, 10, 20))
      .addGrid(rfc.numTrees, Array(10, 50, 100))
      .addGrid(rfc.maxBins, Array(16, 32, 48)).build()

    // 8. 通过二分类的评估器进行评估
    val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("SurvivedIndex").setMetricName("areaUnderROC")
    // 9. 结合交叉验证得到最佳模型的参数，那么交叉验证中需要设置Pipeline,网格参数生成器，评估器，交叉验证折叠的次数
    // 比如折叠的次数为10，那么会生成10个（测试，训练）数据集对，每一个数据集对中的1/10的数据会当成测试集，9/10的数据会当成训练集
    val cv: CrossValidator = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramMap).setNumFolds(2)
    // 9.1 交叉验证评估器通过DataFrame数据训练出模型
    val cvModel: CrossValidatorModel = cv.fit(trainDf)

    // 10. 通过训练出模型进行预测
    // 10.1 获取最好的模型
    val bestModel: PipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    // 10.1.1 选择索引是5的pipelineModel中的成员，PipelineModel中的estimator已经被fit后成为transformer了
    val rfcBestModel: RandomForestClassificationModel = bestModel.asInstanceOf[PipelineModel].stages(5).asInstanceOf[RandomForestClassificationModel]
    // 10.1.2 获取RandomForestClassificationModel的重要的特征
    val importancesVector: linalg.Vector = rfcBestModel.featureImportances
    // 10.1.3 打印特征向量，打印重要的特征
    println(importancesVector)
    //    (9,[0,1,2,3,4,5,6,7,8],[0.3571442316030072,0.02868788017597425,0.13432757515958677,0.02308271177004318,0.12950593321054085,0.07913464623618699,0.1701980601347919,0.05349619741059557,0.024422764299273255])

    // 10.2 通过最好的模型对测试数据集进行预测
    val resutl: DataFrame = bestModel.transform(testDf)
    resutl.show(false)
    //    +-----------+--------+------+-------------------------------------------------------+------+------------------+-----+-----+----------------+-------+-----+--------+---------+------+--------+-------------+----------+-------------+------------------------------------------------------+---------------------------------------+----------------------------------------+----------+
    //    |PassengerId|Survived|Pclass|Name                                                   |Sex   |Age               |SibSp|Parch|Ticket          |Fare   |Cabin|Embarked|title    |mother|SexIndex|EmbarkedIndex|titleIndex|SurvivedIndex|Features                                              |rawPrediction                          |probability                             |prediction|
    //    +-----------+--------+------+-------------------------------------------------------+------+------------------+-----+-----+----------------+-------+-----+--------+---------+------+--------+-------------+----------+-------------+------------------------------------------------------+---------------------------------------+----------------------------------------+----------+
    //    |1          |0       |3     |Braund, Mr. Owen Harris                                |male  |22.0              |1    |0    |A/5 21171       |7.25   |null |S       |Mr       |0     |0.0     |0.0          |0.0       |0.0          |(9,[4,5,6,7],[3.0,22.0,7.25,1.0])                     |[45.092430520982006,4.907569479017978] |[0.9018486104196404,0.09815138958035959]|0.0       |
    //      |2          |1       |1     |Cumings, Mrs. John Bradley (Florence Briggs Thayer)    |female|38.0              |1    |0    |PC 17599        |71.2833|C85  |C       |Mr       |0     |1.0     |1.0          |0.0       |1.0          |[1.0,1.0,0.0,0.0,1.0,38.0,71.2833,1.0,0.0]            |[6.04311802019773,43.95688197980227]   |[0.12086236040395461,0.8791376395960454]|1.0       |
    //      |3          |1       |3     |Heikkinen, Miss. Laina                                 |female|26.0              |0    |0    |STON/O2. 3101282|7.925  |null |S       |Miss     |0     |1.0     |0.0          |1.0       |1.0          |[1.0,0.0,1.0,0.0,3.0,26.0,7.925,0.0,0.0]              |[19.06715968377948,30.93284031622052]  |[0.3813431936755896,0.6186568063244104] |1.0       |
    //      |4          |1       |1     |Futrelle, Mrs. Jacques Heath (Lily May Peel)           |female|35.0              |1    |0    |113803          |53.1   |C123 |S       |Mr       |0     |1.0     |0.0          |0.0       |1.0          |[1.0,0.0,0.0,0.0,1.0,35.0,53.1,1.0,0.0]               |[6.943773053856348,43.056226946143646] |[0.13887546107712698,0.861124538922873] |1.0       |
    //      |5          |0       |3     |Allen, Mr. William Henry                               |male  |35.0              |0    |0    |373450          |8.05   |null |S       |Mr       |0     |0.0     |0.0          |0.0       |0.0          |(9,[4,5,6],[3.0,35.0,8.05])                           |[44.11299708243986,5.8870029175601335] |[0.8822599416487974,0.11774005835120269]|0.0       |
    //      |6          |0       |3     |Moran, Mr. James                                       |male  |30.020306859205775|0    |0    |330877          |8.4583 |null |Q       |Mr       |0     |0.0     |2.0          |0.0       |0.0          |(9,[1,4,5,6],[2.0,3.0,30.020306859205775,8.4583])     |[44.20281939314409,5.797180606855912]  |[0.8840563878628818,0.11594361213711825]|0.0       |
    // 10.3获取auc面积值
    val auc: Double = evaluator.evaluate(resutl)

    // 10.4 将预测的结果打印出来。
    println(auc)
    //0.9034813946270305
  }
}

