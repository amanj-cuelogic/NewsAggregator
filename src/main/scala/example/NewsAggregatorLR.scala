package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{ Tokenizer, HashingTF, IndexToString, StringIndexer, RegexTokenizer, IDF }
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.log4j._
import org.apache.spark.sql.Row

object NewsAggregatorLR {

  def main(args: Array[String]) = {

    val spark = SparkSession
      .builder()
      .appName("NewsAggregatorLR")
      .master("local[*]")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.ERROR)

    val newsFile = spark.read.option("header", "true").csv("uci-news-aggregator.csv").na.drop()

    val businessRDD = spark.sparkContext.textFile("./bbc/business/*.txt")

    val businessLabel = businessRDD.filter(x => x.nonEmpty).map(x => Row(x, "b"))

    val entertainmentRDD = spark.sparkContext.textFile("./bbc/entertainment/*.txt")

    val entertainmentLabel = entertainmentRDD.filter(x => x.nonEmpty).map(x => Row(x, "e"))

    val politicsRDD = spark.sparkContext.textFile("./bbc/politics/*.txt")

    val politicsLabel = politicsRDD.filter(x => x.nonEmpty).map(x => Row(x, "p"))

    val sportRDD = spark.sparkContext.textFile("./bbc/sport/*.txt")

    val sportLabel = sportRDD.filter(x => x.nonEmpty).map(x => Row(x, "s"))

    val techRDD = spark.sparkContext.textFile("./bbc/tech/*.txt")

    val techLabel = techRDD.filter(x => x.nonEmpty).map(x => Row(x, "t"))

    val newsRDD = businessLabel.union(entertainmentLabel).union(politicsLabel).union(politicsLabel).union(sportLabel).union(techLabel)

    val filteredVal = newsFile.filter(r => {
      r.getAs("CATEGORY").equals("b") || r.getAs("CATEGORY").equals("m") || r.getAs("CATEGORY").equals("e") || r.getAs("CATEGORY").equals("t")
    }).na.drop()

    val tweetsRDD = filteredVal.select("TITLE", "CATEGORY").rdd.union(newsRDD)
    
    val tweetsDF = spark.createDataFrame(tweetsRDD.map(r => (r.get(0).toString(),r.get(1).toString()))).toDF("sentence","category")
    
    //tweetsDF.show()
    //filteredVal.show(300,false)

    val indexer = new StringIndexer().setInputCol("category").setOutputCol("label")
    val indexed = indexer.fit(tweetsDF).transform(tweetsDF)
    
    val converter = new IndexToString().setInputCol("label").setOutputCol("categoryVal")
    val converted = converter.transform(indexed)

    //converted.select("label", "categoryVal").distinct().show(100)
    
    //indexed.show()

    //println(indexed.count())

    //indexed.show()

    val tokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\W");
    val tokenizedDF = tokenizer.transform(indexed.na.drop(Array("sentence")))

    //tokenizedDF.select("words").show(false)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(tokenizedDF)

    //featurizedData.select("TITLE", "words","rawFeatures").show()

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.show()

    val Array(trainingData, testData) = rescaledData.randomSplit(Array(0.7, 0.3), seed = 1234L)

    val lr = new LogisticRegression()
      .setMaxIter(1000)
      .setRegParam(0.01)
      .setElasticNetParam(0.1)
      .setFeaturesCol("features")

    val lrModel = lr.fit(trainingData)

    println(s"Coefficients: \n${lrModel.coefficientMatrix}")
    println(s"Intercepts: \n${lrModel.interceptVector}")

    val predictions = lrModel.transform(testData)

    //predictions.select("category", "label", "probability", "prediction").show(false)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)

    lrModel.write.overwrite().save("target/tmp/newsAggregatorLRModel")

    spark.stop()
  }
}