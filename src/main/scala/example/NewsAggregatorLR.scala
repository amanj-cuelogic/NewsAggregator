package example


import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer, HashingTF, IndexToString, StringIndexer, RegexTokenizer, IDF, StopWordsRemover}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.log4j._


object NewsAggregatorLR {
  
  def main(args: Array[String]) = {
    
    val spark = SparkSession
      .builder()
      .appName("NewsAggregatorLR")
      .master("local[*]")
      .getOrCreate()
      
    Logger.getLogger("org").setLevel(Level.ERROR)
    
   val newsFile = spark.read.option("header", "true").csv("uci-news-aggregator.csv").na.drop()

    val filteredVal = newsFile.filter(r => {
      r.getAs("CATEGORY").equals("b") || r.getAs("CATEGORY").equals("m") || r.getAs("CATEGORY").equals("e") || r.getAs("CATEGORY").equals("t") 
    }).na.drop()
    
    //filteredVal.show(300,false)
    
    val indexer = new StringIndexer().setInputCol("CATEGORY").setOutputCol("label")
    val indexed = indexer.fit(filteredVal).transform(filteredVal)
    
    
    val converter = new IndexToString().setInputCol("label").setOutputCol("categoryVal")
    val converted = converter.transform(indexed)
    
    converted.select("label", "categoryVal").distinct().show(100)
    //indexed.show()
    
    //println(indexed.count())
    
    //indexed.show()
    
    val tokenizer = new RegexTokenizer().setInputCol("TITLE").setOutputCol("rawWords").setPattern("\\W");
    val tokenizedDF = tokenizer.transform(indexed.na.drop(Array("TITLE")))
    
    //tokenizedDF.show()
    
    val stRemover = new StopWordsRemover().setInputCol("rawWords").setOutputCol("words")
    val stRemovedDF = stRemover.transform(tokenizedDF)
    
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(60000)
    val featurizedData = hashingTF.transform(stRemovedDF)
    
    //featurizedData.select("TITLE", "words","rawFeatures").show()
    
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    
    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.show()
    
    val Array(trainingData, testData) = rescaledData.randomSplit(Array(0.7,0.3),seed=1234L)
    
    val lr = new LogisticRegression()
      .setMaxIter(1000)
      .setRegParam(0.01)
      .setElasticNetParam(0.1)
      .setFeaturesCol("features")
      
    
    val lrModel = lr.fit(trainingData)
    
    println(s"Coefficients: \n${lrModel.coefficientMatrix}")
    println(s"Intercepts: \n${lrModel.interceptVector}")
    
    val predictions = lrModel.transform(testData)
    
    predictions.select("CATEGORY","label","probability","prediction").show(false)
    
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
      
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    
    val recallEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
      
    val accuracy = accuracyEvaluator.evaluate(predictions)
    println("Test set accuracy = "+accuracy)
    
    val precision = precisionEvaluator.evaluate(predictions)
    println("Test set Precision = "+precision)
    
    val recall = recallEvaluator.evaluate(predictions)
    println("Test set recall = "+recall)
    
    lrModel.write.overwrite().save("target/tmp/newsAggregatorLRModel")
    
    spark.stop()
  }
}