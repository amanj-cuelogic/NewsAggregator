package com.example

//import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StringIndexer, IndexToString}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.log4j._

object NewsAggregator {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
        .builder()
        .master("local[*]")
        .appName("NewsAggregator")
        .getOrCreate()

    Logger.getLogger("org").setLevel(Level.ERROR)
    
   val newsFile = spark.read.option("header", "true").csv("uci-news-aggregator.csv").na.drop()

    val filteredVal = newsFile.filter(r => {
      r.getAs("CATEGORY").equals("b") || r.getAs("CATEGORY").equals("m") || r.getAs("CATEGORY").equals("e") || r.getAs("CATEGORY").equals("t") 
    }).na.drop()
    
    //filteredVal.show(300,false)
    
    val indexer = new StringIndexer().setInputCol("CATEGORY").setOutputCol("label")
    val indexed = indexer.fit(filteredVal).transform(filteredVal)
    
//    
    val converter = new IndexToString().setInputCol("label").setOutputCol("categoryVal")
    val converted = converter.transform(indexed)
//    
    //converted.select("label", "categoryVal").distinct().show(100)
    //indexed.show()
//    
    println(indexed.count())
    
    //indexed.show()
    
    val tokenizer = new RegexTokenizer().setInputCol("TITLE").setOutputCol("words").setPattern("\\W");
    val tokenizedDF = tokenizer.transform(indexed.na.drop(Array("TITLE")))
    
    //tokenizedDF.show()
    
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(tokenizedDF)
    
    //featurizedData.select("TITLE", "words","rawFeatures").show()
    
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    
    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.select("TITLE", "words","rawFeatures","features").show()
    
    val Array(trainingData, testData) = rescaledData.randomSplit(Array(0.7,0.3),seed=1234L)
    
    val model = new NaiveBayes().fit(trainingData)
    
    val predictions = model.transform(testData)
    
    //predictions.show()
    
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
      
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = "+accuracy)
      
    model.write.overwrite().save("target/tmp/newsAggregatorBayesModel")
    
    spark.stop()
    
  }
}