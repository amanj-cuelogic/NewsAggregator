package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.log4j._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF

object NewsAggregatorDemo {
  
  def main(args: Array[String]) = {
    
    val spark = SparkSession
      .builder()
      .appName("NewsAggregatorDemo")
      .master("local[*]")
      .getOrCreate()
    
    val testDF = spark.createDataFrame(Seq(
      (0,"Narendra Modi government finally manages to break the Swiss bank black money vault"),
      (1,"Aadhaar mandatory for opening bank account, financial transactions of Rs 50000 and above"),
      (2,"Sensex closes at 3-week low, Nifty below 9600; Lupin, Cipla, Infosys, Wipro top losers"),
      (3,"HTC U11 Squeezable Smartphone launched in India for Rs 51990"),
      (4,"Micromax Bharat 2: Over half a million units sold in 50 days, claims company"),
      (5,"Life on Mars: Elon Musk reveals details of his colonisation vision"),
      (6,"China's Quantum Satellite Dispatches Transmissions Over a Record Distance of 1200 Kilometres"),
      (7,"Swine Flu On the Rise; Will There Be Another Epidemic? Home Remedies For Swine Flu"),
      (8,"Healthy hearts: Olive oil as good as statins in reducing cholesterol levels")
    )).toDF("id","sentence")
    
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val tokenizedDF = tokenizer.transform(testDF.na.drop())
    
    //tokenizedDF.show()
    
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(tokenizedDF)
    
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    
    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.show()
    
    val model = NaiveBayesModel.load("target/tmp/newsAggregatorBayesModel")
    val predictions = model.transform(rescaledData)
    
    predictions.show()
    
  }
}