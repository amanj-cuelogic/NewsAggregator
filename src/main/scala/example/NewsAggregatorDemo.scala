package example

//import org.apache

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.LogisticRegressionModel


import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF

import org.apache.log4j._

object NewsAggregatorDemo {
  
  def main(args: Array[String]) = {
    
    val spark = SparkSession
      .builder()
      .appName("NewsAggregatorDemo")
      .master("local[*]")
      .getOrCreate()
    
    val testDF = spark.createDataFrame(Seq(
      (0,"GST won't make border check post disappear, not so soon", 1.0),
      (1,"Market Live: Sensex holds 150-pt gains, Nifty above 9600; RIL, ITC, HDFC twins lead",1.0),
      (2,"Baby born on Jet Airways plane gets free air tickets for life",1.0),
      (3,"UPSC exam tests students on GST, PM Modi's other pet schemes",1.0),
      (4,"Moto C Plus With 4000mAh Battery to Launch in India Today",2.0),
      (5,"OnePlus 5 Crosses 5.25 Lakh Registrations in China, TV Ad Outs Phone Ahead of Launch",2.0),
      (6,"Amazon starts Sale, offers big discounts on iPhone 6, iPhone SE, Moto phones and others",2.0),
      (7,"Jio Offers 20% More Data Under New Scheme. How To Get It",2.0),
      (8,"'Cars 3' zooms past 'Wonder Woman' with $53.5 million haul",0.0),
      (9,"Krushna Abhishek's partner in crime Sudesh Lehri to join him on new show, but Sunil Grover isn't yet confirmed",0.0),
      (10,"SRK And Suhana Steal The Show As B-Town Celebrities Attend Gauri's Bash",0.0),
      (11,"Delhi: Newborn declared dead by hospital, found to be alive just before burial",3.0),
      (12,"Meditation, yoga cut risk of cancer, depression by reversing DNA, says study",3.0),
      (13,"ACT therapy can relieve depression and anxiety symptoms in chronic pain patients",3.0),
      (14,"Eczema is likely not associated with an increase in cardiovascular diseases",3.0)
    )).toDF("id","sentence","actLabels")
    
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
    
    predictions.select("sentence","actLabels","prediction").show(false)
    
  }
}