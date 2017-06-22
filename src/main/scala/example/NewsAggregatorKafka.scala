package example

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe

import org.apache.log4j._
import org.apache.spark.streaming._
import scala.collection.mutable.Map
import org.apache.spark.SparkConf
import org.apache.spark.sql.cassandra._
import com.datastax.spark.connector._
import com.datastax.spark.connector.cql._
import org.apache.spark.ml.feature.{RegexTokenizer,StopWordsRemover}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vectors
import scala.collection.immutable.Vector

object NewsAggregatorKafka {

  case class NewsTweet(key: String, value: String)

  var RegexList = Map[String, String]()
  RegexList += ("punctuation" -> "[^a-zA-Z0-9]")
  RegexList += ("digits" -> "\\b\\d+\\b")
  RegexList += ("white_space" -> "\\s+")
  RegexList += ("small_words" -> "\\b[a-zA-Z0-9]{1,2}\\b")
  RegexList += ("urls" -> "(https?\\://)\\S+")

  def removeRegex(txt: String, flag: String) = {
    val regex = RegexList.get(flag)
    var cleaned = txt
    regex match {
      case Some(value) =>
        if (value.equals("white_space")) cleaned = txt.replaceAll(value, "")
        else cleaned = txt.replaceAll(value, " ")

      case None => println("No regex flag matches")
    }
    cleaned
  }

  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("NewsAggregatorKafka").setMaster("local[*]")
    val ssc = new StreamingContext(conf, Seconds(5))

    Logger.getLogger("org").setLevel(Level.ERROR)

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> "localhost:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "newsAggregator",
      "connection.max.idle.ms" -> "54000",
      "retry.backoff.ms" -> "10000")

    val tweets = KafkaUtils
      .createDirectStream[String, String](
        ssc,
        PreferConsistent,
        Subscribe[String, String](
          Array("news"),
          kafkaParams))

    val tweetsDS = tweets.map(record => (record.value()))

    val cleanData: (String => String) = (arg: String) => {
      
      var text = arg
      text = removeRegex(text, "urls")
      text = removeRegex(text, "punctuation")
      text = removeRegex(text, "digits")
      text = removeRegex(text, "small_words")
      text = removeRegex(text, "white_space")
      
      text

    }

    val dataCleaner = udf(cleanData)
    
    tweetsDS.foreachRDD(rdd => {

      val spark = SparkSession.builder().config(rdd.sparkContext.getConf).getOrCreate()

      import spark.implicits._

      val newsDF = rdd.toDF("sentence")

      val filterednewsDF = newsDF.withColumn("filteredSentence", dataCleaner('sentence))
      
      val tokenizer = new Tokenizer().setInputCol("filteredSentence").setOutputCol("rawWords")
      val tokenizedDF = tokenizer.transform(filterednewsDF.na.drop())

      //tokenizedDF.show(false)
      
      val stRemover = new StopWordsRemover().setInputCol("rawWords").setOutputCol("words")
      val stRemovedDF = stRemover.transform(tokenizedDF)

      val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(60000)
      val featurizedData = hashingTF.transform(stRemovedDF)

      val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
      val idfModel = idf.fit(featurizedData)

      val rescaledData = idfModel.transform(featurizedData)
      //rescaledData.show()

      val model = LogisticRegressionModel.load("target/tmp/newsAggregatorLRModel")
      val predictions = model.transform(rescaledData)

      //predictions.printSchema()
      
//      val calculateProb : (Vector[Double] => Double) = (arg: Vector[Double]) => {
//          
//         val prob = arg.reduce((x,y)  => x + y)
//         prob
//      }
//      
//      val probCalculator = udf(calculateProb)
      
      //val refinedPredictions = predictions.withColumn("cummProb", probCalculator('probability))
      
      
      predictions.select("sentence", "probability","prediction").show(false)

      predictions.select("sentence", "prediction" ,"probability").write
        .mode("append")
        .format("org.apache.spark.sql.cassandra")
        .options(Map("table" -> "newsclassification", "keyspace" -> "news"))
        .save()

    })

    ssc.start()
    ssc.awaitTermination()

  }
}