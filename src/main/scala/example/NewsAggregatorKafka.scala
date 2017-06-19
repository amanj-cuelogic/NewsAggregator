package example

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer

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

object NewsAggregatorKafka {

  case class NewsTweet(key:String, value: String)

  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("NewsAggregatorKafka").setMaster("local[*]")
    val ssc = new StreamingContext(conf, Seconds(5))
    

    Logger.getLogger("org").setLevel(Level.ERROR)

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> "localhost:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "use_a_separate_group_id_for_each_stream",
      "auto.offset.reset" -> "latest",
      "enable.auto.commit" -> (false: java.lang.Boolean))

    val tweets = KafkaUtils.createDirectStream[String, String](ssc, PreferConsistent, Subscribe[String, String](Array("testTopic"), kafkaParams))

    val tweetsStream = tweets.map(record => NewsTweet(record.key(), record.value()))
    
    
    ssc.start()
    ssc.awaitTermination()
    
    //    val tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words")
    //    val tokenizedDF = tokenizer.transform(newsDF.na.drop())
    //    
    //    //tokenizedDF.show()
    //    
    //    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20000)
    //    val featurizedData = hashingTF.transform(tokenizedDF)
    //    
    //    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    //    val idfModel = idf.fit(featurizedData)
    //    
    //    val rescaledData = idfModel.transform(featurizedData)
    //    //rescaledData.show()
    //    
    //    val model = NaiveBayesModel.load("target/tmp/newsAggregatorBayesModel")
    //    val predictions = model.transform(rescaledData)
    //    
    //    predictions.select("sentence","actLabels","prediction").show(false)

  }
}