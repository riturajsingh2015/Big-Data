import static org.apache.spark.sql.functions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.*;

/**
 * https://spark.apache.org/docs/latest/api/java/index.html
 */
public class WikipediaPageRank {
	public static void main(String[] args) {
		SparkSession spark = SparkSession
			.builder()
			.appName("PageRank")
			//.config("spark.master", "local") // only for local debug
			.getOrCreate();

		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
		jsc.setLogLevel("ERROR");
		
		spark.sparkContext().setLogLevel("ERROR");
		spark.sparkContext().setCheckpointDir("checkpoint");

		StructType linkSchema = new StructType()
				.add("source", DataTypes.StringType)
				.add("target", DataTypes.StringType);

		Pattern wikilinksPattern = Pattern.compile("\\[\\[(.*?)[\\|#\\]].*?\\]?\\]");

		Dataset<Row> wikipediaDF = spark.read()
				  .format("xml")
				  .option("rowTag", "page")
				  .option("mode", "FAILFAST")
				  .load("wikipedia.xml");
		
		Dataset<Row> redirections = wikipediaDF.filter("redirect._title IS NOT NULL").select(col("title"),col("redirect._title").as("redirectTo"));
		Dataset<Row> articles = wikipediaDF.filter("ns == '0'").select(col("title"),col("revision.text").as("text"));

		Dataset<Row> links = articles.flatMap((FlatMapFunction<Row, Row>) row -> {
						String text = (String) ((Row)row.get(1)).get(0);
						
						// find all linkings to other pages
						List<Row> linksTo = new ArrayList<>();
						Matcher wikilinksMatcher = wikilinksPattern.matcher(text);
						while(wikilinksMatcher.find()) {
							linksTo.add(RowFactory.create(row.get(0), wikilinksMatcher.group(1)));
						}
						if(linksTo.isEmpty()) {
							// add the actual page which links to no one
							linksTo.add(RowFactory.create(row.get(0), null));
						}
					
					return linksTo.iterator();
				}, RowEncoder.apply(linkSchema));

		// Weiterleitungen auflösen
		links = links
			.join(redirections, col("title").equalTo(col("target")),"left")
			.select(col("source"), (when(col("redirectTo").isNull(),col("target")).otherwise(col("redirectTo")).as("target")))
			.join(redirections, col("source").equalTo(col("title")),"left_anti");
		// Weiterleitungsartikel entfernen
		articles = articles.join(links, col("source").equalTo(col("title")), "semi");

		// Nicht existente Seiten filtern.
		links = links.join(articles, col("title").equalTo(links.col("target")), "semi");

		// PageRank
		final double d = 0.85;
		
		Dataset<Row> pages = articles.select(col("title").as("page"))
			.withColumn("rank", lit(1.0))
			.cache();

		links = links.groupBy("source")
			.agg(collect_list("target"))
			.cache();

		StructType resultSchema = new StructType()
			.add("target", DataTypes.StringType)
			.add("rank", DataTypes.DoubleType);

		Dataset<Row> old_cached_pages = pages;
		Dataset<Row> cached_pages;
		for(int i = 0; i < 30; i++) {

			Dataset<Row> contrib = links.join(pages, col("source").equalTo(col("page")))
				.flatMap((FlatMapFunction<Row, Row>) row -> {
					List<String> targets = row.<String>getList(1);
					double rank = row.<Double>getAs("rank") / targets.size();

					return targets.stream()
						.map(s -> RowFactory.create(s, rank))
						.collect(Collectors.toList())
						.iterator();
				}, RowEncoder.apply(resultSchema))
				.groupBy("target")
				.sum("rank");

			cached_pages = pages.join(contrib, col("page").equalTo(col("target")), "left_outer")
				.select(col("page"), when(col("sum(rank)").isNotNull(), col("sum(rank)")).otherwise(0).multiply(d).plus(1.0 - d).as("rank"))
				.cache();
			
			pages = cached_pages.checkpoint();

			old_cached_pages.unpersist();
			old_cached_pages = cached_pages;
		}

		pages.sort(desc("rank"))
			.write()
			.option("sep", "\t")
			.csv("result");
	}
}
