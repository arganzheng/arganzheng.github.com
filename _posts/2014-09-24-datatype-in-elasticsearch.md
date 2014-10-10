---
title: ElasticSearch的数据类型
layout: post
---

ES使用JSON作为数据交互格式，所以简单来说，JSON支持的数据类型，ES都支持。

### 1. core simple field types

* String: string
* Whole number: byte, short, integer, long
* Floating point: float, double
* Boolean: boolean
* Date: date


ES对于没有定义mapping的未知字段会采用[dynamic mapping](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/dynamic-mapping.html)进行类型猜测，并且将其加入到type mapping中。这个行为可以通过 dynamic 配置项进行控制。

* true: Add new fields dynamically — the default
* false: Ignore new fields
* strict: Throw an exception if an unknown field is encountered

相应的mappings配置如下：

	PUT /my_index
	{
	    "mappings": {
	        "my_type": {
	            "dynamic":      "strict", 
	            "properties": {
	                "title":  { "type": "string"},
	                "stash":  {
	                    "type":     "object",
	                    "dynamic":  true 
	                }
	            }
	        }
	    }
	}


**TIPS** 日期格式

JSON并没有日期类型，日期是以特定字符串格式形式表示。比如`postDate: "2014-09-24"`。ES能够根据字符串的格式猜测出这是一个Date类型。日期字符串格式可以通过format和dynamic_date_formats配置项指定（[date-format](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/mapping-date-format.html)），而且支持多个格式如：yyyy/MM/dd HH:mm:ss||yyyy/MM/dd。

	{
	    "tweet" : {
	        "_timestamp" : {
	            "enabled" : true,
	            "path" : "post_date",
	            "format" : "YYYY-MM-dd",
	            "default" : "1970-01-01"
	        }
	    }
	}


可以通过date_detection配置项单独关闭日期类型检查：

	curl -XPUT "http://localhost:9200/myindex" -d
	{
	   "mappings": {
	      "tweet": {
	         "date_detection": false
	      }
	   }
	}


### 2. [complex core field types](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/complex-core-fields.html)

除了简单的基本类型，ES还支持如下复杂数据类型：


####  multi-value fields

对应JSON中的数组。如tag

	{ "tag": [ "search", "nosql" ]}

但是事实上ES并没有array类型，因为默认就是支持的。具体参见[array type](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/mapping-array-type.html#mapping-array-type)。

####  empty fields

* empty string: ""
* null value: null
* empty array: []
* array_with_null_value: [null]

#### multi-level objects

对应JSON中的嵌套对象，例如：

	{
	    "tweet":            "Elasticsearch is very flexible",
	    "user": {
	        "id":           "@johnsmith",
	        "gender":       "male",
	        "age":          26,
	        "name": {
	            "full":     "John Smith",
	            "first":    "John",
	            "last":     "Smith"
	        }
	    }
	}

ES会自动检测和映射为object类型，也可以通过如下mapping配置：

	{
	  "gb": {
	    "tweet": { 
	      "properties": {
	        "tweet":            { "type": "string" },
	        "user": { 
	          "type":             "object",
	          "properties": {
	            "id":           { "type": "string" },
	            "gender":       { "type": "string" },
	            "age":          { "type": "long"   },
	            "name":   { 
	              "type":         "object",
	              "properties": {
	                "full":     { "type": "string" },
	                "first":    { "type": "string" },
	                "last":     { "type": "string" }
	              }
	            }
	          }
	        }
	      }
	    }
	  }
	}


**NOTES && TIPS** how inner objects are indexed

Lucene并没有嵌套对象的概念，事实上，A Lucene document consists of a flat list of key-value pairs. Lucene only indexes scalar or simple values, not complex datastructures. ES其实是做了如下平坦化处理：

	{
	    "tweet":            [elasticsearch, flexible, very],
	    "user.id":          [@johnsmith],
	    "user.gender":      [male],
	    "user.age":         [26],
	    "user.name.full":   [john, smith],
	    "user.name.first":  [john],
	    "user.name.last":   [smith]
	}

#### arrays of inner objects

Finally, consider how an array containing inner objects would be indexed. Let’s say we have a followers array which looks like this:

	{
	    "followers": [
	        { "age": 35, "name": "Mary White"},
	        { "age": 26, "name": "Alex Jones"},
	        { "age": 19, "name": "Lisa Smith"}
	    ]
	}

This document will be flattened as we described above, but the result will look like this:

	{
	    "followers.age":    [19, 26, 35],
	    "followers.name":   [alex, jones, lisa, smith, mary, white]
	}

这其实是有问题的，ES提供了一个称之为[nested objects](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/nested-objects.html)的解决方案。比较恶心，这里不讨论。

