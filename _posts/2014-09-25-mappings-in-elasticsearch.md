---
title: ElasticSearch的mappings
layout: post
---


在ES中，每个文档都会归属到一个类型(type)下。这个类型相当于关系型数据库中的表结构。所以每个类型有类似于DB一样的schema定义，称之为mapping。mapping定义了type中的fields信息：

* datatype for each field，具体参见前面的文章[ElasticSearch的数据类型](http://blog.arganzheng.me/posts/datatype-in-elasticsearch.html)。
* how the field should be handled by ES

大部分情况下，我们是不需要对type进行自定义mapping的，但是在某些情况下我们需要自定义mapping：

* distinguish between full text string fields and exact value string fields,
* use language-specific analyzers,
* optimize a field for partial matching,
* specify custom date formats,
* and much more.

对于非string类型的字段，基本上你只需要配置type配置项就可以了：

	{
	    "number_of_clicks": {
	        "type": "integer"
	    }
	}

对于string类型的字段，一般都是参与到全文搜索中。That is, their value will be passed through an analyzer before being indexed and a full text query on the field will pass the query string through an analyzer before searching. 所以，对于String类型字段最重要的配置是index和analyzer。这两个配置项控制着索引和搜索的行为：

**index**

* analyzed: First analyze the string, then index it. In other words, index this field as full text.
* not_analyzed: Index this field, so it is searchable, but index the value exactly as specified. Do not analyze it.
* no: Don’t index this field at all. This field will not be searchable.

默认是analyzed。如果需要原样保留值，可以把它设置为not_analyzed：

	{
	    "tag": {
	        "type":     "string",
	        "index":    "not_analyzed"
	    }
	}

**NOTE** 

其他的简单类型，比如long, double, date 等，也可以配置index参数，但是只能够配置no 和 not_analyzed，因为它们的值是不可能被分析的。

**analyzer**

对于需要分析的string类型的字段，可以通过analyzer配置项指定使用的分析器：

	{
	    "tweet": {
	        "type":     "string",
	        "analyzer": "english"
	    }
	}

还可以分别指定index time和search time使用不同的分析器：

	PUT /my_index/my_type/_mapping
	{
	    "my_type": {
	        "properties": {
	            "name": {
	                "type":            "string",
	                "index_analyzer":  "autocomplete", 
	                "search_analyzer": "standard" 
	            }
	        }
	    }
	}

一般情况下，索引时使用的分析器和搜索时使用的分析器应该是一样的。但是有些时候，确实会需要不同的分析器。比如，在构建索引的时候我们可能需要对同义词(synonyms)进行索引。但是在搜索的时候，我们并不需要一次性搜索所有的同义词。另一种常见的场景就是实现search-as-you-type功能，需要在构建索引期间使用edge_ngram过滤器[index time search-as-you-type](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/_index_time_search_as_you_type.html)。


	PUT /my_index
	{
	    "settings": {
	        "number_of_shards": 1, 
	        "analysis": {
	            "filter": {
	                "autocomplete_filter": { 
	                    "type":     "edge_ngram",
	                    "min_gram": 1,
	                    "max_gram": 20
	                }
	            },
	            "analyzer": {
	                "autocomplete": {
	                    "type":      "custom",
	                    "tokenizer": "standard",
	                    "filter": [
	                        "lowercase",
	                        "autocomplete_filter" 
	                    ]
	                }
	            }
	        }
	    }
	}

	PUT /my_index/my_type/_mapping
	{
	    "my_type": {
	        "properties": {
	            "name": {
	                "type":            "string",
	                "index_analyzer":  "autocomplete", 
	                "search_analyzer": "standard" 
	            }
	        }
	    }
	}

但是这个会增加索引的大小和构建速度，因为一个词被拆分成多个了。

[controlling analysis-default analyzers](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/_controlling_analysis.html#_default_analyzers)

