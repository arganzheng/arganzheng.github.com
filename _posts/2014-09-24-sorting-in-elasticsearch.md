---
title: ElasticSearch字段排序
layout: post
---


需求
----

如果我们需要根据一个字段进行排序，如何支持呢？

像日期、价格这些数值型的（日期底层其实是转换为数值），排序其实是很自然也是很简单的。但是如果是字符串类型的字段呢？有几个问题需要考虑：

1. 分词对排序的影响
2. 大小写、单复数之类区别

字符串类型的字段一般都需要进行搜索的，那么需要对其进行Analyzed。但是Analyzed的结果就是将该字符串解析为分散的token(term)。这样子，ES就不知道到底是根据那个term来排序，事实上根据哪一个都是错的，应该是整个字符串。所以为了对字符串类型字段进行排序，需要保存整个字符串为一个term——也就是说保存一个not_analyzed string。同时为了保证该字段能够被搜索，还要保存该字符串analyzed的结果。

The naive approach to indexing the same string in two ways would be to include two separate fields in the document: one which is analyzed for searching, and one which is not_analyzed for sorting.

ES提供了一个便利的机制——[multi-field mapping](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/multi-fields.html)，可以让我们简单配置一下达到这个目的：

	"tweet": {  // 1
	    "type":     "string",
	    "analyzer": "english",
	    "fields": {
	        "raw": {  // 2
	            "type":  "string",
	            "index": "not_analyzed"
	        }
	    }
	}


1. The main tweet field is just the same as before: an analyzed full text field.
2. The new tweet.raw sub-field is not_analyzed.

现在，我们可以通过tweet字段作为搜索，而用tweet.raw字段作为排序：

	GET /_search
	{
	    "query": {
	        "match": {
	            "tweet": "elasticsearch"
	        }
	    },
	    "sort": "tweet.raw"
	}


但是依赖于额外的not_analyzed字段进行排序也有局限性——它只能完全匹配原始字符串，如果稍微有大小写、单复数之类的区别，会被单做不同的值的。如何处理这种情况呢？得益于Lucene Analyzer的pipeline机制，只要简单配置一下就可以了，比如使用如下配置可以组建一个emit a single lower case token的分析器:

	PUT /my_index
	{
	  "settings": {
	    "analysis": {
	      "analyzer": {
	        "case_insensitive_sort": {
	          "tokenizer": "keyword",    
	          "filter":  [ "lowercase" ] 
	        }
	      }
	    }
	  }
	}

然后我们可以这样子使用：

	PUT /my_index/_mapping/user
	{
	  "properties": {
	    "name": {
	      "type": "string",
	      "fields": {
	        "lower_case_sort": { // 1
	          "type":     "string",
	          "analyzer": "case_insensitive_sort"
	        }
	      }
	    }
	  }
	}

1. The name.lower_case_sort field will provide us with case-insentive sorting.


ES还提供了unicode sorting: [unicode collation algorithm](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/sorting-collations.html#uca)。

参考文档
--------

1. [string sorting and multi-fields](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/multi-fields.html)
2. [sorting and collations](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/sorting-collations.html)