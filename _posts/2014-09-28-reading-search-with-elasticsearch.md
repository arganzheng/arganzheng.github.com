---
title: Reading搜索
layout: post
---

### 需求

* 用户可以根据 isbn、作者和书名进行搜索。
	* 相关搜索：相关搜索关键词
	* 根据时间进行排序？
	* 根据状态进行排序？
	* 相似书籍（[more like this](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-more-like-this.html)）
	* 智能提示 [suggesters](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-suggesters.html)
		* auto-complete [completion suggester](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-suggesters-completion.html)
		* spell correction
			* term suggester
			* phrase suggster
2. 用户可以根据分类（tag）进行浏览，其实就是根据tag进行搜索
	* 根据出版日期排序

TODO：要不要提供 拥有者的搜索？还是拥有者走DB？

### 步骤

#### 一、构建索引

1. 新增分享的时候
2. 上下架
3. 


#### 二、搜索

GET /reading/book/_search
{
  "query": {
    "multi_match": {
      "query": "lucene",
      "fields": ["isbn10","isbn13","title", "subtitle", "authors"]
    }
  }
  , "from": 0
  , "size": 10
}


**TIPS**

通过简化需求达到降低业务复杂度和提高性能的作用——拥有关系在书籍详情页展示，而不是在list页面就展示。这样，list页面就只需要关注book表，而不需要关联book_owner_ship表。亚马逊的多卖家也是这样处理的。

从业务上讲，用户首先关注的是书，而并不关心谁拥有这本书。（当然，如果是书只是一个幌子就另当别论了。。。），如果他真的想查看某个拥有者的数据，可以通过拥有者进行搜索。

然后，书籍基本是不变的，但是该书籍的拥有关系却是可能会改变，比如新增、删除拥有关系，上下架、借出和归还。这样，就需要在更新DB拥有关系的同时，更新索引。这两种方式的构建索引和搜索实现成本都是不一样的。

但是这种方式带来一个问题就是不能根据用户分享的时间进行排序。比如我们想要让最新上架（分享）的书籍排在前面，特别在没有搜索的情况下。这种情况下必须将这个时间因子构建到索引中，要不是实现不了的。

#### 方案一：搜索引擎只构建书籍索引

一、构建索引

1. 新增索引：在书籍第一次分享的时候（调用豆瓣接口获取数据信息保存到DB的时候），将书籍信息构建到搜索引擎中。
2. 删除索引：在书籍的所有拥有关系删除之后删除该书籍。也可以不删除，影响就是可能搜索到这个书籍，但是发现没有人分享。

二、搜素

可以对数据的任何信息进行搜索。比如isbn、作者、书名。并且可以根据出版时间排序，等等。

但是由于没有拥有关系，所以有如下效果：

1. 无法按照拥有者进行搜索
2. 可能搜索出借阅不了的书籍（下架、借出、或者拥有关系已经被删除）。从业务上来说，已经下架的书籍是不应该被搜索出来的，其他状态被搜索出来是没有什么关系的。不过也不是很大问题。


#### 方案二：搜索引擎构建书籍和拥有关系索引

一、构建索引

1. 书籍的增删需要更新数据索引
	1. 新增索引：在书籍第一次分享的时候（调用豆瓣接口获取数据信息保存到DB的时候），将书籍信息构建到搜索引擎中。
	2. 删除索引：在书籍的所有拥有关系删除之后删除该书籍。也可以不删除，影响就是可能搜索到这个书籍，但是发现没有人分享。
2. 拥有关系的增删改也需要跟新索引
	1. 分享
	2. 上下架
	3. 借出
	4. 删除

另外，还需要对书籍和拥有关系进行建模。假设我们需要查询 某个用户分享的书籍。

这里有几种方式：

1. 完全照搬关系型数据库的做法，使用ID作为外键关联。但是搜索引擎没有join语句，所以需要在应用层进行。具体可以参见：[application-side joins](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/application-joins.html)。这个有个好处，就是订阅关系的修改和书籍的修改独立。不会互相影响。
2. 反范式或者宽表化，其实就是冗余。可以在拥有关系中，冗余书籍的信息，比如title。具体参见：[denormalizing your data](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/denormalization.html)。但是这里带来的坏处跟DB做冗余是一样的：记录变大 和 更新级联。不过这里书籍基本上是不会更新的。所以冗余一个不更新的字段是没有问题的。
3. [nested objects](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/nested-objects.html)
4. [parent-child relationship](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/parent-child.html)

后两者可以达到通过child(nested)对象查找parents对象的效果。在这里例子中，也就是通过拥有关系查询书籍的作用。

具体参见: [managing relations inside elasticsearch](http://www.elasticsearch.org/blog/managing-relations-inside-elasticsearch/)

但是这些还是有缺点的。

另一种做法是反范式，把需要搜索的字段冗余过来。比如这里的书籍的owners和newest_share_time，还有read_count。但是缺点就是需要跟新冗余字段，可能涉及到索引重建。ES支持[partial update](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/partial-updates.html)。但是如果冗余字段变更频繁，就不是很合适了。不过这种需要参与搜索的字段变更频繁的情况即使是nested object和parent-child，也是有同样的问题。相对于说反范式为宽表更简单一些。

另外，对于tags和authors等array类型字段，可以使用script来进行操作。这样可以避免需要先获取和并发覆盖：

	# Partial update with a script and params
	POST /website/blog/1/_update
	{
	   "script" : "ctx._source.tags+=new_tag",
	   "params" : {
	      "new_tag" : "search"
	   }
	}

	# Retrieve the updated doc
	GET /website/blog/1


	# Partial update with a script and params
	POST /website/blog/1/_update
	{
	   "script" : "ctx._source.tags.remove(old_tag)",
	   "params" : {
	      "old_tag" : "search"
	   }
	}

不过要先开启dynamic script功能，在elasticsearch.yml中配置：

	script.disable_dynamic: false


还有一种折中：list页面在没有搜索的情况下走DB，搜索的时候走ES。这样DB可以自由的join来控制展示逻辑。但是DB的性能是要比ES差一些的。特别在负责SQL的情况下。

总结一下，我们需要索引的字段有：

* isbn
* title
* subtitle
* authors
* tags?

需要展示的字段有

* pubdate
* pageCount
* tags
* imageUrl

**TIPS** 使用[multi field type](http://www.elasticsearch.org/guide/en/elasticsearch/reference/0.90/mapping-multi-field-type.html)对一个字段进行多次mapping。具体参见：[Mapping WordPress Posts to Elasticsearch](http://gibrown.wordpress.com/2013/04/17/mapping-wordpress-posts-to-elasticsearch/)。一般用于有统计需求的字段，比如tags, author等。

mapping可以这么定义：

	PUT /reading
	{
	  "mappings": {
	    "book": {
	      "date_detection": false,
	      "properties": {
	      	"isbn10": {
	      		"type": "string",
	      		"index": "not_analyzed"		
	      	},
	      	"isbn13": {
	      		"type": "string",
	      		"index": "not_analyzed"			      		
	      	},
	        "title": {
	          "type": "string", 
	          "analyzer": "ik"  
		  	  },
		  	"subtitle":{
		  	    "type": "string", 
	          	"analyzer": "ik" 
		  	},
			"image": {
	      		"type": "string",
	      		"index": "not_analyzed"			      		
	      	},
		  	"authors":{
		  	    "type": "string", 
				"analyzer": "ik",
		  	    "index_name": "author"
		  	  },
		  	"pubdate": {
	          "type": "date",
	          "format" : "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"
	        },
	        "tags":{ 
	          "type": "string",
			  "analyzer": "ik", 
	          "index_name": "tag"	          
	        },
	        "pageCount":{ 
	          "type": "string",
	          "index": "not_analyzed"
	        },
			"postDate": {
	          "type": "date",
	          "format" : "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"
	        }
		  }
	    }
	  }
	}

**TIPS**

这里引文分词器用了ik插件，安装完成之后需要重启ES。




可以这样测试：

GET http://localhost:9200/reading/_analyze?pretty=true
中华人民共和国万岁

GET http://localhost:9200/reading/_analyze?pretty=true&analyzer=standard "中华人民共和国万岁"

支持我们的搜索：

	GET /reading/book/_search
	{
	  "query": {
	    "multi_match": {
	      "query": "全文",
	      "fields": ["isbn10", "isbn13", "title", "subtitle", "authors", "tags"]
	    }
	  }
	}