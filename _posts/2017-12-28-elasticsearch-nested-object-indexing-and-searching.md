---
layout: post
title: ElasticSearch如何支持嵌套属性检索
catalog: true
tags: [elasticsearch, 索引]
---


地球是圆的，对象往往也不是扁平的。所以嵌套属性就很常见了，比如`name.first`, `address.city`, etc. 

我们知道ElasticSearch是Document Oriented的NoSQL，文档本身就是JSON表示，JSON本身就是一个自由嵌套的数据结构(JSON documents are hierarchical in nature: the document may contain inner objects which, in turn, may contain inner objects themselves.)。那么ElasticSearch支持嵌套属性的索引和检索吗？让我们深入了解一下。


### [Array dataType](https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html)

事实上ES没有专门的数组类型，因为默认value就可以有一个或者多个值。但是只有一个要求，就是数组元素的类型必须保持一致，如:

* an array of strings: [ "one", "two" ]
* an array of integers: [ 1, 2 ]
* an array of arrays: [ 1, [ 2, 3 ]] which is the equivalent of [ 1, 2, 3 ]
* an array of objects: [ { "name": "Mary", "age": 12 }, { "name": "John", "age": 10 }]

When adding a field dynamically, the first value in the array determines the field type. All subsequent values must be of the same datatype or it must at least be possible to coerce subsequent values to the same datatype.

不需要提前或者显示指定数组类型：

```
PUT my_index/my_type/1
{
  "message": "some arrays in this document...",
  "tags":  [ "elasticsearch", "wow" ], 
  "lists": [ 
    {
      "name": "prog_list",
      "description": "programming list"
    },
    {
      "name": "cool_list",
      "description": "cool stuff list"
    }
  ]
}

PUT my_index/my_type/2 
{
  "message": "no arrays in this document...",
  "tags":  "elasticsearch",
  "lists": {
    "name": "prog_list",
    "description": "programming list"
  }
}

GET my_index/_search
{
  "query": {
    "match": {
      "tags": "elasticsearch" 
    }
  }
}
```

上面两个文档都会被检索到。

> ### Multi-value fields and the inverted index
> 
> The fact that all field types support multi-value fields out of the box is a consequence of the origins of Lucene. Lucene was designed to be a full text search engine. In order to be able to search for individual words within a big block of text, Lucene tokenizes the text into individual terms, and adds each term to the inverted index separately.
> 
> This means that even a simple text field must be able to support multiple values by default. When other datatypes were added, such as numbers and dates, they used the same data structure as strings, and so got multi-values for free.


### [Object Type](https://www.elastic.co/guide/en/elasticsearch/reference/current/object.html)


下面的对象：

```
PUT my_index/my_type/1
{ 
  "region": "US",
  "manager": { 
    "age":     30,
    "name": { 
      "first": "John",
      "last":  "Smith"
    }
  }
}
```

但是Lucene底层的索引接口其实就是针对简单的Key-value键值对的，所以ES内部会将其平坦化，转成下面的key-value对进行索引构建：

```
{
  "region":             "US",
  "manager.age":        30,
  "manager.name.first": "John",
  "manager.name.last":  "Smith"
}
```

显示的mapping定义大概如下：

```
PUT my_index
{
  "mappings": {
    "my_type": { 
      "properties": {
        "region": {
          "type": "keyword"
        },
        "manager": { 
          "properties": {
            "age":  { "type": "integer" },
            "name": { 
              "properties": {
                "first": { "type": "text" },
                "last":  { "type": "text" }
              }
            }
          }
        }
      }
    }
  }
}
```


### [Nested datatype](https://www.elastic.co/guide/en/elasticsearch/reference/current/nested.html)

嵌套类型(nested type)是对象类型(object type)的一个特例。它主要是为了解决数组对象的索引问题。

例如下面的文档：

```
PUT my_index/my_type/1
{
  "group" : "fans",
  "user" : [ 
    {
      "first" : "John",
      "last" :  "Smith"
    },
    {
      "first" : "Alice",
      "last" :  "White"
    }
  ]
}
```

user是一个数组对象。按照前面的Object Type说明，ES会将其平坦化为如下键值对：

```
{
  "group" :        "fans",
  "user.first" : [ "alice", "john" ],
  "user.last" :  [ "smith", "white" ]
}
```

这样`user.first`和`user.last`字段被平坦化成多值字段(数组)，这样有一个问题，就是 alice 和 white 本来合起来是一个人的全面，平坦化之后这个信息就丢失了。这会导致如果我们搜索 'user.first=alice AND user.last=smith' 的语句也是能够检索到这个文档的：

```
GET my_index/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "user.first": "Alice" }},
        { "match": { "user.last":  "Smith" }}
      ]
    }
  }
}
```

为了解决这个问题，ES引入了Nested 类型来解决数组对象(arrays of objects)的索引问题。内部实现上，ES会把nested类型对象作为一个单独的隐藏文档进行索引，这意味着每一个嵌套对象都可以被单独的检索，数组中的元素不会互相干扰。

> Internally, nested objects index each object in the array as a separate hidden document, meaning that each nested object can be queried independently of the others, with the nested query:

```
PUT my_index
{
  "mappings": {
    "my_type": {
      "properties": {
        "user": {
          "type": "nested" 
        }
      }
    }
  }
}
```

这样再次查询 'user.first=alice AND user.last=smith' 就检索不到了，因为Alice and Smith are not in the same nested object。

但是要注意的是这种做法会导致索引的文档数倍增：索引一个有100个嵌套字段的文档实际上是构建101个文档，因为每一个嵌套属性事实上都是单独隐藏文档。另外，索引和查询语法都很奇怪。所以，如果不是嵌套数组对象，不要使用这个类型。


参考文档
-------

1. [Elasticsearch Reference [6.1] » Mapping » Field datatypes » Array datatype](https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html)
2. [Elasticsearch Reference [6.1] » Mapping » Field datatypes » Object datatype](https://www.elastic.co/guide/en/elasticsearch/reference/current/object.html)
3. [Elasticsearch Reference [6.1] » Mapping » Field datatypes » Nested datatype](https://www.elastic.co/guide/en/elasticsearch/reference/current/nested.html)
4. [Elasticsearch: The Definitive Guide [2.x] » Modeling Your Data » Nested Objects » Querying a Nested Object](https://www.elastic.co/guide/en/elasticsearch/guide/current/nested-query.html)

