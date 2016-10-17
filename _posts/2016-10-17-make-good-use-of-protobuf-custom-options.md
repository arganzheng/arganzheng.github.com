---
title: 巧用protobuf的自定义options
layout: post
---

知识图谱在线系统的schema除了定义每个实体的属性之外，还有一个信息需要策略同学告诉我们——就是哪些字段需要索引，以及索引的方式，这点非常类似于ElasticSearch的Mapping：

	PUT /my_index
	{
	  "mappings": {
	    "my_type": {
	      "properties": {
	        "status_code": {
	          "type": "string",
	          "index": "not_analyzed"
	        }
	      }
	    }
	  }
	}

mapping针对每个字段，有个index属性，可以指定索引的方式：

* no： Do not add this field value to the index. With this setting, the field will not be queryable.
* not_analyzed： Add the field value to the index unchanged, as a single term. This is the default for all fields that support this option except for string fields. not_analyzed fields are usually used with term-level queries for structured search.
* analyzed： This option applies only to string fields, for which it is the default. The string field value is first analyzed to convert the string into terms (e.g. a list of individual words), which are then indexed. At search time, the query string is passed through (usually) the same analyzer to generate terms in the same format as those in the index. It is this process that enables full text search.

ES使用了JSON作为schema，但是我们已经用了protobuf，能不能直接在protobuf的基础上直接增加这个信息呢？查看了一下protobuf的文档，Protobuf提供的[Custom Options](https://developers.google.com/protocol-buffers/docs/proto#customoptions)刚好可以解决这个问题。

我们可以自定义这么一个Field Option:

	package graphsearch;

	import "google/protobuf/descriptor.proto";

	enum IndexType {
		NO = 0;
		NOT_ANALYZED = 1;
		ANALYZED = 2;
	}

	extend google.protobuf.FieldOptions {
	    optional IndexType index = 51234 [default = NO];
	}


说明：

* Options分为file-level,message-level,field-level等几种，这里我们使用field-level的options。
* 自定义选项可以被定义为proto中的任意类型，如string, int32, enum, 甚至message。这里使用了枚举，而且默认值设置为NO，跟ES不一样。

然后就可以这么使用了：

	package graphsearch;

	import "schema_index.proto";

	message Person {
	    required string name = 1 [(index) = NO];
	    required uint32 age = 2;
	}

注意：使用这个选项的时候，选项名称必须被放置在()里，以表明这是一个扩展。

然后，我们可以这样读取这个option的值：

	Descriptors.Descriptor desc = Person::descriptor();
	List<Descriptors.FieldDescriptor> field_desc_list = desc.getFields();
	foreach(Descriptors.FieldDescriptor field_desc : field_desc_list){
		Descriptors.Descriptor options = field_desc.getOptions();
		Enum value = options.GetExtension(index);
		// ...
	}


参考文档
-------

1. [Elasticsearch Reference [2.4] » Mapping » Mapping parameters » index](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-index.html)
2. [Options](https://developers.google.com/protocol-buffers/docs/proto#options)
3. [Python Protocol Buffer field options](http://stackoverflow.com/questions/32836315/python-protocol-buffer-field-options)
4. [Descriptors.Descriptor](https://developers.google.com/protocol-buffers/docs/reference/java/com/google/protobuf/Descriptors.Descriptor)
5. [Descriptors.FieldDescriptor](https://developers.google.com/protocol-buffers/docs/reference/java/com/google/protobuf/Descriptors.FieldDescriptor)
6. [DescriptorProtos.FieldOptions](https://developers.google.com/protocol-buffers/docs/reference/java/com/google/protobuf/DescriptorProtos.FieldOptions)

