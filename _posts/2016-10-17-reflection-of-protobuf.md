---
title: protobuf中的反射
layout: post
---

像Python，javascript这样的动态脚本语言具有很强的反射能力，即使是java这类的静态类型语言，也提供了一定程度的反射能力，而C/C++这样的纯静态语言没有运行时的类型信息，但是Protobuf本身具有很强的反射(reflection)功能，可以根据 type name 创建具体类型的 Message 对象，也能够动态获取和设置某个属性。

背景知识
-------

![protobuf_classdiagram.png](/media/images/protobuf_classdiagram.png)

简单来说，protobuf对于每个message都有一个相应的descriptor，这个descriptor包含该message的所有元信息。我们可以根据这个descriptor创建这个相应的message实例。

关键类：

* [Descriptor](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor#Descriptor): Describes a type of protocol message, or a particular group within a message.
* [DescriptorPool](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor#DescriptorPool): Used to construct descriptors.
* [MessageFactory](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#MessageFactory): Abstract interface for a factory for message objects.
* [Reflection](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#Reflection): This interface contains methods that can be used to dynamically access and modify the fields of a protocol message.


实战
----

### 1、根据message type name动态创建相应的message对象实例

步骤如下：

1. 用 DescriptorPool::generated_pool() 找到一个 DescriptorPool 对象，它包含了程序编译的时候所链接的全部 protobuf Message types
2. 用 DescriptorPool::FindMessageTypeByName() 根据 type name 查找该type name 对应的 Descriptor
3. 再用 MessageFactory::generated_factory() 找到 MessageFactory 对象，它能创建程序编译的时候所链接的全部 protobuf Message types
4. 然后，用 MessageFactory::GetPrototype() 找到具体 Message Type 的 default instance
5. 最后，用 prototype->New() 创建对象

具体实现如下：

	Message* createMessage(const std::string& typeName) {
		Message* message = NULL;
		const Descriptor* descriptor = DescriptorPool::generated_pool()->FindMessageTypeByName(typeName);
		if (descriptor) {
			const Message* prototype = MessageFactory::generated_factory()->GetPrototype(descriptor);
			if (prototype) {
				message = prototype->New();
			}
		}
		return message;
	}

使用的时候根据需要，可以强制类型转换得到具体的message：

	Message* message = createMessage("graphsearch.Person");
	assert(message != NULL);
	graphsearch::Person* person = dynamic_cast<T*>(message);

上面的createMessage只是得到一个default instance，但是这个instance的各个字段值还是默认值，但是拥有具体的类型我们就可以进行序列化和反序列化了：
	
	message->ParseFromString(data);

或者:

    person->ParseFromString(data);


### 2、动态获取和设置字段值

pb的Message基类提供了一个Reflection，这个类非常强大，可以利用他达到动态设置字段值的效果。

官方文档说的很直白，这个类就是用来动态访问和设置字段值的：

> [Reflection](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#Reflection): This interface contains methods that can be used to dynamically access and modify the fields of a protocol message.

比如知识图谱，实体定义如下：

	message Entity {
	    /************************************** 
	     *        实体公共属性                  *
	     **************************************/

	    // 实体id，需要稳定且唯一
	    required string id = 1;
	   
	    // 实体类型(一个实体只能属于一个类型)
	    required string entity_type = 2; 
	   
	    // 实体名称
	    required string name = 3;
	   
	    // 实体别名
	    repeated string aliases = 4;  
	   
	    // 实体描述
	    optional string description  = 5; 

	    // 具体实体内容，如Person, Product
	    // 序列化成bytes，根据entity_type反序列化     
	    required bytes data = 20;
	}

假设有个具体的实体类型Person：

	import "schema_index.proto";

	message Person {
	    // 实体id，需要稳定且唯一
	    required string id = 1;
	   
	    // 实体类型(一个实体只能属于一个类型)
	    required string entity_type = 2; 
	   
	    // 实体名称
	    required string name = 3;
	   
	    // 实体别名
	    repeated string aliases = 4;  
	   
	    // 实体描述
	    optional string description  = 5; 

	    // 实体特有属性
	    required uint32 age = 10;

	    // ...

	    option (schema_index) = "3,4"; // 需要建立索引的字段
	}

离线系统调用我们的添加实体接口：

    string add_entity(Entity entity);

我们会根据schema_index得到需要构建索引的字段，然后拿到这个字段的值进行索引构建。


参考文档
-------

1. [Protobuf message object creation by name](http://stackoverflow.com/questions/29960871/protobuf-message-object-creation-by-name)
2. [C++ Reference >> C++API >> Protobuf >> descriptor.h >> DescriptorPool.generated_pool.details](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor#DescriptorPool.generated_pool.details)
3. [protobuf通过反射来赋值](https://www.cppfans.org/1758.html)
4. [一种自动反射消息类型的 Google Protobuf 网络传输方案](http://www.cnblogs.com/Solstice/archive/2011/04/03/2004458.html)