---
title: protobuf中的反射
layout: post
---

像Python，javascript这样的动态脚本语言具有很强的反射能力，即使是java这类的静态类型语言，也提供了一定程度的反射能力，而C/C++这样的纯静态语言没有运行时的类型信息，但是Protobuf本身具有很强的反射(reflection)功能，可以根据 type name 创建具体类型的 Message 对象，也能够动态获取和设置某个属性。

背景知识
-------

protobuf对于每个元素都有一个相应的descriptor，这个descriptor包含该元素的所有元信息，非常类似于Spring中的Bean Definition。下面是各个Descriptor（元数据描述类）的类图：

![protobuf_descriptors_classdiagram.png](/img/in-post/protobuf_descriptors_classdiagram.png)

1. FileDescriptor: 对一个proto文件的描述，它包含文件名、包名、选项（如package, java_package, java_outer_classname等）、文件中定义的所有message、文件中定义的所有enum、文件中定义的所有service、文件中所有定义的extension、文件中定义的所有依赖文件（import）等。在FileDescriptor中还存在一个DescriptorPool实例，它保存了所有的dependencies(依赖文件的FileDescriptor)、name到GenericDescriptor的映射、字段到FieldDescriptor的映射、枚举项到EnumValueDescriptor的映射，从而可以从该DescriptorPool中查找相关的信息，因而可以通过名字从FileDescriptor中查找Message、Enum、Service、Extensions等。可以通过`--descriptor_set_out`指定生成某个proto文件相对应的FileDescriptorSet文件。
2. Descriptor: 对一个message定义的描述，它包含该message定义的名字、所有字段、内嵌message、内嵌enum、关联的FileDescriptor等。可以使用字段名或字段号查找FieldDescriptor。
3. FieldDescriptor：对一个字段或扩展字段定义的描述，它包含字段名、字段号、字段类型、字段定义(required/optional/repeated/packed)、默认值、是否是扩展字段以及和它关联的Descriptor/FileDescriptor等。
4. EnumDescriptor：对一个enum定义的描述，它包含enum名、全名、和它关联的FileDescriptor。可以使用枚举项或枚举值查找EnumValueDescriptor。
5. EnumValueDescriptor：对一个枚举项定义的描述，它包含枚举名、枚举值、关联的EnumDescriptor/FileDescriptor等。
6. ServiceDescriptor：对一个service定义的描述，它包含service名、全名、关联的FileDescriptor等。
7. MethodDescriptor：对一个在service中的method的描述，它包含method名、全名、参数类型、返回类型、关联的FileDescriptor/ServiceDescriptor等。

具体描述可以参考官方文档: [
descriptor.h](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor)

有意思的是，这些Descriptor类其实本身也是通过protobuf定义的：[
descriptor.pb.h](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor.pb)，然后你可以通过`--descriptor_set_out`指定生成某个proto文件相对应的FileDescriptorSet文件，这个文件就是`message FileDescriptorSet`序列化的结果。需要的时候你可以使用`FileDescriptorSet.ParseFrom`得到proto的元信息。

通过这些Descriptor我们就可以在运行期间获取到各种元数据（如某个Message有哪些字段，每个字段的类型等），从而动态的做一些事情（如动态的设置某个属性的值）。

那么怎样获取到这些Descroptor呢？

总的来说有两种方式：

1、动态编译：使用protobuf的动态编译机制，在运行时对某个proto文件进行动态编译，从而得到其所有元数据(descriptor):

    DiskSourceTree sourceTree;
    //look up .proto file in current directory
    sourceTree.MapPath("", "./");
    Importer importer(&sourceTree, NULL);
    //runtime compile foo.proto
    importer.Import("foo.proto");
 
    const Descriptor *descriptor = importer.pool()->FindMessageTypeByName("test.Foo");

**NOTES**

1. 其实importer.Import("foo.proto")会返回一个FileDescriptor，也可以通过这个file descriptor对该proto文件进行操作。
2. `Importer(SourceTree* source_tree, MultiFileErrorCollector* error_collector)`，貌似PB没有提供默认的MultiFileErrorCollector实现，需要自己实现一个，实现其实蛮简单的。
3. 如果要编译的proto文件有import其他的proto文件，那么有可能编译报错，需要把原来的proto文件也放在指定目录。

2、静态编译

其实Protobuf默认生成的xxx.pb.cc文件会有一个静态类，在这个静态类的构造函数会把自己注册进去，放在`DescriptorPool::generated_pool`中，这样就可以在运行期间通过`DescriptorPool::generated_pool`拿到注册的元信息了。

* [static void MessageFactory::InternalRegisterGeneratedFile(const char* filename, void(*)(const string&)register_messages)](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#MessageFactory.InternalRegisterGeneratedFile.details)
	* For internal use only: Registers a .proto file at static initialization time, to be placed in generated_factory.
	* The first time GetPrototype() is called with a descriptor from this file, |register_messages| will be called, with the file name as the parameter. It must call InternalRegisterGeneratedMessage() (below) to register each message type in the file. This strange mechanism is necessary because descriptors are built lazily, so we can't register types by their descriptor until we know that the descriptor exists. |filename| must be a permanent string.
* [static void MessageFactory::InternalRegisterGeneratedMessage(const Descriptor* descriptor, const Message* prototype)](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#MessageFactory.InternalRegisterGeneratedMessage.details)
	* For internal use only: Registers a message type.
	* Called only by the functions which are registered with InternalRegisterGeneratedFile(), above.

**NOTES**

由于使用的是类静态初始化，假如这个proto文件没有被使用，就不会触发初始化，解决方案是手动的触发这个类：比如调用`foo.set_bar("xxx");`或者直接`import foo.pb.cc`。都有点恶心。。

![protobuf_classdiagram.png](/img/in-post/protobuf_classdiagram.png)

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


### 3、动态编译

我们还可以使用PB 提供的 google::protobuf::compiler 包在运行时动态编译指定的.proto 文件来使用其中的 Message。这样就可以通过修改.proto文件实现动态消息，有点类似配置文件的用法。完成这个工作主要的类叫做 importer，定义在 importer.h 中。

	#include <iostream>
	#include <google/protobuf/descriptor.h>
	#include <google/protobuf/descriptor.pb.h>
	#include <google/protobuf/dynamic_message.h>
	#include <google/protobuf/compiler/importer.h>
	 
	using namespace std;
	using namespace google::protobuf;
	using namespace google::protobuf::compiler;
	 
	int main(int argc, const char *argv[])
	{
	    DiskSourceTree sourceTree;
	    //look up .proto file in current directory
	    sourceTree.MapPath("", "./");
	    Importer importer(&sourceTree, NULL);
	    //runtime compile foo.proto
	    importer.Import("foo.proto");
	 
	    const Descriptor *descriptor = importer.pool()->FindMessageTypeByName("Pair");
	    cout << descriptor->DebugString();
	 
	    // build a dynamic message by "Pair" proto
	    DynamicMessageFactory factory;
	    const Message *message = factory.GetPrototype(descriptor);
	    // create a real instance of "Pair"
	    Message *pair = message->New();
	 
	    // write the "Pair" instance by reflection
	    const Reflection *reflection = pair->GetReflection();
	 
	    const FieldDescriptor *field = NULL;
	    field = descriptor->FindFieldByName("key");
	    reflection->SetString(pair, field, "my key");
	    field = descriptor->FindFieldByName("value");
	    reflection->SetUInt32(pair, field, 1111);
	 
	    cout << pair->DebugString();
	 
	    delete pair;
	 
	    return 0;
	}


### 4、动态定义proto

能不能通过程序生成protobuf文件呢？毕竟对用户来说protobuf还是有点偏向于程序化，小白用户可能更喜欢用表格来定义消息格式，然后我们内部转换成相应的proto格式的消息？答案是可以的。FileDescriptorProto允许你动态的定义你的proto文件：

	FileDescriptorProto file_proto;
	file_proto.set_name("my.proto");  
	file_proto.set_syntax("proto3");  
	  
	DescriptorProto *message_proto = file_proto.add_message_type();  
	message_proto->set_name("mymsg");  
	  
	FieldDescriptorProto *field_proto = NULL;  
	  
	field_proto = message_proto->add_field();  
	field_proto->set_name("len");  
	field_proto->set_type(FieldDescriptorProto::TYPE_UINT32);  
	field_proto->set_number(1);  
	field_proto->set_label(FieldDescriptorProto::LABEL_OPTIONAL);  
	  
	field_proto = message_proto->add_field();  
	field_proto->set_name("type");  
	field_proto->set_type(FieldDescriptorProto::TYPE_UINT32);  
	field_proto->set_number(2);  
	  
	DescriptorPool pool;  
	const FileDescriptor *file_descriptor = pool.BuildFile(file_proto);  
	cout << file_descriptor->DebugString();

上面代码所生成的结构和下面的my.proto文件是一样的：

	syntax = "proto3";  
	message mymsg  
	{  
	    uint32 len = 1;  
	    uint32 type = 2;  
	}


推荐阅读
-------

1. [Protobuf message object creation by name](http://stackoverflow.com/questions/29960871/protobuf-message-object-creation-by-name)
2. [C++ Reference >> C++API >> Protobuf >> descriptor.h >> DescriptorPool.generated_pool.details](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor#DescriptorPool.generated_pool.details)
3. [protobuf通过反射来赋值](https://www.cppfans.org/1758.html)
4. [一种自动反射消息类型的 Google Protobuf 网络传输方案](http://www.cnblogs.com/Solstice/archive/2011/04/03/2004458.html)
5. [玩转Protocol Buffers](http://www.searchtb.com/2012/09/protocol-buffers.html) 淘宝搜索技术博客写的一篇文章
6. [Self-describing Messages](https://developers.google.com/protocol-buffers/docs/techniques?hl=zh-CN#self-description) 介绍了一种自描述的消息描述机制，类似于Avro。
7. [Google Protocol Buffer 的使用和原理](https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/) 非常深入浅出的文章，强烈推荐。


