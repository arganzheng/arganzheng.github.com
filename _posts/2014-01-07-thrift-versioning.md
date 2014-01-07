---
title: Thrift的序列化版本控制
layout: post
---


在现实世界中，唯一的不变就是变化。即使是接口定义也不例外。所以，做好接口版本控制，升级和兼容是非常重要的一个指标。

> The system must be able to support reading of old data from log files, as well as requests from out-of-date clients to new servers, and vice versa.

跟PB一样，都是通过Field Identifiers实现的。即，对IDL中定义的字段用一个唯一的数字进行标识。每个序列化字段的标识是由这个field id及其类型声明组合构成的。

> Versioning in Thrift is implemented via field identifiers. The field header for every member of a struct in Thrift is encoded with a unique field identifier. The combination of this field identifier and its type specifier is used to uniquely identify the field.

	struct Example {
		1:i32 number=10,
		2:i64 bigNumber, 
		3:double decimals, 
		4:string name="thrifty"
	}

即上面的 1:i32、2:i64、3:double、4:string，就是每个字段的标识。至于字段名称，反而不重要，所以可以随意变更字段名称。只要id和类型不变就可以了。

当反序列化的时候，根据读出的id和type，在自己IDL定义中对比，如果id存在，那么按照相应的类型进行反序列化，如果不存在，那么简单的忽略扔掉。所以不会出错。现在app platform的反序列化问题就在于没有使用id进行对比，序列化和反序列化的依据都是同一份IDL文件生成出来的。而且对序列化顺序进行了假设。


Isset 关于字段是否被设置了
-----------------------

> When an unexpected field is encountered, it can be safely ignored and discarded. When an expected field is not found, there must be some way to signal to the developer that it was not present. This is implemented via an inner isset structure inside the defined objects. (Isset functionality is implicit with a null value in PHP, None in Python and nil in Ruby.) Essentially, the inner isset object of each Thrift struct contains a boolean value for each field which denotes whether or not that field is present in the struct. When a reader receives a struct, it should check for a field being set before operating directly on it.

这个问题其实很常见，假如你收到一个请求，里面有几个参数是null，但是你怎么知道是客户没有设置这几个参数的值呢（取默认值了）？还是就是传递了null值过来呢？这就是需要isSet方法来告诉接收端了。内部实现一般是为每个字段默认生成一个boolean xxx_isSet的附属字段，如果相应的xxx字段被设置值了，那么就把对应的xxx_isSet字段设置为true。一般在setter方法里面进行。


案例分析
-------

>There are four cases in which version mismatches may occur.
> 
1. *Added field, old client, new server.* In this case, the old client does not send the new field. The new server recognizes that the field is not set, and implements default behavior for out-of-date requests.
2. *Removed field, old client, new server.* In this case, the old client sends the removed field. The new server simply ignores it.
3. *Added field, new client, old server.* The new client sends a field that the old server does not recognize. The old server simply ignores it and processes as normal.
4. *Removed field, new client, old server.* This is the most dangerous case, as the old server is unlikely to have suitable default behavior implemented for the missing field. It is recommended that in this situation the new server be rolled out prior to the new clients.

可见，只有case 4 才需要先升级server。不过一般来说字段删除是不允许。而且一般也是server先做兼容性升级，client再跟着升级。client先升级一般风险比较高。

Protocol/Transport Versioning
-----------------------------

这个对于RPC来说当然是必须的，每个协议包头都会包含当前协议的版本，方便协议的升级。比如TCP4和TCP6。协议包头一般还包含checkSum、length、QoS、source、target等业务无关的信息。




