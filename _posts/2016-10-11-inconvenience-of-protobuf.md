---
title: Protobuf Buffer的缺陷
layout: post
catalog: true
---


最近在搞知识图谱，知识图谱的模型比较统一——就是实体和关系。其中实体和关系可以挂载关联的属性，属性以key-value键值对存在。称之为Property Graph Model。

* 属性: 

	Property: key->value

* 实体

	Entity: (id) => (type, property*)

其中: type决定该实体的schema，如Person, Product，一个实体必须且只能挂在一个type下面。type这里相当于行业。


* 关系: SPO三元组，附加可选的k-v属性对

	Assoc: (id1, atype, id2) => (time, property*)

其中atype是关系的类型，起到labeled Edge的作用。

根据这个定义我们很容易定义出如下实体类：

	package me.arganzheng.kg.model;

	import java.util.ArrayList;
	import java.util.List;

	import com.google.gson.Gson;
	import com.google.gson.GsonBuilder;

	public class Entity {

	    String id;
	    String type;
	    String name;
	    List<String> aliases;
	    String description;
	    String image;
	    String url;

	    List<Property> properties = new ArrayList<Property>();

	    /**
	     * Add or set a property value for the {@code Element} given its key.
	     */
	    public <V> Property<V> addProperty(final String key, final V value, float confidence) {
	        Property<V> p = new Property<V>();
	        p.key = key;
	        p.value = value;
	        p.confidence = confidence;

	        this.properties.add(p);

	        return p;
	    }

	    public <V> Property<V> getProperty(final String key) {
	        for (Property<V> p : this.properties) {
	            if (p.key.equals(key)) {
	                return p;
	            }
	        }
	        return null;
	    }

	    public static void main(String[] args) {
	        Entity entity = new Entity();
	        entity.id = "1";
	        entity.name = "argan";
	        List<String> aliases = new ArrayList<String>();
	        aliases.add("Forrest");
	        aliases.add("Gump");
	        entity.aliases = aliases;
	        entity.type = "Person";
	        entity.description = "arganzheng test KG model";
	        entity.addProperty("age", 35, 10.0f);
	        entity.addProperty("nickname", "magi", 10.0f);
	        

	        Gson gson = new GsonBuilder().create();
	        System.out.println(gson.toJson(entity));

	        Property<Integer> age = entity.getProperty("age");
	        System.out.println(age.value);

	    }

	    static class Property<V> {

	        String key;

	        V value;

	        float confidence;

	    }
	}

List<Property> 查找不是很方便，可以改成map:

	public class Entity {

	    String id;
	    String type;
	    String name;
	    List<String> aliases;
	    String description;
	    String image;
	    String url;

	    Map<String, Property> properties = new HashMap<String, Property>();

	    /**
	     * Add or set a property value for the {@code Element} given its key.
	     */
	    public <V> Property<V> addProperty(final String key, final V value, float confidence) {
	        Property<V> p = new Property<V>();
	        p.value = value;
	        p.confidence = confidence;

	        this.properties.put(key, p);

	        return p;
	    }

	    public <V> Property<V> getProperty(final String key) {
	        return this.properties.get(key);
	    }

	    public static void main(String[] args) {
	        Entity entity = new Entity();
	        entity.id = "1";
	        entity.name = "argan";
	        List<String> aliases = new ArrayList<String>();
	        aliases.add("Forrest");
	        aliases.add("Gump");
	        entity.aliases = aliases;
	        entity.type = "Person";
	        entity.description = "arganzheng test KG model";
	        entity.addProperty("age", 35, 10.0f);
	        entity.addProperty("nickName", "magi", 10.0f);

	        Gson gson = new GsonBuilder().create();
	        System.out.println(gson.toJson(entity));

	        Property<Integer> age = entity.getProperty("age");
	        Property<String> nickName = entity.getProperty("nickName");
	        System.out.println(age.value);
	        System.out.println(nickName.value);

	    }

	    static class Property<V> {

	        V value;

	        float confidence;

	    }

	}

当然，如果觉得Property太过于动态，也可以采用静态类继承的方式：

比如定义一个Entity基类：


	public class Entity {

	    String id;
	    String type;
	    String name;
	    List<String> aliases;
	    String description;
	    String image;
	    String url;

	}

然后各个行业的类型定义成子类，如Person：


	public class Person extends Entity {

	    int age; 

	    Date birthdate;
	    	   
	    float height;

	    float weight;

	    // ...

	    public static void main(String[] args) {
	        Person entity = new Person();
	        entity.id = "1";
	        entity.name = "argan";
	        List<String> aliases = new ArrayList<String>();
	        aliases.add("Forrest");
	        aliases.add("Gump");
	        entity.aliases = aliases;
	        entity.type = "Person";
	        entity.description = "arganzheng test KG model";
	        entity.age = 10;
	        entity.birthdate = new Date();

	        Gson gson = new GsonBuilder().create();
	        System.out.println(gson.toJson(entity));

	        System.out.println(entity.age);
	    }
	}

用起来比较确定，就是模型可能会变动，需要客户端更新。

然后我们需要提供RPC接口给上层业务使用，假设我们要提供的接口如下：

	List<Entity> entities_search(List<String> ids);

这时候需要对这些对象进行序列化了，然后问题来了。凡是语言无关的序列化方案都没有没有泛型、继承的概念。以protobuf为例子，泛型只能用types表示；没有继承，只有组合，类似于C的struct。有几种方式可以选择，每种都有各自的优缺点。

总体来说有两种方式:

1. 分散式
2. 集中式

### 1、分散式

每个继承的子类都使用一个独立的类，在消息中的第一个字段为父类的消息，并且名称可以固定成为super。例如我们上面的例子：

	message Entity { 	// 基类
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;			
	}

	message Person {
		required Entity super = 1;

		/// 子类自己的属性..
 		optional int age = 2; 

	    ...
	}

	message Product {
		required Entity super = 1;

		/// 子类自己的属性..
 		optional float price = 2; 

	    ...
	}

这种方式使用方式如下：

写：

	Entity super;
	Person person;

	super.set_id(..);
	super.set_name(..);
	..

	person.set_entity(super);
	person.set_age(..);

读：

	String id = person.get_super().get_id();
	int age = person.get_age();

是不是觉得使用起来特别别扭？一点继承的意思都没有，完全就是组合嘛。。

为了解决这种别扭，干脆就把基类在各个子类展开定义算了：

	message Person {
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;			

		/// 子类自己的属性..
 		optional int age = 2; 

	    ...
	}

	message Product {
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;			


		/// 子类自己的属性..
 		optional float price = 2; 

	    ...
	}

这样使用起来就顺畅的多了：

写：

	Person person;

	person.set_id(..);
	person.set_name(..);
	..

	person.set_age(..);

读：

	String id = person.get_id();
	int age = person.get_age();

缺点就是每个子类都要重复定义一次，如果有修改的话需要改动多处，而且没有一个通用的父类接受。这里其实Protobuf是可以提供类似于 C 的宏定义的，一处定义，到处展开。可以达到代码复用的效果。


### 法二、集中式

另外一种方式是集中式，整个继承体系使用一个独立的类，把所有的子类都封装在一起，类似于C中的union。

	message Entity { 	// 基类
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;	

		// 子类作为父类的一个optional字段：
		optional Person person = 60;
		optional Product product = 61;
		// .. 其他的子类 ..
	}

	message Person {
		/// 子类自己的属性..
 		optional int age = 2; 

	    ...
	}

	message Product {
		/// 子类自己的属性..
 		optional float price = 2; 

	    ...
	}

这种方式的使用方式如下：

写：

	Entity entity;
	Person person;

	entity.set_id(..);
	entity.set_name(..);
	..

	person.set_age(..);

	entity.set_person(person);

读：
	
	string id = entity.get_id();
	if (entity.has_person()) {
		int age = entity.get_person().get_age();
		...
	}
	
子类变成父类的一个属性了。。感觉更别扭。。

而且这种方式还需要检查哪个子类存在，写入的时候还可以写入多个子类。不过Protobuf提供了一个语法可以防止这种情况发生——[Oneof](https://developers.google.com/protocol-buffers/docs/proto#oneof)，达到了真正类似于union的效果：

	message Entity { 	// 基类
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;	

		// 子类作为父类的一个optional字段，使用oneof语法：
		oneof extension {
			Person person = 60;
			Product product = 61;
			// .. 其他的子类 ..
		} 
	}

然后可以这种使用：

写：

	Entity entity;
	Person person;
	Product product;

	entity.set_id(..);
	person.set_age(..);
	product.set_price(..);

	entity.set_person(person);
	entity.set_product(product); // will clear person field!

读：

	switch(entityoneof_name_case()) {
	
	  case Entity::kPerson: {
	  	Person person = entity.get_person();
		int age = person.get_age();
	  	...
	  	break;
	  }
	  case Entity::kProduct: {
	  	Product product = entity.get_product();
	  	float price = product.get_price();
	  	...
	  	break;
	  }

	  ...

	  default: 
	  	break;
	}

可惜子类的名字并不是统一的，看起来也是跟上面的optional差不多，只是会保证只有一个子类会被设置成功。另外，读取的时候，需要同时用到子类和父类（子类和父类看起来像是毫无关系的两个类一样）。


跟这种方式类似，有一种称之为 Embedded Serialized Messages的方法，就是把子类序列化成父类的一个字段，然后用一个type字段表明子类类型，这样就可以根据这个类型对子类进行反序列化了：

	message Entity { 	// 基类
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;	

		enum Type {
	        Person = 1;
	        Product = 2;
	        // ...
	    }

		// 子类序列化成bytes
		required Type type = 100;
    	required bytes subclass = 101;
	}

然后每个子类还是自己定义：

	message Person {
		/// 子类自己的属性..
 		optional int age = 2; 

	    ...
	}

	message Product {
		/// 子类自己的属性..
 		optional float price = 2; 

	    ...
	}

使用方式跟上面的oneof也是非常类似：

写：

	Entity entity;
	Person person;

	entity.set_id(..);
	entity.set_name(..);
	..

	person.set_age(..);

	// 序列化为二进制，保存在父类中
	entity.set_type(Type::Person);
    person.SerializeToString(entity.add_subclass());

读：
	
	string id = entity.get_id();
	if (entity.get_type() == Entity::Person) {
		Person person;
		person.ParseFromString(entity.get_subclass());
		int age = person.get_age();
		...
	}


protobuf还有另一种语法，可以把所有子类的名字都统一，就是[extensions](https://developers.google.com/protocol-buffers/docs/proto#extensions)。
这个名称其实有点诡异，并不是java的继承的意思。就是纯粹的扩展父类：

	message Entity { 	// 基类
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;	

		// 父类预留一些字段作为扩展：
		extensions 100 to max;
	}

然后“子类”可以定义扩展属性：

person.proto:

	import "entity.proto";	

	extend Entity {  
	    optional int age = 100;  
	    ...
	} 

product.proto:

	import "entity.proto";	

	extend Entity {  
	    optional float price = 200;  
	    ...
	} 


看起来像是直接在entity中增加了新的字段，但是使用起来还是有区别的：所有的扩展属性都放在一个叫做extension的属性中：

	Entity entity;
	entity.set_id(..);
	
	entity.SetExtension(age, 30); // 不能直接 entity.set_age(30); 

	entity.SetExtension(price, 30.0); // 不能直接 entity.set_price(30.0); 

	int age = entity.GetExtension(age); // 不能直接 entity.get_age();

	float price = entity.GetExtension(price); // 不能直接 entity.get_price();


**NOTE**

1、注意tag数字不要冲突，否则会发生类型冲突。

2、可以看到这种方式并没有子类的概念，所有的子类字段都打散存放在extension字段中了。相当于一个超级扁平化的父类。protobuf提供了一个[Nested Extensions](https://developers.google.com/protocol-buffers/docs/proto#nested-extensions)，对新增字段增加了一个命名空间的概念，可以相对缓解这个问题：


person.proto:

	import "entity.proto";	

	message Person { 
		extend Entity {  
	    	optional int age = 100;  
	    	...
	    }
	} 

product.proto:

	import "entity.proto";	

	message Product { 
		extend Entity {  
		    optional float price = 200;  
		    ...
		} 
	}

然后使用的时候可以这么搞：

	Entity entity;
	entity.set_id(..);
	
	entity.SetExtension(Person::age, 30); // 不能直接 entity.set_age(30); 

	entity.SetExtension(Product::price, 30.0); // 不能直接 entity.set_price(30.0); 

	int age = entity.GetExtension(Person::age); // 不能直接 entity.get_age();

	float price = entity.GetExtension(Product::price); // 不能直接 entity.get_price();

相对来说看起来清晰一些。

还可以稍微变换一下，通过在子类中扩展父类，让其指向自己，而不是一个个打散的属性：

	message Entity { 	// 基类
		// 实体id，需要稳定且唯一
		required string id  = 1;			

		// 实体类型(一个实体只能属于一个类型)
		required enum type  = 2;			

		// 实体名称
		required string name  = 3;			

		// 实体别名
		repeated string aliases  = 4;		

		// 实体描述
		optional string description  = 5;	

		// An image to help identify the entity.
		optional string image  = 6;			

		// The official website URL of the entity, if available.
		optional string url  = 7;	

		// 父类预留一些字段作为扩展：
		extensions 100 to max;

		enum Type {
	        Person = 1;
	        Product = 2;
	    }

	    required Type type = 8;
	}

然后“子类”可以定义扩展属性：

person.proto:

	import "entity.proto";	

	message Person { 
		extend Entity {  
	    	required Person entity = 100;

	    }
	    optional int age = 1; 
    	... 
	} 

product.proto:

	import "entity.proto";	

	message Product { 
		extend Entity {  
	    	required Product entity = 100;

	    }
	    optional float price = 200;  
	    ...
	}

使用起来大概是这个样子：

写：

	Entity entity;
	entity.set_id(..);
	
	entity.MutableExtension(Person::entity)->set_age(30); 
	entity.set_type(Entity::Person);

	// product类似
	entity.MutableExtension(Product::entity)->set_price(30.0);
	entity.set_type(Entity::Product);

读：
	switch(entity->type()) {
	
	  case Entity::Person: {
	  	int age = entity.MutableExtension(Person::entity)->age();
	  	...
	  	break;
	  }
	  case Entity::Person: {
	  	float price = entity.MutableExtension(Product::entity)->price();
	  	...
	  	break;
	  }

	  ...

	  default: 
	  	break;
	}

不过使用起来也是怪怪的。。


总结
---

protobuf中没有继承的概念，只能使用类似于组合的方式来达到类似的目的，每种方式都有各自优缺点，使用起来都没有继承优雅，需要自己根据实际情况做权衡：

1. 分散式
    1. 子类包含父类
    2. 直接将父类在子类展开
2. 集中式
    1. 所有子类作为父类的一个 optional 字段，类似于C的 union 
    2. 所有子类作为父类的一个 optional字 段，通过 protobuf 的 oneof 语法，类似于C的 union
    3. Embedded Serialized Messages：将子类序列化成父类的一个 bytes 字段，然后用一个 type 字段表明子类类型
    4. 使用protobuf的 extensions 语法，扩展"父"类，所有扩展字段存放在“父”类的一个 extensions 字段。
    5. 使用protobuf的 Nested Extensions 语法，扩展"父"类，所有扩展字段存放在“父”类的一个 extensions 字段，但是扩展属性有各自的命名空间。
    6. Nested Extensions 的一个特殊例子：扩展属性就是子类本身。

从对用户的使用友好度来说，1.2 和 2.5/2.6 是相对比较友好的。其他的情况使用起来多多少少有点别扭。。


参考文章
-------

1. [Protocol Buffer Polymorphism](http://www.indelible.org/ink/protobuf-polymorphism/)
2. [Protobuf Examples for C++ and Java (1)](http://aboutfedora.blogspot.com/2012/10/protobuf-examples-for-c-and-java-1.html)


