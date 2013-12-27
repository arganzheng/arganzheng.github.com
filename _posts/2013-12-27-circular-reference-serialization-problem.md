---
title: 循环引用序列化问题
layout: post
---
	

假如一个对象里面引用了自己，那么序列化的时候如果不做特殊处理就会死循环了。在XStream中是采用`reference`处理的。反序列化的时候也根据`reference`来特殊处理。JSON也是类似的，不过貌似支持不是很好。

	package me.arganzheng.study.xstream;

	import org.apache.commons.lang.StringUtils;

	import com.thoughtworks.xstream.XStream;
	import com.thoughtworks.xstream.io.json.JettisonMappedXmlDriver;

	/**
	 * 循环引用序列化例子
	 * 
	 * @author arganzheng
	 * @date 2013-12-27
	 */
	class Person {

	    String firstname = "argan";
	    String lastname  = "zheng";

	    Person me      = this;

	    @Override
	    public String toString() {
	        return "Person [firstname=" + firstname + ", lastname=" + lastname + "]";
	    }
	}

	public class XStreamTest {

	    public static void main(String[] args) {

	        XStream xs = new XStream();
	        xs.alias(StringUtils.uncapitalize(Person.class.getSimpleName()), Person.class);

	        Person person = new Person();
	        person.me = new Person();

	        String xml = xs.toXML(person);
	        System.out.println(xml);

	        Person p = (Person) xs.fromXML(xml);
	        System.out.println(p);

	        XStream jsonXStream = new XStream(new JettisonMappedXmlDriver());
	        //jsonXStream.setMode(XStream.NO_REFERENCES);
	        String json = jsonXStream.toXML(person);
	        System.out.println(json);
	    }
	}


输入结果：

	<person>
	  <firstname>argan</firstname>
	  <lastname>zheng</lastname>
	  <me>
	    <firstname>argan</firstname>
	    <lastname>zheng</lastname>
	    <me reference=".."/>
	  </me>
	</person>
	Person [firstname=argan, lastname=zheng]
	{"com.qq.b2b2c.api.container.controller.Person":{"firstname":"argan","lastname":"zheng","me":[{"firstname":"argan","lastname":"zheng"},{"@reference":".."}]}}
