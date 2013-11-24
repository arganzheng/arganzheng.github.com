---
layout: post
title: content negotiation using spring mvc
---


元数据管理平台的REST接口突然抽风，总是返回XML格式，不管我怎么设置accept头。Debug了一下，发现每次都是优先使用`ServletPathExtensionContentNegotitionStrategy`。谷歌了一下，原来Spring默认的三个ContentNegotiationStrategy是PPA Strategy (path extension, then parameter, then Accept header) ，顺便也是先path extension，然后parameter(默认是format参数)，然后才是accept头。要让Spring使用accept头解析，只要配置一下`ContentNegotiationManagerFactoryBean`：

    <bean id="contentNegotiationManager"
                class="org.springframework.web.accept.ContentNegotiationManagerFactoryBean">
                <property name="favorPathExtension" value="false" />
                <property name="favorParameter" value="false" />
                <property name="ignoreAcceptHeader" value="false" />
                <property name="mediaTypes">
                        <value>
                                html=text/html
                                json=application/json
                                jsonp=application/javascript
                                *=*/*
                        </value>
                </property>
        </bean>

但是配置之后发现并没有生效，这是最容易出错的一点，就是一定要记得在`<mvc:annotation-driven>`中配置使用的`content-negotiation-manager`。因为@ResponseBody是一个anotation来的。Spring处理anotation的配置入口是`<mvc:annotation-driven>`，会导致重复配置，但是没有办法，只能尽量使用ref了。

    <mvc:annotation-driven
    	content-negotiation-manager="contentNegotiationManager">
		<mvc:message-converters>
			<bean class="com.qq.b2b2c.api.util.MappingJsonpHttpMessageConverter">
			</bean>
		</mvc:message-converters>
	</mvc:annotation-driven>

配置之后就可以了。

### 参考文档

1. [Content Negotiation using Spring MVC](http://spring.io/blog/2013/05/11/content-negotiation-using-spring-mvc)