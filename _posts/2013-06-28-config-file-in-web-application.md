配置文件串串SHOW
================

2009--18 星期四 热
 
## 一、有哪些配置文件

1. 入口配置文件：web.xml；由web或应用服务器为每个web项目加载的配置文件。
2. 应用上下文：包括web框架特有配置文件：如struts的struts-config.xml（或struts2的struts.xml）文件和SpringMVC的${dispatcherServletName}-servlet.xml配置文件。和Spring的配置文件，一般按层划分为业务层xxx-service.xml和持久层xxx-data.xml。
说明：事实上，web层配置文件也可以使用Spring配置，这就是所谓的框架整合。

<a href="http://www.flickr.com/photos/arganzheng/9287336069/" title="spring-config-file by arganzheng, on Flickr"><img src="http://farm3.staticflickr.com/2822/9287336069_deaf342f10_o.jpg" width="520" height="181" alt="spring-config-file"></a>
 
## 二、配置文件的加载过程
web.xml是所有配置文件的入口配置文件，Tomcat在启动时会自动加载该配置文件，并且对其中定义的listener和servlet等进行相应的加载和初始化。Tomcat并不清楚（也不关心）其他配置文件的存在，因此，加载其他配置文件应该是你（框架）的事情。那么怎么加载呢？前面说过Tomcat在启动时会自动加载和初始化listener和servlet，那么我们可以自定义一个listener或servlet，Tomcat会根据web.xml加载和初始化他们，而他们则负责加载相应的配置文件。这正是所有Web框架采用的做法。
 
## 三、加载规则
web.xml有Tomcat在启动时候加载。Web层框架的配置文件（如SpringMVC的${dispatcherServletName}-servlet.xml和Struts的struts-config.xml配置文件）一般由框架定义的dispatcher servlet在初始化时加载。而Spring的应用上下文配置文件则由Spring框架在web.xml中定义的一个listener加载。需要注意的是web.xml的定义是顺序相关的，listener的定义要在servlet之前。
例子：
1. SpringMVC为例

    <webapp version=...>
     <context-param><!-- 指定Spring应用上下文配置文件所在位置 -->
        <param-name>contextConfigLocation</param-name>
        <param-value> <!-- classpath:applicationContext-ussp.xml，或使用逗号隔开放在一行-->
            /WEB-INF/myapp-sercurity.xml
            /WEB-INF/myapp-service.xml
            /WEB-INF/myapp-data.xml
         </param-value>
    
     <listener><!-- Listener用于加载Spring应用上下文配置文件 -->
        <listener-class>org.springframework.web.context.ContextLoaderListener
        </listener-class>
     </listener>
    
     <servlet> <!-- CenterDispatcherServlet用于加载Web MVC框架配置文件 -->
       <servlet-name>mydispatcherServlet</servlet-name>
       <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
       <init-param><!-- 指定Web MVC框架配置文件所在位置，在SpringMVC中如果不指定，默认为${dispatcherServletName}-servlet.xml文件 -->
         <param-name>contextConfigLocation</param-name>
         <param-value>/WEB-INF/xxx-springmvc.xml</param-value>
       <init-param>
     </servlet>
    
     <servlet-mapping><!-- 指定URL映射分发器 -->
        <servlet-name>mydispatcherServlet</servlet-name>
        <url-pattern>*.htm</url-pattern>
     </servlet-mapping>
    </webapp>
 
说明：

1. 加载Spring应用上下文：在<context-param>指定where to store the Spring context file。一般采用分层配置的方法。上例中将所有应用配置文件一次写在web.xml文件中，另一种方法是使用import引入其他配置文件（一般上层引入下层）。
一旦加载完Spring应用上下文，可以通过如下方式从ServletContext中获取Spring的WebApplicationContext：

    public void setServlet(ActionServlet actionServlet){
        super.setServlet(actionServlet);
        ServletContext servletContext = actionServlet.getServletContext();
        WebApplicationContext wac = WebApplicationContextUtils.getRequiredWebApplicationContext(servletContext);
        this.petStore = (PetStoreFacade)was.getBean("petStore");
    }
 
注意：需要注意的是在web层也常常将配置文件进行模块分解，而且与Spring上下文分解由惊人的类似：
一种是在web.xml中一次指定所有配置文件，下面以struct1.x为例：
  
    <servlet> <!-- CenterDispatcherServlet用于加载Web MVC框架配置文件 -->
      <servlet-name>mydispatcherServlet</servlet-name>
      <servlet-class>org.apache.struts.action.ActionServlet</servlet-class>
      <init-param>
        <param-name>config</param-name>
        <param-value>/WEB-INF/struts-config.xml</param-value>
      <init-param>
      <init-param>
        <param-name>config/userMgr</param-name>
        <param-value>/WEB-INF/userMgr/struts-config-user.xml</param-value>
      <init-param>
      <!-- 依次可增加多个Struts配置文件-->
    </servlet>

另一种方式是通过在入口配置文件中使用<include file="xxx.xml" />实现的。
 
2. 加载SpringMVC框架配置文件：在<servlet>指定SpringMVC的front-end controller（center dispatcher），其中可以用<init-param>指定配置文件位置。
 
3. 将Spring应用上下文按层分解为多个配置文件是一个好习惯，另一个好习惯是使用property文件和Spring占位符来加强安全性和灵活性：将host-specific parameters放在一个regular java properties file and user Spring's PropertyPlaceHolderConfigurer class to map these parameters to your bean properties.
例子：a properties file （WEB-INF/jdbc.properties）

    jdbc.dirver=org.postgresql.Driver
    jdbc.url=jdbc:postgresql://localhost/test
    jdbc.user=...
    jdbc.password=..
     
and our bean configuration will look like this:

    <bean id="propertyConfiguration"
       class="org.springframework.beans.facory.config.PropertyPlaceHolderConfigurer>
       <property name="location">
          <value>/WEB-INF/jdbc.properties</value>
       </property>
    </bean>

如果有多个properties文件，那么可以使用locations属性：
  
    <bean id="propertyConfiguration"
      class="org.springframework.beans.facory.config.PropertyPlaceHolderConfigurer>
     <property name="locations">
        <list>
          <value>/WEB-INF/jdbc.properties</value>
          <value>/WEB-INF/security.properties</value>
          <value>/WEB-INF/application.properties</value>
       </list>
     </property>
    </bean>
 
这样我们在配置dataSource的时候就可以这样配置：
  
    <bean id="dataSource"
     class="org.springframework.jdbc.datasource.DriverManagerDataSource"
     <property name="dirverClassName">
         <value>${jdbc.driver}</value>
     </property>
     <property name="url">
         <value>${jdbc.url}</value>
     </property>
     ...
    </bean>
  
当上下文被载入时，占位符变量都被jdbc.properties中的属性值替代。
 

补充：ServletContextListener
============================
>
Implemeations of this interface receive notification about changes to the servlet context of the web application they are part of. To receive notification events, the implementation class must be configured in the deployment descriptor for the web application(that is in the web.xml file).
 