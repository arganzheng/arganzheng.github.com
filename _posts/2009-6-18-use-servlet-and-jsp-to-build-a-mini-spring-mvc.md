---
layout: post
title: 使用Servlet和JSP模拟最小化的SpringMVC框架
---

 
web服务器或应用服务器都是容器，容器只知道与Listener或Servlet打交道，因此所谓的Web MVC框架其实就是预定义了一些Servlet帮我们做了本应该是我们做的事情。
 
关键：如何用Servlet和JSP实现Web MVC框架

一个Web MVC简单的实现如下：在Servlet中实现web控制逻辑，在它们结束之后，再导向到适当的JSP视图。相应的模型对象将通过Servlet API保存在请求对象（HttpRequest）的属性中，在Servlet控制器和JSP视图之间传递。这也是Web MVC最根本的精髓。
   
    public class MyServlet extends HttpServlet{
       //重载doGet()方法
       protected void doGet(HttpServletReqest request, HttpRequestResponse response)
                       throws ServletException, IOException{
           //将Model注入request对象的属性中
           request.setAttributes("message", "Hello world!");
           request.setAttributes("headline", "my ad-hoc servlet controller");
           //转发到相应的JSP页面显示处理结果
           request.getRequestDispatcher("/myview.jsp").forword(request, response);
       }
    }
    
该动作的URL必须在标准的web.xml部署描述符中配置，每一个动作对应一个servlet。JSP的名称既可以在servlet中硬编码，也可以通过web.xml的servlet初始化参数(init-params属性）指定。

**Tip** HttpServlet

每个HttpServlet类都有三个方法供容器调用：

    init()==>service()==>destory()
    
其中service()方法一般定义如下：
    
    protected void service(HttpServletRequest request, HttpServletResponse response)
               throws ServletException, IOException{
       string method = reqest.getMethod();
        if(method.equals(METHOD_GET))
           doGet(request, response);
        else if(method.equals(METHOD_POST))
           doPost(request, response);
        else if( ... )
           //other METHOD
        else{
           //Servlet doesn't currently support
           String errMsg = "Method not support");
           response.sendError(HttpServletResponse.SC_NOT_IMPLEMENTED, errmsg);
        }
    }
 
现在我们有足够的基础来实现一个简单的Web MVC框架了，我们这里以模拟一个简单Spring MVC为例：

<a href="http://www.flickr.com/photos/arganzheng/9313955568/" title="spring-in-action-13.1 by arganzheng, on Flickr"><img src="http://farm8.staticflickr.com/7328/9313955568_13420b6e9c_o.jpg" width="487" height="347" alt="spring-in-action-13.1"></a>

### 配置和类定义：

#### 1. 首先在web.xml中配置dispatcher servlet：
    
    <servlet>
      <servlet-name>mydispatcher</servlet-name>
      <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
      <load-on-startup>1</load-on-startup>
    </servlet>
     
    <servlet-mapping>
      <servlet-name>roadrantz</servlet-name>
      <url-pattern>*.htm</url-pattern>
    </servlet-mapping>

#### 2. 定义一个Controller并在SpringMVC配置文件（默认为${dispatcherServletName}-config.xml）配置映射：
##### 2.1 定义一个Controller
    
    public class HomePageController implements Controller{
       public ModelAndView handleRequest(HttpServletRequest request,
                             HttpServletResponse response) throws Exception {
           return new ModelAndView("home", "message", greeting);
     
       private String greeting;
       public void setGreeting(String greeting){
          this.greeting = greeting;
       }
    }

##### 2.2 配置一个Controller
在mydispatcher-servlet.xml文件中做如下简单配置：
    
    <bean name="/home.htm"
       class="com.springframework.training.mvc.HomePageController>
      <property name="greeting">
         <value>Hello world!</value>
      </property>
    </bean>
 
#### 3. 声明一个视图解析器

另一个需要在mydispatcher-servlet.xml中配置的Bean是视图解析器Bean。一个视图解析器的工作是根据ModelAndView返回的视图名字，将其映射成一个视图。在上例中，我们需要一个视图解析器将"home"解析成一个JSP文件来呈现主页。
>
As you’ll see in section 13.4, Spring MVC comes with several view resolvers from which to choose. But for views that are rendered by JSP, there’s none simpler than InternalResourceViewResolver:
    <bean id="viewResolver"
    class="org.springframework.web. servlet.view.InternalResourceViewResolver">
      <property name="prefix">
          <value>/WEB-INF/jsp/</value>
      </property>
      <property name="suffix">
         <value>.jsp</value>
       </property>
    </bean>
 
#### 4. 创建一个JSP页面

最后一步就是创建一个JSP页面来显示Model，在上例中这个JSP名字为home.jsp并且存放在/WEB-INF/jsp目录下：
   
    <%@ page contentType="text/html" %>
    <html>
      <head><title>Spring training, Inc</title></head>
      <body>
         <h2>${message}</h2>
      </body>
    </html>

### 小结：一个简单的SpringMVC请求过程：

<a href="http://www.flickr.com/photos/arganzheng/9313955674/" title="spring-in-action-13.4 by arganzheng, on Flickr"><img src="http://farm3.staticflickr.com/2844/9313955674_e84e75c468_o.jpg" width="541" height="365" alt="spring-in-action-13.4"></a>
 
### 模拟：

在Tomcat启动时，会根据web.xml中定义的servlet自动加载我们声明的分发器mydispatcher，这相当于：

    org.springframework.web.servlet.DispatcherServlet mydispatcher = new org.springframework.web.servlet.DispatcherServlet();
    mydispatcher.init(); //这个方法会加载SpringMVC的配置文件
    
然后用户请求一个/home.htm：
根据映射，所有*.htm由mydispatcher处理，于是tomcat将这个请求交给mydispatcher。
mydispatcher要根据这个URL来确定真正处理这个请求的Controller，这是BeanNameHandlerMapping做的事情：

    BeanNameHandlerMapping handlerMapping = new BeanNameHandlerMapping();
    //得到处理控制器
    Controller controller = handlerMapping.handlerMapping("/home.htm");
    //处理该请求
    ModelAndView mv = controller.handlerRequest(request, response);
     
    //创建视图解析器
    InternalResouceViewResolver viewResolver = new InternalResouceViewResolver();
    //解析逻辑视图名字为JSP文件
    String url = viewResolver.resolverView(mv.getView()); //prefix+view+suffix
     
    // 将Model注入request对象的属性中
    for(Model model: mv.getModel()){
           request.setAttributes(model.getKey(), model.getValue());
    }
           //转发到相应的JSP页面显示处理结果
          request.getRequestDispatcher(url).forword(request, response);
    }
    
这就是所有的过程，就是这么简单。