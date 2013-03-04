---
layout: post
title: Restful Spring MVC
---


Spring MVC本身对Restful支持非常好。它的`@RequestMapping`、`@RequestParam`、`@PathVariable`、`@ResponseBody`注解很好的支持了REST。[18.2 Creating RESTful services](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch18s02.html)

### 1. `@RequestMapping`

Spring uses the @RequestMapping method annotation to define the URI Template for the request. 类似于struts的action-mapping。 可以指定POST或者GET。

### 2. `@PathVariable`

The @PathVariable method parameter annotation is used to indicate that a method parameter should be bound to the value of a URI template variable. 用于抽取URL中的信息作为参数。（注意，不包括请求字符串，那是`@RequestParam`做的事情。）

    @RequestMapping("/owners/{ownerId}", method=RequestMethod.GET)
    public String findOwner(@PathVariable String ownerId, Model model) {
            // ...
    }
如果变量名与pathVariable名不一致，那么需要指定：

    @RequestMapping("/owners/{ownerId}", method=RequestMethod.GET)
    public String findOwner(@PathVariable("ownerId") String theOwner, Model model) {
        // implementation omitted
    }

>**Tip**
>
method parameters that are decorated with the @PathVariable annotation can be of any simple type such as int, long, Date... Spring automatically converts to the appropriate type and throws a TypeMismatchException if the type is not correct.


### 3. `@RequestParam`

官方文档居然没有对这个注解进行说明，估计是遗漏了（真不应该啊）。这个注解跟`@PathVariable`功能差不多，只是参数值的来源不一样而已。它的取值来源是请求参数（querystring或者post表单字段）。

对了，因为它的来源可以是POST字段，所以它支持更丰富和复杂的类型信息。比如文件对象:

    @RequestMapping("/imageUpload")
    public String processImageUpload(@RequestParam("name") String name,
                    @RequestParam("description") String description,
                    @RequestParam("image") MultipartFile image) throws IOException {
        this.imageDatabase.storeImage(name, image.getInputStream(), 
                                        (int) image.getSize(), description);
        return "redirect:imageList";
    }

### 4. `@RequestBody`和`@ResponseBody`

这两个注解其实用到了Spring的一个非常灵活的设计——`HttpMessageConverter` [18.3.2 HTTP Message Conversion](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch18s03.html#rest-message-conversion)

与`@RequestParam`不同，`@RequestBody`和`@ResponseBody`是针对整个HTTP请求或者返回消息的。前者只是针对HTTP请求消息中的一个 name=value 键值对(名称很贴切)。

`HtppMessageConverter`负责将HTTP请求消息(HTTP request message)转化为对象，或者将对象转化为HTTP响应体(HTTP response body)。
    
    public interface HttpMessageConverter<T> {

        // Indicate whether the given class is supported by this converter.
        boolean supports(Class<? extends T> clazz);

        // Return the list of MediaType objects supported by this converter.
        List<MediaType> getSupportedMediaTypes();

        // Read an object of the given type form the given input message, and returns it.
        T read(Class<T> clazz, HttpInputMessage inputMessage) throws IOException, 
                                                                        HttpMessageNotReadableException;

        // Write an given object to the given output message.
        void write(T t, HttpOutputMessage outputMessage) throws IOException, 
                                                                HttpMessageNotWritableException;

    }        
    
    
Spring MVC对`HttpMessageConverter`有多种默认实现，基本上不需要自己再自定义`HttpMessageConverter`
>
+ StringHttpMessageConverter - converts strings
+ FormHttpMessageConverter - converts form data to/from a MultiValueMap<String, String>
+ ByteArrayMessageConverter - converts byte arrays
+ SourceHttpMessageConverter - convert to/from a javax.xml.transform.Source
+ RssChannelHttpMessageConverter - convert to/from RSS feeds
+ MappingJacksonHttpMessageConverter - convert to/from JSON using Jackson's ObjectMapper
+ etc...

然而对于RESTful应用，用的最多的当然是`MappingJacksonHttpMessageConverter`。

但是`MappingJacksonHttpMessageConverter`不是默认的`HttpMessageConverter`：

    public class AnnotationMethodHandlerAdapter extends WebContentGenerator
    implements HandlerAdapter, Ordered, BeanFactoryAware {

        ...
        
        public AnnotationMethodHandlerAdapter() {
            // no restriction of HTTP methods by default
            super(false);

            // See SPR-7316
            StringHttpMessageConverter stringHttpMessageConverter = new StringHttpMessageConverter();
            stringHttpMessageConverter.setWriteAcceptCharset(false);
            this.messageConverters = new HttpMessageConverter[]{new ByteArrayHttpMessageConverter(), stringHttpMessageConverter,
            new SourceHttpMessageConverter(), new XmlAwareFormHttpMessageConverter()};
        }
    }   
如上：默认的`HttpMessageConverter`是`ByteArrayHttpMessageConverter`、`stringHttpMessageConverter`、`SourceHttpMessageConverter`和`XmlAwareFormHttpMessageConverter`转换器。所以需要配置一下：
        
    <bean class="org.springframework.web.servlet.mvc.annotation.AnnotationMethodHandlerAdapter">
        <property name="messageConverters">
        <list>
            <bean class="org.springframework.http.converter.StringHttpMessageConverter">
            <property name="supportedMediaTypes">
                <list>
                <value>text/plain;charset=GBK</value>
                </list>
            </property>
            </bean>
            <bean class="org.springframework.http.converter.json.MappingJacksonHttpMessageConverter" />
        </list>
        </property>
    </bean>

配置好了之后，就可以享受`@Requestbody`和`@ResponseBody`对JONS转换的便利之处了：

    @RequestMapping(value = "api", method = RequestMethod.POST)
    @ResponseBody
    public boolean addApi(@RequestBody
        Api api, @RequestParam(value = "afterApiId", required = false)
        Integer afterApiId) {
            Integer id = apiMetadataService.addApi(api);
            return id > 0;
    }

    @RequestMapping(value = "api/{apiId}", method = RequestMethod.GET)
    @ResponseBody
    public Api getApi(@PathVariable("apiId")
        int apiId) {
            return apiMetadataService.getApi(apiId, Version.primary);
    }

一般情况下我们是不需要自定义`HttpMessageConverter`，不过对于Restful应用，有时候我们需要返回jsonp数据：

    package me.arganzheng.study.springmvc.util;

    import java.io.IOException;
    import java.io.PrintStream;

    import org.codehaus.jackson.map.ObjectMapper;
    import org.codehaus.jackson.map.annotate.JsonSerialize.Inclusion;
    import org.springframework.http.HttpOutputMessage;
    import org.springframework.http.converter.HttpMessageNotWritableException;
    import org.springframework.http.converter.json.MappingJacksonHttpMessageConverter;
    import org.springframework.web.context.request.RequestAttributes;
    import org.springframework.web.context.request.RequestContextHolder;
    import org.springframework.web.context.request.ServletRequestAttributes;

    public class MappingJsonpHttpMessageConverter extends MappingJacksonHttpMessageConverter {

        public MappingJsonpHttpMessageConverter() {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.setSerializationConfig(objectMapper.getSerializationConfig().withSerializationInclusion(Inclusion.NON_NULL));
        setObjectMapper(objectMapper);
        }

        @Override
        protected void writeInternal(Object o, HttpOutputMessage outputMessage) throws IOException, HttpMessageNotWritableException {
        String jsonpCallback = null;

        RequestAttributes reqAttrs = RequestContextHolder.currentRequestAttributes();
        if(reqAttrs instanceof ServletRequestAttributes){
            jsonpCallback = ((ServletRequestAttributes)reqAttrs).getRequest().getParameter("jsonpCallback");
        }

        if(jsonpCallback != null){
            new PrintStream(outputMessage.getBody()).print(jsonpCallback + "(");
        }
        
        super.writeInternal(o, outputMessage);
        
        if(jsonpCallback != null){
            new PrintStream(outputMessage.getBody()).println(");");
        }
        }
    }

如果请求的参数中带有`jsonpCallback`，那么会返回jsonp格式数据。比如：
http://open.buy.qq.com/meta/api/1.xhtml?jsonpCallback=clientFunction。
会返回`clientFunction(…);`

### 5. 返回多种表现形式(Returning multiple representations)

对于Restful服务，一个资源往往有多种表现形式，比如最常见的就是返回xml和json格式数据，还有就是RSS和ATOM。怎样让客户端告诉Restful服务，我希望得到什么样表现形式的资源呢？
一般来说client有两者方式通知Server它希望拿到的资源格式：

1. 使用不同URI来表示同个资源的不同表现形式。一般使用不同的文件拓展名。如http://blog.arganzheng.me/users/argan.xml表示返回xml格式数据，而http://blog.arganzheng.me/users/aganzheng.json表示返回json格式.
2. 使用同个URI，但是通过Accept HTTP request header来告诉server它理解的media types。例如同样请求http://blog.arganzheng.me/users/argan，如果带上`text/xml` accept header表示请求一个XML资源，带上`application/pdf`则表示期望收到pdf格式资源。

Spring提供了[`ContentNegotiatingViewResolver`](http://static.springsource.org/spring/docs/3.0.x/javadoc-api/org/springframework/web/servlet/view/ContentNegotiatingViewResolver.html)来解决这个问题：

  
    public class ContentNegotiatingViewResolver extends WebApplicationObjectSupport implements ViewResolver, Ordered {

        private static final Log logger = LogFactory.getLog(ContentNegotiatingViewResolver.class);
    
        private static final String ACCEPT_HEADER = "Accept";
    
        private static final boolean jafPresent =
            ClassUtils.isPresent("javax.activation.FileTypeMap", ContentNegotiatingViewResolver.class.getClassLoader());
    
        private static final UrlPathHelper urlPathHelper = new UrlPathHelper();
    
    
        private int order = Ordered.HIGHEST_PRECEDENCE;
    
        private boolean favorPathExtension = true;
    
        private boolean favorParameter = false;
    
        private String parameterName = "format";
    
        private boolean useNotAcceptableStatusCode = false;
    
        private boolean ignoreAcceptHeader = false;
    
        private boolean useJaf = true;
    
        private ConcurrentMap<String, MediaType> mediaTypes = new ConcurrentHashMap<String, MediaType>();
    
        private List<View> defaultViews;
    
        private MediaType defaultContentType;
    
        private List<ViewResolver> viewResolvers;
    
    
        // ignore some setter and getter...    
        
        public void setMediaTypes(Map<String, String> mediaTypes) {
          Assert.notNull(mediaTypes, "'mediaTypes' must not be null");
          for (Map.Entry<String, String> entry : mediaTypes.entrySet()) {
            String extension = entry.getKey().toLowerCase(Locale.ENGLISH);
            MediaType mediaType = MediaType.parseMediaType(entry.getValue());
            this.mediaTypes.put(extension, mediaType);
          }
        }
    
        public void setDefaultViews(List<View> defaultViews) {
          this.defaultViews = defaultViews;
        }
    
        public void setDefaultContentType(MediaType defaultContentType) {
          this.defaultContentType = defaultContentType;
        }
        
        public void setViewResolvers(List<ViewResolver> viewResolvers) {
          this.viewResolvers = viewResolvers;
        }
    
    
        @Override
        protected void initServletContext(ServletContext servletContext) {
          if (this.viewResolvers == null) {
            Map<String, ViewResolver> matchingBeans =
                BeanFactoryUtils.beansOfTypeIncludingAncestors(getApplicationContext(), ViewResolver.class);
            this.viewResolvers = new ArrayList<ViewResolver>(matchingBeans.size());
            for (ViewResolver viewResolver : matchingBeans.values()) {
              if (this != viewResolver) {
                this.viewResolvers.add(viewResolver);
              }
            }
          }
          if (this.viewResolvers.isEmpty()) {
            logger.warn("Did not find any ViewResolvers to delegate to; please configure them using the " +
                "'viewResolvers' property on the ContentNegotiatingViewResolver");
          }
          OrderComparator.sort(this.viewResolvers);
        }
    
        public View resolveViewName(String viewName, Locale locale) throws Exception {
          RequestAttributes attrs = RequestContextHolder.getRequestAttributes();
          Assert.isInstanceOf(ServletRequestAttributes.class, attrs);
          List<MediaType> requestedMediaTypes = getMediaTypes(((ServletRequestAttributes) attrs).getRequest());
          if (requestedMediaTypes != null) {
            List<View> candidateViews = getCandidateViews(viewName, locale, requestedMediaTypes);
            View bestView = getBestView(candidateViews, requestedMediaTypes);
            if (bestView != null) {
              return bestView;
            }
          }
          if (this.useNotAcceptableStatusCode) {
            if (logger.isDebugEnabled()) {
              logger.debug("No acceptable view found; returning 406 (Not Acceptable) status code");
            }
            return NOT_ACCEPTABLE_VIEW;
          }
          else {
            logger.debug("No acceptable view found; returning null");
            return null;
          }
        }
    
        
        protected List<MediaType> getMediaTypes(HttpServletRequest request) {
          if (this.favorPathExtension) {
            String requestUri = urlPathHelper.getRequestUri(request);
            String filename = WebUtils.extractFullFilenameFromUrlPath(requestUri);
            MediaType mediaType = getMediaTypeFromFilename(filename);
            if (mediaType != null) {
              if (logger.isDebugEnabled()) {
                logger.debug("Requested media type is '" + mediaType + "' (based on filename '" + filename + "')");
              }
              return Collections.singletonList(mediaType);
            }
          }
          if (this.favorParameter) {
            if (request.getParameter(this.parameterName) != null) {
              String parameterValue = request.getParameter(this.parameterName);
              MediaType mediaType = getMediaTypeFromParameter(parameterValue);
              if (mediaType != null) {
                if (logger.isDebugEnabled()) {
                  logger.debug("Requested media type is '" + mediaType + "' (based on parameter '" +
                      this.parameterName + "'='" + parameterValue + "')");
                }
                return Collections.singletonList(mediaType);
              }
            }
          }
          if (!this.ignoreAcceptHeader) {
            String acceptHeader = request.getHeader(ACCEPT_HEADER);
            if (StringUtils.hasText(acceptHeader)) {
              try {
                          List<MediaType> mediaTypes = MediaType.parseMediaTypes(acceptHeader);
                          MediaType.sortByQualityValue(mediaTypes);
                          if (logger.isDebugEnabled()) {
                              logger.debug("Requested media types are " + mediaTypes + " (based on Accept header)");
                          }
                          return mediaTypes;
              }
              catch (IllegalArgumentException ex) {
                if (logger.isDebugEnabled()) {
                  logger.debug("Could not parse accept header [" + acceptHeader + "]: " + ex.getMessage());
                }
                return null;
              }
            }
          }
          if (this.defaultContentType != null) {
            if (logger.isDebugEnabled()) {
              logger.debug("Requested media types is " + this.defaultContentType +
                  " (based on defaultContentType property)");
            }
            return Collections.singletonList(this.defaultContentType);
          }
          else {
            return Collections.emptyList();
          }
        }
    
        
        protected MediaType getMediaTypeFromFilename(String filename) {
          String extension = StringUtils.getFilenameExtension(filename);
          if (!StringUtils.hasText(extension)) {
            return null;
          }
          extension = extension.toLowerCase(Locale.ENGLISH);
          MediaType mediaType = this.mediaTypes.get(extension);
          if (mediaType == null && this.useJaf && jafPresent) {
            mediaType = ActivationMediaTypeFactory.getMediaType(filename);
            if (mediaType != null) {
              this.mediaTypes.putIfAbsent(extension, mediaType);
            }
          }
          return mediaType;
        }
    
        
        protected MediaType getMediaTypeFromParameter(String parameterValue) {
          return this.mediaTypes.get(parameterValue.toLowerCase(Locale.ENGLISH));
        }
    
        private List<View> getCandidateViews(String viewName, Locale locale, List<MediaType> requestedMediaTypes)
            throws Exception {
    
          List<View> candidateViews = new ArrayList<View>();
          for (ViewResolver viewResolver : this.viewResolvers) {
            View view = viewResolver.resolveViewName(viewName, locale);
            if (view != null) {
              candidateViews.add(view);
            }
            for (MediaType requestedMediaType : requestedMediaTypes) {
              List<String> extensions = getExtensionsForMediaType(requestedMediaType);
              for (String extension : extensions) {
                String viewNameWithExtension = viewName + "." + extension;
                view = viewResolver.resolveViewName(viewNameWithExtension, locale);
                if (view != null) {
                  candidateViews.add(view);
                }
              }
    
            }
          }
          if (!CollectionUtils.isEmpty(this.defaultViews)) {
            candidateViews.addAll(this.defaultViews);
          }
          return candidateViews;
        }
    
        private List<String> getExtensionsForMediaType(MediaType requestedMediaType) {
          List<String> result = new ArrayList<String>();
          for (Entry<String, MediaType> entry : this.mediaTypes.entrySet()) {
            if (requestedMediaType.includes(entry.getValue())) {
              result.add(entry.getKey());
            }
          }
          return result;
        }
    
        private View getBestView(List<View> candidateViews, List<MediaType> requestedMediaTypes) {
          MediaType bestRequestedMediaType = null;
          View bestView = null;
          for (MediaType requestedMediaType : requestedMediaTypes) {
            for (View candidateView : candidateViews) {
              if (StringUtils.hasText(candidateView.getContentType())) {
                MediaType candidateContentType = MediaType.parseMediaType(candidateView.getContentType());
                if (requestedMediaType.includes(candidateContentType)) {
                  bestRequestedMediaType = requestedMediaType;
                  bestView = candidateView;
                  break;
                }
              }
            }
            if (bestView != null) {
              if (logger.isDebugEnabled()) {
                logger.debug("Returning [" + bestView + "] based on requested media type '" +
                    bestRequestedMediaType + "'");
              }
              break;
            }
          }
          return bestView;
    
        }
        
        ...
        
    }


可以看到`ContentNegotiationViewResolver`有点类似于ComposeCommand（参见Command模式 by GoF），它本身实现了ViewResolver接口，所以它是一个ViewResolver，但是它组合了一堆的ViewResolver，根据一定的规则（前面讨论的content negotiation）将视图请求转发给最match的ViewResolver。

所以关键在两点：

#### 1. content negotiation规则

>This view resolver uses the requested media type to select a suitable View for a request. This media type is determined by using the following criteria:
>
1. If the requested path has a file extension and if the setFavorPathExtension(boolean) property is true, the mediaTypes property is inspected for a matching media type.
2. If the request contains a parameter defining the extension and if the setFavorParameter(boolean) property is true, the mediaTypes property is inspected for a matching media type. The default name of the parameter is format and it can be configured using the parameterName property.
3. If there is no match in the mediaTypes property and if the Java Activation Framework (JAF) is both enabled and present on the classpath, FileTypeMap.getContentType(String) is used instead.
4. If the previous steps did not result in a media type, and ignoreAcceptHeader is false, the request Accept header is used.
>   
Once the requested media type has been determined, this resolver queries each delegate view resolver for a View and determines if the requested media type is compatible with the view's content type). The most compatible view is returned.

##### 2. 供选择的SingleViewResolver
>    
1. The ContentNegotiatingViewResolver does not resolve views itself, but delegates to other ViewResolvers. By default, these other view resolvers are picked up automatically from the application context, though they can also be set explicitly by using the viewResolvers property. Note that in order for this view resolver to work properly, the order property needs to be set to a higher precedence than the others (the default is Ordered.HIGHEST_PRECEDENCE.)
> 
    说明：即`private List<ViewResolver> viewResolvers;`属性。需要注意的是Spring会自动加载和注册所有其他的ViewResolver到`ContentNegotiationViewResolover`的`viewResolvers`属性。但是你需要告诉Spring MVC，你希望controller返回的view都是由`ContentNegotiationViewResolover`来解析，而不是其他定义的ViewResolver。这是通过order配置项来决定。你应该给`ContentNegotiationViewResolover`配置最高的order(其实默认就是最高了)。
>
2. Additionally, this view resolver exposes the defaultViews property, allowing you to override the views provided by the view resolvers. Note that these default views are offered as candicates, and still need have the content type requested (via file extension, parameter, or Accept header, described above). You can also set the default content type directly, which will be returned when the other mechanisms (Accept header, file extension or parameter) do not result in a match.
>
    说明：即`private List<View> defaultViews;`和`private MediaType defaultContentType;`属性。


关于`ContentNegotiatingViewResolver`，下面两篇文章都不错，值得一看：

1. [Content Negotiation using Spring MVC's ContentNegotiatingViewResolver](http://blog.eyallupu.com/2009/07/content-negotiation-using-spring-mvcs.html):使用了`viewResolvers`配置。
2. [ADDING AN ATOM VIEW TO AN APPLICATION USING SPRING'S REST SUPPORT](http://blog.springsource.com/2009/03/16/adding-an-atom-view-to-an-application-using-springs-rest-support/):使用了`ViewResolvers`配置。
3. [Spring 3 MVC ContentNegotiatingViewResolver Example](http://www.mkyong.com/spring-mvc/spring-3-mvc-contentnegotiatingviewresolver-example/):使用了`defaultViews`配置。

注意：`@ResponseBody`是为了单个View准备的，即它只能转换成一种格式，对于`ContentNegotiatingViewResolver`，需要多个**Single**ViewResolver来接收。

### 6. 客户端调用[Accessing RESTful services on the Client](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch18s03.html#rest-resttemplate)

Spring MVC不仅大大的简化了服务端RESTful服务的开发和开放，还提供了一些辅助类来方便客户端调用REST服务。

以前Client如果要调用REST服务，一般是使用HttpClient来发送HTTP请求：
    
    String uri = "http://example.com/hotels/1/bookings";
    
    PostMethod post = new PostMethod(uri);
    String request = // create booking request content
    post.setRequestEntity(new StringRequestEntity(request));
    
    httpClient.executeMethod(post);
    
    if (HttpStatus.SC_CREATED == post.getStatusCode()) {
      Header location = post.getRequestHeader("Location");
      if (location != null) {
        System.out.println("Created new booking at :" + location.getValue());
      }
    }

太过底层，而且代码比较冗长，一般都要手动封装一下（即类似于SDK，封装了签名和HTTP发送和接受细节）。我们看一下Spring MVC是怎么解决这个问题的。

#### [RestTemplate](http://blog.springsource.org/2009/03/27/rest-in-spring-3-resttemplate/)

RestTemplate是client-site HTTP access的核心类。正如它的名称所示，`RestTemplate`非常类似于`JdbcTemplate`, `JmsTemplate`等XXXTemplate。这意味着`RestTemplate`是线程安全的并且可以通过callback来定制它的行为。

TIPS：Spring提供的Template类非常灵活和好用，种类也很丰富。当你需要做一些事情的时候可以先考虑一下有没有相应的template可以用。

RestTemplate默认使用`java.net`包下的基础类来创建HTTP请求。你可以实现`ClientHttpRequestFactory`接口，提供你自己的Http请求工厂类。Spring提供了`CommonsClientHttpRequestFactory`，这个工厂类使用Jakarta Commons HttpClient来创建HTTP请求。这样就可以使用HttpClient提供的认证和链接池功能了。

[RestTemplate提供的方法如下](http://static.springsource.org/spring/docs/3.0.x/javadoc-api/org/springframework/web/client/RestTemplate.html)：

>**HTTP Method**	**RestTemplate Method**
>
* DELETE	delete(String url, String… urlVariables)
* GET	getForObject(String url, Class<T> responseType, String… urlVariables)
* HEAD	headForHeaders(String url, String… urlVariables)
* OPTIONS	optionsForAllow(String url, String… urlVariables)
* POST	postForLocation(String url, Object request, String… urlVariables)
* PUT	put(String url, Object request, String…urlVariables)
* ANY	exchange(String, HttpMethod, HttpEntity, Class, Object...)
        execute(String, HttpMethod, RequestCallback, ResponseExtractor, Object...)

方法名称很有规律，都是这个pattern——`${HTTP Method}${WhatIsReturne}`。例如getForObject() will perform a GET, convert the HTTP response into an object type of your choice, and returns that object. postForLocation will do a POST, converting the given object into a HTTP request, and returns the response HTTP Location header where the newly created object can be found. As you can see, these methods try to enforce REST best practices.

其中getForObject()、postForLocation()和put()方法接收或者返回的参数通过`HttpMessageConverter`来转换为Http Request或者Http Response。这点与前面介绍服务端RESTful的`@RequestBody`和`@ResponseBody`是一样的，Spring MVC默认会注册常用的Converter，你也可以自定义。

另外，每个方法的第一个参数都是一个url string，但是这个URI可以带有变量(还记得`@PathVariable`吗:)哦。参数有两种方式绑定值：

1. 作为字符串变量数组(String variable arguments array)
    
        String result = restTemplate.getForObject("http://example.com/hotels/{hotel}/bookings/{booking}", String.class, "42", "21");
        
    会转换为一个对`http://example.com/hotels/42/bookings/21`的GET请求。
    
2. 或者Map对象(Map)

   The map variant expands the template based on variable name, and is therefore more useful when using many variables, or when a single variable is used multiple times.
        
        Map<String, String> vars = new HashMap<String, String>();
        vars.put("hotel", "42");
        vars.put("booking", "21");
        String result = restTemplate.getForObject("http://example.com/hotels/{hotel}/bookings/{booking}", String.class, vars);
        会转换为一个对`http://example.com/hotels/42/rooms/42`的GET请求。

关于RestTemplate使用的具体例子可以参考这篇文章[
REST IN SPRING 3: RESTTEMPLATE](http://blog.springsource.org/2009/03/27/rest-in-spring-3-resttemplate/)。写的非常好，强烈推荐！
    
### 7. 设计的URL

在开发功能模块之前，应该先把URL设计好。比查对 **消息** 这个资源的操作URL可以这么设计：
    
    http://arganzheng.me/messages/show/123456
    http://arganzheng.me/messages/preview/123456
    http://arganzheng.me/messages/delete/123456
    http://arganzheng.me/messages/new
    http://arganzheng.me/message/update
    
说明：可以看到我们的URL中有动作在里面，事实上纯粹的RESTful URL是把动作隐含在HTTP头中：GET、PUT、DELETE、POST。。不过这样对用户编码有要求，这个相对简单点。

要支持这种URL，web.xml需要这么配置：

    <!-- REST servlet-mapping -->	<servlet-mapping>		<servlet-name>DispatcherServlet<srvlet-name>		<url-pattern>/</url-pattern>	<srvlet-mapping>
	
但是这样的话有个问题，就是静态文件也被mapping了，会导致找不到资源。Spring提供了一个resources配置项支持静态文件的处理[16.14.5 Configuring Serving of Resources](http://static.springsource.org/spring/docs/3.1.x/spring-framework-reference/html/mvc.html#mvc-config-static-resources)：

    <!-- Forwards requests to the "/" resource to the "welcome" view -->	<mvc:view-controller path="/" view-name="index"/>		<!-- Handles HTTP GET requests for /resources/** by efficiently serving up static resources in the ${webappRoot}/resources/ directory -->	<mvc:resources mapping="/resources/**" location="/resources/" />	<!-- 注：配置了mvc:resources就必须配置这个选项，否则handler mapping都失效了 		@see  http://stackoverflow.com/questions/7910845/the-handler-mapping-from-the-mvcresource-override-other-mappings-which-defined 	-->	<mvc:annotation-driven />

这样所有请求：http://arganzheng.me/resources/**的会映射到webapp下的resources目录，而不是找我们的controller处理。

但是有个奇怪的问题，就是配置这个之后，原来动态东西就不能访问到了，提示找不到对应的handler，解决方案是增加一个`<mvc:annotation-driven />`配置。具体参见[The handler mapping from the mvc:resource override other mappings which defined with annotation](http://stackoverflow.com/questions/7910845/the-handler-mapping-from-the-mvcresource-override-other-mappings-which-defined)。

另外，静态的html页面一般不放在resources路面下，而是直接在根目录下，比如：http://arganzheng.me/index.html或者http://arganzheng.me/404.html。所以应该在web.xml中在配置一个url-mapping规则：

    <!-- 避免被Spring DispatcherServlet接管 -->	<servlet-mapping>		<servlet-name>default<srvlet-name>		<url-pattern>*.html</url-pattern>	<srvlet-mapping>

--EOF--


   

