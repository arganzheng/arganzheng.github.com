---
layout: post
title: Rest Response and Exception
---


在笔者的上一篇博文 [Restful Spring MVC](http://blog.arganzheng.me/posts/restful-springmvc.html) 中，详细的介绍了使用Spring MVC构建RESTful的web应用的各个方面，不过对于RESTful应用还有很重要的一块笔者没有介绍到，就是异常和错误信息rest化。

对于一个RESTful web应用，正常情况下，应该返回一个200的HTTP response，返回内容格式由[ContentNegotiatingViewResolver](https://gist.github.com/arganzheng/6656136)协商决定，一般是json或者xml格式。如果出错了，业务一般是抛出业务异常，如果没有特殊处理，那么将返回一个出错HTML页面，状态码是500。我们希望这些异常能够映射到HTTP status，并且能够按照json/xml的格式返回，这时候就需要引入Spring MVC的[Handling exceptions](http://docs.spring.io/spring/docs/3.2.x/spring-framework-reference/html/mvc.html#mvc-exceptionhandlers)机制了。

> Besides implementing the HandlerExceptionResolver interface, which is only a matter of implementing the resolveException(Exception, Handler) method and returning a ModelAndView, you may also use the provided SimpleMappingExceptionResolver or create @ExceptionHandler methods. The SimpleMappingExceptionResolver enables you to take the class name of any exception that might be thrown and map it to a view name. This is functionally equivalent to the exception mapping feature from the Servlet API, but it is also possible to implement more finely grained mappings of exceptions from different handlers. The @ExceptionHandler annotation on the other hand can be used on methods that should be invoked to handle an exception. Such methods may be defined locally within an @Controller or may apply globally to all @RequestMapping methods when defined within an @ControllerAdvice class. The following sections explain this in more detail.

> The DefaultHandlerExceptionResolver works transparently by setting the status of the response. However, it stops short of writing any error content to the body of the response while your application may need to add developer-friendly content to every error response for example when providing a REST API. You can prepare a ModelAndView and render error content through view resolution -- i.e. by configuring a ContentNegotiatingViewResolver, MappingJacksonJsonView, and so on. However, you may prefer to use @ExceptionHandler methods instead.


有两种方式:

1. 实现`HandlerExceptionResolver`接口，或者拓展`AbstractHandlerExceptionResolver`抽象基类。然后配置xml配置文件。
2. 使用`@ExceptionHandler`注解

一般采用前者全局控制。

一个很好的例子是: [spring-mvc-rest-exhandler](https://github.com/arganzheng/spring-mvc-rest-exhandler)。不过它搞的稍微复杂了一点，功能很强大。


    <!-- Allow Exceptions to be handled in annotated methods if desired.  Otherwise fallback to the
         'restExceptionResolver' below: -->
    <bean id="annotationMethodHandlerExceptionResolver"
          class="org.springframework.web.servlet.mvc.annotation.AnnotationMethodHandlerExceptionResolver">
        <property name="order" value="0"/>
    </bean>
    <bean id="restExceptionResolver" class="com.stormpath.spring.web.servlet.handler.RestExceptionHandler">
        <property name="order" value="100"/>
        <property name="messageConverters">
            <list>
                <ref bean="jacksonHttpMessageConverter"/>
            </list>
        </property>
        <property name="errorResolver">
            <bean class="com.stormpath.spring.web.servlet.handler.DefaultRestErrorResolver">
                <property name="localeResolver" ref="localeResolver"/>
                <property name="defaultMoreInfoUrl" value="mailto:support@mycompany.com"/>
                <property name="exceptionMappingDefinitions">
                    <map>
                        <!-- 404 -->
                        <entry key="com.stormpath.blog.spring.mvc.rest.exhandler.UnknownResourceException" value="404, _exmsg"/>

                        <!-- 500 (catch all): -->
                        <entry key="Throwable" value="500"/>
                    </map>
                </property>
            </bean>
        </property>
    </bean>
    
推荐的RESTful是业务异常会影响HTTP response status，不过很多平台还是把业务异常通过业务消息返回。也就是说HTTP status Code统一返回200，返回的格式包含业务处理结果，一般是errorCode字段和errorMessage字段。成功处理结果errorCode=0。这时候也可以不返回errorCode和errorMessage头部。不过统一返回比较方便客户端解析。可以定义一个RestResponse<T>类，这样就可以使用上`@ResponseBody`了：


    package com.qq.ecc.openapi.eventpush.common;
    
    public class RestResponse<T> {
    
        private int    errorCode;
        private String errorMessage;
        private T      data;
    
        public int getErrorCode() {
            return errorCode;
        }
    
        public void setErrorCode(int errorCode) {
            this.errorCode = errorCode;
        }
    
        public String getErrorMessage() {
            return errorMessage;
        }
    
        public void setErrorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
        }
    
        public T getData() {
            return data;
        }
    
        public void setData(T data) {
            this.data = data;
        }
    }

这样就可以使用上`@ResponseBody`了：
    
    @Controller
    @RequestMapping("/users")
    public class UserController {
    
        private static Map<String, User> users = new HashMap<String, User>();
    
        static {
            users.put("jsmith", new User("Jane Smith", "jsmith"));
            users.put("djones", new User("Don Jones", "djones"));
            users.put("argan", new User("Argan Zheng", "argan"));
        }
    
        @RequestMapping(value = "/{username}", method = GET)
        @ResponseBody
        public User getUser(@PathVariable
        String username) {
            // simulate Manager/DAO call:
            return findUser(username);
        }
    
        private User findUser(String username) throws UnknownResourceException {
            if (!StringUtils.hasText(username)) {
                throw new IllegalArgumentException("Username is required.");
            }
    
            // simulate a successful lookup for 2 users:
            if (users.containsKey(username)) {
                return users.get(username);
            }
    
            // any other lookup throws an exception (not found):
            throw new UnknownResourceException("Unable to find user with username '" + username + "'");
        }
    
        @RequestMapping(method = RequestMethod.POST)
        @ResponseBody
        public RestResponse<User> addUser(@RequestBody
        User user) {
            users.put(user.getUsername(), user);
            RestResponse<User> resp = new RestResponse<User>();
            resp.setData(user);
            
            return resp;
        }
    }


这是正常的情况下，我们用`@ResponseBody`就可以正确返回RESTful格式的消息。那么异常情况下呢，比如上面的findUser方法，会抛出`UnknownResourceException`业务异常，我们希望这个异常能够返回这么一个格式：

    {
      "errorCode":404,
      "errorMessage":"Unable to find user with username 'argan2'"
    }

HTTP status还是200。

其实就是业务异常映射为errorCode和errorMessage（这里不考虑http status）。这个看实现的灵活度，最简单的实现方式就是要求业务异常类中必须还有这两个字段，那么转换的时候直接抽取就可以了。[Spring MVC Exception Handling Example](http://www.mkyong.com/spring-mvc/spring-mvc-exception-handling-example/)。由一个地方统一配置一个errorCode枚举类或者常量，不失为一个简单有效的方式。

    package com.qq.b2b2c.api.container.common;

    /**
     * API自定义异常基类
     * 
     * @author arganzheng
     * @date 2013-7-25
     */
    public class ApiException extends RuntimeException {
    
        private static final long serialVersionUID = 8372652214797382977L;
    
        private int               errorCode;
    
        public ApiException(int errorCode, String message){
            super(message);
            this.errorCode = errorCode;
        }
    
        public ApiException(int errorCode, String message, Throwable ex){
            super(message, ex);
            this.errorCode = errorCode;
        }
    
        public int getErrorCode() {
            return errorCode;
        }
    
        public void setErrorCode(int errorCode) {
            this.errorCode = errorCode;
        }
    
    }

    
    /**
     * API统一异常处理类，将异常以XML或者JSON形式返回给用户
     * 
     * @author arganzheng
     * @date 2013-07-25
     */
    public class ApiHandlerExceptionResolver extends AbstractHandlerExceptionResolver {
    
        private final static Logger              log = Logger.getLogger(ApiHandlerExceptionResolver.class);
    
        @Autowired
        @Qualifier("jacksonHttpMessageConverter")
        private MappingJsonpHttpMessageConverter jacksonHttpMessageConverter;
    
        @Resource(name = "xstreamMarshaller")
        private Marshaller                       xstreamMarshaller;
    
        @Override
        protected ModelAndView doResolveException(HttpServletRequest request, HttpServletResponse response, Object handler,
                                                  Exception ex) {
            // 异常日志统一在这里记录
            log.error(ex.getMessage(), ex);
    
            // 如果浏览器想要JSON
            MediaType jsonMediaType = getJsonMediaType(request);
            boolean isJson = jsonMediaType != null || request.getRequestURI().endsWith(Constants.FORMAT_JSON)
                             || Constants.FORMAT_JSON.equalsIgnoreCase(request.getParameter("format"));
            if (isJson) {
                try {
                    writeJsonResponse(ex, jsonMediaType, response);
                    return new ModelAndView(); // ruturn Empty ModelAndView表示到此结束了。
                } catch (Exception e) {
                    log.error("Error rendering json response!", e);
                }
            } else {
                try {
                    writeXmlResponse(ex, jsonMediaType, response);
                    return new ModelAndView(); // ruturn Empty ModelAndView表示到此结束了。
                } catch (Exception e) {
                    log.error("Error rendering xml response!", e);
                }
            }
    
            return null; // 如果自己处理不了，return null让其他的ExceptionResolver处理。
        }
    
        private void writeXmlResponse(Exception ex, MediaType xmlMediaType, HttpServletResponse response)
                                                                                                         throws XmlMappingException,
                                                                                                         IOException {
            response.setContentType("application/xml;charset=utf-8");
            if (ex instanceof ApiException) {
                ApiException rex = (ApiException) ex;
                xstreamMarshaller.marshal(new ApiResponse(rex.getErrorCode(), ex.getMessage()),
                                          new StreamResult(response.getWriter()));
            } else if (ex instanceof IllegalArgumentException || ex instanceof MissingServletRequestParameterException) {
                xstreamMarshaller.marshal(new ApiResponse(ApiErrorCode.BIZ_ARGUMENT_INVALID, "请求参数不合法：" + ex.getMessage()),
                                          new StreamResult(response.getWriter()));
            } else {
                log.error(ex.getMessage(), ex);
                xstreamMarshaller.marshal(ApiResponse.ERROR_UNKNOWN, new StreamResult(response.getWriter()));
            }
        }
    
        private void writeJsonResponse(Exception ex, MediaType jsonMediaType, HttpServletResponse response)
                                                                                                           throws HttpMessageNotWritableException,
                                                                                                           IOException {
            if (jsonMediaType == null) {
                jsonMediaType = MediaType.APPLICATION_JSON;
            }
            response.setContentType("application/xml;charset=utf-8");
            if (ex instanceof ApiException) {
                ApiException rex = (ApiException) ex;
                jacksonHttpMessageConverter.write(new ApiResponse(rex.getErrorCode(), ex.getMessage()), jsonMediaType,
                                                  new ServletServerHttpResponse(response));
            } else if (ex instanceof IllegalArgumentException || ex instanceof MissingServletRequestParameterException) {
                jacksonHttpMessageConverter.write(new ApiResponse(ApiErrorCode.BIZ_ARGUMENT_INVALID, "请求参数不合法："
                                                                                                     + ex.getMessage()),
                                                  jsonMediaType, new ServletServerHttpResponse(response));
            } else {
                log.error(ex.getMessage(), ex);
                jacksonHttpMessageConverter.write(ApiResponse.ERROR_UNKNOWN, jsonMediaType,
                                                  new ServletServerHttpResponse(response));
            }
        }
    
        private MediaType getJsonMediaType(HttpServletRequest request) {
            List<MediaType> acceptedMediaTypes = new ServletServerHttpRequest(request).getHeaders().getAccept();
            if (acceptedMediaTypes != null) {
                for (MediaType mediaType : acceptedMediaTypes) {
                    if (Constants.FORMAT_JSON.equalsIgnoreCase(mediaType.getSubtype())) {
                        return mediaType;
                    }
                }
            }
            return null;
        }
    
    }


另一种实现方式就是自定义映射关系，其实就是把映射关系配置在Spring配置文件了：



    <!-- Allow Exceptions to be handled in annotated methods if desired.  Otherwise fallback to the
         'restExceptionResolver' below: -->
    <bean id="annotationMethodHandlerExceptionResolver"
          class="org.springframework.web.servlet.mvc.annotation.AnnotationMethodHandlerExceptionResolver">
        <property name="order" value="0"/>
    </bean>
    <bean id="restExceptionResolver" class="com.stormpath.spring.web.servlet.handler.RestExceptionHandler">
        <property name="order" value="100"/>
        <property name="messageConverters">
            <list>
                <ref bean="jacksonHttpMessageConverter"/>
            </list>
        </property>
        <property name="errorResolver">
            <bean class="com.stormpath.spring.web.servlet.handler.DefaultRestErrorResolver">
                <property name="localeResolver" ref="localeResolver"/>
                <property name="defaultMoreInfoUrl" value="mailto:support@mycompany.com"/>
                <property name="exceptionMappingDefinitions">
                    <map>
                        <!-- 404 -->
                        <entry key="com.stormpath.blog.spring.mvc.rest.exhandler.UnknownResourceException" value="404, _exmsg"/>

                        <!-- 500 (catch all): -->
                        <entry key="Throwable" value="500"/>
                    </map>
                </property>
            </bean>
        </property>
    </bean>


个人感觉搞的复杂了，也没有太大必要。







