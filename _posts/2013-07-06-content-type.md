---
title: Content-Type头部
layout: post
---


content-type是一个非常重要的HTTP Header，设置错误的话，浏览器端是没有办法正常显示的。作为API容器，需要根据format来定义content-type，并且强制使用UTF-8。

    package me.arganzheng.study.api.container.controller;

    @Controller
    public class ApiDispatcher {
    
        private static final Logger logger         = Logger.getLogger(ApiDispatcher.class);
    
        private static final String FORMAT_DEFAULT = "xml";
    
        @RequestMapping(value = "{moduleName}/{apiName}.xhtml")
        public void dispatch(@PathVariable String moduleName,//
                             @PathVariable String apiName, //
                             HttpServletRequest request,//
                             HttpServletResponse response) {
            Map<String, String> parameterMap = getParameterMap(request.getParameterMap());
            String format = parameterMap.remove("format");
            if (StringUtils.isEmpty(format)) {
                format = FORMAT_DEFAULT;
            }
            String out = StringUtils.EMPTY;
            String apiName4Idl = StringUtils.EMPTY;
            try {
                OpenApiBridge bridge = new OpenApiBridgeImpl();
                apiName4Idl = getApiName4Idl(apiName);
                Service service = getService(apiName4Idl);
                out = bridge.invoke(parameterMap, apiName4Idl, service, format);
            } catch (Exception ex) {
                logger.error("dispatch api failed! moduleName: " + moduleName + ", apiName:" + apiName4Idl, ex);
            }
            
            String contentTypeAndCharset = getContentType(format) + ";charset=UTF-8";
            response.addHeader("Content-Type", contentTypeAndCharset);
            
            try {
                response.getWriter().write(out);
            } catch (IOException e) {
                logger.error("IO Exception thrown when trying to write response", e);
            }
        }
    
        private String getContentType(String format) {
            if (StringUtils.equals(format, "json")) {
                return "application/json";
            } else {
                return "application/xml";
            }
        }
    }
    
**Tips** 这里直接使用了HttpServletResponse，虽然难看了点，但是控制力强，实现简单。这里如果要使用@ResponseBody，需要自定义一个。注意，不能两者同时使用。


### 使用@ResponseBody的编码方式

`@ResponseBody`其实是调用对应的`MessageConverter`来实现消息转换和输入输出的，这里也会有字节流到字符串的问题，需要指定charset。注意，在HTTP中默认的编码不是`UTF-8`，而是`ISO-8859-1`。比如`StringHttpMessageConverter`。如果你数据库和JVM编码都是UTF-8，但是使用了`@ReponseBody` 返回了String对象，那么是一定会有乱码问题的。

    package org.springframework.http.converter;
    
    import java.io.IOException;
    import java.io.UnsupportedEncodingException;
    import java.nio.charset.Charset;
    import java.util.ArrayList;
    import java.util.List;
    
    import org.springframework.http.HttpInputMessage;
    import org.springframework.http.HttpOutputMessage;
    import org.springframework.http.MediaType;
    import org.springframework.util.StreamUtils;
    
    public class StringHttpMessageConverter extends AbstractHttpMessageConverter<String> {
    
        public static final Charset DEFAULT_CHARSET = Charset.forName("ISO-8859-1");
    
        private final Charset defaultCharset;
    
    	private final List<Charset> availableCharsets;
    
    	private boolean writeAcceptCharset = true;
    
    
    	/**
    	 * A default constructor that uses {@code "ISO-8859-1"} as the default charset.
    	 * @see #StringHttpMessageConverter(Charset)
    	 */
    	public StringHttpMessageConverter() {
    		this(DEFAULT_CHARSET);
    	}
    
    	/**
    	 * A constructor accepting a default charset to use if the requested content
    	 * type does not specify one.
    	 */
    	public StringHttpMessageConverter(Charset defaultCharset) {
    		super(new MediaType("text", "plain", defaultCharset), MediaType.ALL);
    		this.defaultCharset = defaultCharset;
    		this.availableCharsets = new ArrayList<Charset>(Charset.availableCharsets().values());
    	}
    
    	private Charset getContentTypeCharset(MediaType contentType) {
    		if (contentType != null && contentType.getCharSet() != null) {
    			return contentType.getCharSet();
    		}
    		else {
    			return this.defaultCharset;
    		}
    	}
    }


关于如何让@ReponseBody相应的`MessageConveter`正确的设置charset参数，参见这篇帖子：[Who sets response content-type in Spring MVC (@ResponseBody)](http://stackoverflow.com/questions/3616359/who-sets-response-content-type-in-spring-mvc-responsebody/3617594#3617594) 写的非常好。
