---
title: Spring MVC的异常处理机制
layout: post
---


Spring MVC在找到你的handler之后，会通过反射调用handler的方法：

	
	public class InvocableHandlerMethod extends HandlerMethod {

		/**
		 * Invoke the handler method with the given argument values.
		 */
		private Object invoke(Object... args) throws Exception {
			ReflectionUtils.makeAccessible(getBridgedMethod());
			try {
				return getBridgedMethod().invoke(getBean(), args);
			}
			catch (IllegalArgumentException ex) {
				assertTargetBean(getBridgedMethod(), getBean(), args);
				throw new IllegalStateException(getInvocationErrorMessage(ex.getMessage(), args), ex);
			}
			catch (InvocationTargetException ex) {
				// Unwrap for HandlerExceptionResolvers ...
				Throwable targetException = ex.getTargetException();
				if (targetException instanceof RuntimeException) {
					throw (RuntimeException) targetException;
				}
				else if (targetException instanceof Error) {
					throw (Error) targetException;
				}
				else if (targetException instanceof Exception) {
					throw (Exception) targetException;
				}
				else {
					String msg = getInvocationErrorMessage("Failed to invoke controller method", args);
					throw new IllegalStateException(msg, targetException);
				}
			}
		}
	}

如果Handler方法抛异常，会被catch住，然后执行相应的异常处理逻辑。


	protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
			HttpServletRequest processedRequest = request;
			HandlerExecutionChain mappedHandler = null;
			boolean multipartRequestParsed = false;

			WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

			try {
				ModelAndView mv = null;
				Exception dispatchException = null;

				try {
					...
					// Determine handler for the current request.
					mappedHandler = getHandler(processedRequest, false);
					...

					// Determine handler adapter for the current request.
					HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

					...
					if (!mappedHandler.applyPreHandle(processedRequest, response)) {
						return;
					}

					try {
						// Actually invoke the handler.
						mv = ha.handle(processedRequest, response, mappedHandler.getHandler());
					}
					finally {
						if (asyncManager.isConcurrentHandlingStarted()) {
							return;
						}
					}

					applyDefaultViewName(request, mv);

					mappedHandler.applyPostHandle(processedRequest, response, mv);
				}
				catch (Exception ex) { // handler抛出的异常会在这里被捕获
					dispatchException = ex;
				}
				// 然后在这里被处理
				processDispatchResult(processedRequest, response, mappedHandler, mv, dispatchException);
			}
			catch (Exception ex) { 
				triggerAfterCompletion(processedRequest, response, mappedHandler, ex);
			}
			catch (Error err) {
				triggerAfterCompletionWithError(processedRequest, response, mappedHandler, err);
			}
			finally {
				if (asyncManager.isConcurrentHandlingStarted()) {
					// Instead of postHandle and afterCompletion
					mappedHandler.applyAfterConcurrentHandlingStarted(processedRequest, response);
					return;
				}
				// Clean up any resources used by a multipart request.
				if (multipartRequestParsed) {
					cleanupMultipart(processedRequest);
				}
			}
		}

handler的异常会在`processDispatchResult(processedRequest, response, mappedHandler, mv, dispatchException);`被处理。

	/**
	 * Handle the result of handler selection and handler invocation, which is
	 * either a ModelAndView or an Exception to be resolved to a ModelAndView.
	 */
	private void processDispatchResult(HttpServletRequest request, HttpServletResponse response,
			HandlerExecutionChain mappedHandler, ModelAndView mv, Exception exception) throws Exception {

		boolean errorView = false;

		if (exception != null) {
			if (exception instanceof ModelAndViewDefiningException) {
				logger.debug("ModelAndViewDefiningException encountered", exception);
				mv = ((ModelAndViewDefiningException) exception).getModelAndView();
			}
			else {
				Object handler = (mappedHandler != null ? mappedHandler.getHandler() : null);
				mv = processHandlerException(request, response, handler, exception);
				errorView = (mv != null);
			}
		}

		// Did the handler return a view to render?
		if (mv != null && !mv.wasCleared()) {
			render(mv, request, response);
			if (errorView) {
				WebUtils.clearErrorRequestAttributes(request);
			}
		}
		....

		if (mappedHandler != null) {
			mappedHandler.triggerAfterCompletion(request, response, null);
		}
	}

mv = processHandlerException(request, response, handler, exception); 这个方法会处理异常，并且返回一个error ModelAndView。如果mv不为空，还会渲染这个mv。

	/**
	 * Determine an error ModelAndView via the registered HandlerExceptionResolvers.
	 * @param request current HTTP request
	 * @param response current HTTP response
	 * @param handler the executed handler, or {@code null} if none chosen at the time of the exception
	 * (for example, if multipart resolution failed)
	 * @param ex the exception that got thrown during handler execution
	 * @return a corresponding ModelAndView to forward to
	 * @throws Exception if no error ModelAndView found
	 */
	protected ModelAndView processHandlerException(HttpServletRequest request, HttpServletResponse response,
			Object handler, Exception ex) throws Exception {

		// Check registered HandlerExceptionResolvers...
		ModelAndView exMv = null;
		for (HandlerExceptionResolver handlerExceptionResolver : this.handlerExceptionResolvers) {
			exMv = handlerExceptionResolver.resolveException(request, response, handler, ex);
			if (exMv != null) {
				break;
			}
		}
		if (exMv != null) {
			if (exMv.isEmpty()) {
				return null;
			}
			// We might still need view name translation for a plain error model...
			if (!exMv.hasView()) {
				exMv.setViewName(getDefaultViewName(request));
			}
			if (logger.isDebugEnabled()) {
				logger.debug("Handler execution resulted in exception - forwarding to resolved error view: " + exMv, ex);
			}
			WebUtils.exposeErrorRequestAttributes(request, ex, getServletName());
			return exMv;
		}

		throw ex;
	}

这个方法是Spring MVC最重要的异常处理函数。让我们仔细看一下它做的事情。

首先找到注册的HandlerExceptionResolver(异常处理器)：

	// Check registered HandlerExceptionResolvers...
	ModelAndView exMv = null;
	for (HandlerExceptionResolver handlerExceptionResolver : this.handlerExceptionResolvers) {
		exMv = handlerExceptionResolver.resolveException(request, response, handler, ex);
		if (exMv != null) {
			break;
		}
	}

然后，调用每个HandlerExceptionResolver的resolveException方法。如果该方法返回的不是null，表示已经处理结束。否则表示交个下一次异常处理器处理。

	package org.springframework.web.servlet;

	public class DispatcherServlet extends FrameworkServlet {

		/** List of HandlerExceptionResolvers used by this servlet */
		private List<HandlerExceptionResolver> handlerExceptionResolvers;


		/** Detect all HandlerExceptionResolvers or just expect "handlerExceptionResolver" bean? */
		private boolean detectAllHandlerExceptionResolvers = true;

		/**
		 * Well-known name for the HandlerExceptionResolver object in the bean factory for this namespace.
		 * Only used when "detectAllHandlerExceptionResolvers" is turned off.
		 * @see #setDetectAllHandlerExceptionResolvers
		 */
		public static final String HANDLER_EXCEPTION_RESOLVER_BEAN_NAME = "handlerExceptionResolver";

		/**
		 * Initialize the HandlerExceptionResolver used by this class.
		 * <p>If no bean is defined with the given name in the BeanFactory for this namespace,
		 * we default to no exception resolver.
		 */
		private void initHandlerExceptionResolvers(ApplicationContext context) {
			this.handlerExceptionResolvers = null;

			if (this.detectAllHandlerExceptionResolvers) {
				// Find all HandlerExceptionResolvers in the ApplicationContext, including ancestor contexts.
				Map<String, HandlerExceptionResolver> matchingBeans = BeanFactoryUtils
						.beansOfTypeIncludingAncestors(context, HandlerExceptionResolver.class, true, false);
				if (!matchingBeans.isEmpty()) {
					this.handlerExceptionResolvers = new ArrayList<HandlerExceptionResolver>(matchingBeans.values());
					// We keep HandlerExceptionResolvers in sorted order.
					OrderComparator.sort(this.handlerExceptionResolvers);
				}
			}
			else {
				try {
					HandlerExceptionResolver her =
							context.getBean(HANDLER_EXCEPTION_RESOLVER_BEAN_NAME, HandlerExceptionResolver.class);
					this.handlerExceptionResolvers = Collections.singletonList(her);
				}
				catch (NoSuchBeanDefinitionException ex) {
					// Ignore, no HandlerExceptionResolver is fine too.
				}
			}

			// Ensure we have at least some HandlerExceptionResolvers, by registering
			// default HandlerExceptionResolvers if no other resolvers are found.
			if (this.handlerExceptionResolvers == null) {
				this.handlerExceptionResolvers = getDefaultStrategies(context, HandlerExceptionResolver.class);
				if (logger.isDebugEnabled()) {
					logger.debug("No HandlerExceptionResolvers found in servlet '" + getServletName() + "': using default");
				}
			}
		}
	}

可以看到默认会查找Spring上下文中所有实现了HandlerExceptionResolver接口的类: 

* HandlerExceptionResolver
	* AbstractHandlerExceptionResolver
		* AbstractHandlerMethodExceptionResolver
			* ExceptionHandlerExceptionResolver
		* AnnotationMethodHandlerExceptionResolver：默认开启，配合@ExceptionHandler注解。@deprecated in favor of ExceptionHandlerExceptionResolver
		* DefaultHandlerExceptionResolver：默认开启，处理标准的Spring异常并且将他们转换为相应的HTTP状态码。
		* ResponseStatusExceptionResolver：默认开启，配合@ResponseStatus注解将异常映射为HTTP状态码。
		* SimpleMappingExceptionResolver
	* HandlerExceptionResolverComposite

经过debug发现，Spring会找到如下异常处理器：

1. org.springframework.web.servlet.mvc.method.annotation.ExceptionHandlerExceptionResolver
2. org.springframework.web.servlet.mvc.annotation.ResponseStatusExceptionResolver
3. org.springframework.web.servlet.mvc.support.DefaultHandlerExceptionResolver
4. me.arganzheng.study.springmvc.common.RestHandlerExceptionResolver

特别需要注意的是HandlerExceptionResolver是有顺序的（实现了Ordered接口），如果没有指定，业务自定义的HandlerExceptionResolver会排在最后。

	package me.arganzheng.study.springmvc.common;

	/**
	 * 统一异常处理类，将异常以JSON形式返回给用户。
	 * 
	 * @author zhengzhibin
	 */
	public class RestHandlerExceptionResolver extends AbstractHandlerExceptionResolver {

		private final static Logger log = Logger.getLogger(RestHandlerExceptionResolver.class);

		@Autowired
		@Qualifier("jsonConverter")
		private MappingJsonpHttpMessageConverter jsonConverter;

		@Override
		protected ModelAndView doResolveException(HttpServletRequest request,
				HttpServletResponse response, Object handler, Exception ex) {
			// 异常日志统一在这里记录
			log.error(ex.getMessage(), ex);

			boolean isApiRequest = false;
			if (HttpServletRequestTool.getRequestURIExcludeContextPath(request).startsWith("/api/")) {
				isApiRequest = true;
			}

			MediaType jsonMediaType = getJsonMediaType(request);
			if (jsonMediaType == null || isApiRequest) {
				jsonMediaType = MediaType.APPLICATION_JSON;
			}
			try {
				writeJsonResponse(ex, response);
			} catch (Exception e) {
				log.error("Error rendering json response!", e);
			}
			// ruturn Empty ModelAndView表示到此结束了。
			return new ModelAndView();
		}

		private MediaType getJsonMediaType(HttpServletRequest request) {
			List<MediaType> acceptedMediaTypes = new ServletServerHttpRequest(request).getHeaders()
					.getAccept();
			if (acceptedMediaTypes != null) {
				for (MediaType mediaType : acceptedMediaTypes) {
					if ("json".equalsIgnoreCase(mediaType.getSubtype())) {
						return mediaType;
					}
				}
			}
			return null;
		}

		private void writeJsonResponse(Exception ex, HttpServletResponse response)
				throws HttpMessageNotWritableException, IOException {
			MediaType jsonMediaType = MediaType.APPLICATION_JSON;

			response.setContentType("application/json;charset=utf-8");

			if (ex instanceof RestException) {
				RestException rex = (RestException) ex;
				jsonConverter.write(new RestError(rex.getErrorCode(), ex.getMessage()), jsonMediaType,
						new ServletServerHttpResponse(response));
			} else if (ex instanceof IllegalArgumentException
					|| ex instanceof MissingServletRequestParameterException) {
				jsonConverter.write(new RestError(RestErrorCode.BIZ_ARGUMENT_INVALID,
						"Invalid Request Parameter: " + ex.getMessage()), jsonMediaType,
						new ServletServerHttpResponse(response));
			} else if (ex instanceof ResourceNotFoundException) {
				jsonConverter.write(new RestError(RestErrorCode.RESOURCE_NOT_FOUND, ex.getMessage()),
						jsonMediaType, new ServletServerHttpResponse(response));
			} else {
				jsonConverter.write(new RestError(RestErrorCode.UNKONW_ERROR, ex.getMessage()),
						jsonMediaType, new ServletServerHttpResponse(response));
			}
		}
	}



推荐阅读
--------

1. [Error Handling for REST with Spring](http://www.baeldung.com/2013/01/31/exception-handling-for-rest-with-spring-3-2/)
2. [Exception Handling in Spring MVC](http://spring.io/blog/2013/11/01/exception-handling-in-spring-mvc)

