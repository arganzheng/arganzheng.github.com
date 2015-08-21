---
title: 怎样获取form-data方式POST的数据
layout: post
---

昨天君仔问我说遇到一个诡异的问题，他有一个controller方法，用x-www-form-urlencoded方式提交表单就可以正常绑定数据，但是用form-data方式提交就绑定不了：

	@Controller
	@RequestMapping("/tag")
	public class TagController {

	    @Autowired
	    private NanTianMenFacade nanTianMenFacade;

	    @RequestMapping(value = "/uploadTags")
	    @ResponseBody
	    public TagUploadResult uploadTags(TagParams params) {
	        String tagIds = params.getTagIds();
	        String uuid = params.getUuid();
	        TagUploadResult result = null;
	        try {
	            result = nanTianMenFacade.uploadTags(tagIds, uuid);
	        } catch (Exception e) {
	            result = new TagUploadResult();
	            result.setS(0);
	        }
	        return result;
	    }
	}

对于我来说，一直都是用x-www-form-urlencoded提交表单数据，除非是要上传文件，才会使用form-data方式。也一直以为Spring对不同的提交方式应该是透明的。不过确实用form-data提交普通表单，确实是拿不到提交的数据。DEBUG发现，`request.getParameterMap()`返回的就是空的！

然后发现`org.apache.commons.io.IOUtils.toString(request.getInputStream());`其实是有返回提交的数据的：

	------WebKitFormBoundaryd7FGw9vQyyFIAYd7
	Content-Disposition: form-data; name="uuid"

	123456
	------WebKitFormBoundaryd7FGw9vQyyFIAYd7
	Content-Disposition: form-data; name="tagIds"

	36
	------WebKitFormBoundaryd7FGw9vQyyFIAYd7
	Content-Disposition: form-data; name="aaa"

	bbb
	------WebKitFormBoundaryd7FGw9vQyyFIAYd7--

也就是数据其实是有正常提交到后台的。那么就是这种方式提交的数据，并没有放在pararmeterMap中了。而是要自己去解析这个inputstream的内容，再放回parameterMap中，这样Spring才能够绑定？这个不就是`multipartResolver`做的事情吗？试着配置了一下，发现果然可以了：

	<bean id="multipartResolver"
		class="org.springframework.web.multipart.commons.CommonsMultipartResolver">
		<property name="maxUploadSize" value="10000000"></property>
	</bean>


然后深入Spring MVC的代码，其实是做了一个判断，如果是Multipart，那么就会进行解析，然后将解析结果放回parameterMap中：

	package org.springframework.web.servlet;

	public class DispatcherServlet extends FrameworkServlet {

		protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
			HttpServletRequest processedRequest = request;
			HandlerExecutionChain mappedHandler = null;
			boolean multipartRequestParsed = false;

			WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

			try {
				ModelAndView mv = null;
				Exception dispatchException = null;

				try {
					processedRequest = checkMultipart(request);
					multipartRequestParsed = processedRequest != request;

					// Determine handler for the current request.
					mappedHandler = getHandler(processedRequest, false);
					if (mappedHandler == null || mappedHandler.getHandler() == null) {
						noHandlerFound(processedRequest, response);
						return;
					}

					// Determine handler adapter for the current request.
					HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

					// Process last-modified header, if supported by the handler.
					String method = request.getMethod();
					boolean isGet = "GET".equals(method);
					if (isGet || "HEAD".equals(method)) {
						long lastModified = ha.getLastModified(request, mappedHandler.getHandler());
						if (logger.isDebugEnabled()) {
							logger.debug("Last-Modified value for [" + getRequestUri(request) + "] is: " + lastModified);
						}
						if (new ServletWebRequest(request, response).checkNotModified(lastModified) && isGet) {
							return;
						}
					}

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
				catch (Exception ex) {
					dispatchException = ex;
				}
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

		/**
		 * Convert the request into a multipart request, and make multipart resolver available.
		 * <p>If no multipart resolver is set, simply use the existing request.
		 * @param request current HTTP request
		 * @return the processed request (multipart wrapper if necessary)
		 * @see MultipartResolver#resolveMultipart
		 */
		protected HttpServletRequest checkMultipart(HttpServletRequest request) throws MultipartException {
			if (this.multipartResolver != null && this.multipartResolver.isMultipart(request)) {
				if (request instanceof MultipartHttpServletRequest) {
					logger.debug("Request is already a MultipartHttpServletRequest - if not in a forward, " +
							"this typically results from an additional MultipartFilter in web.xml");
				}
				else {
					return this.multipartResolver.resolveMultipart(request);
				}
			}
			// If not returned before: return original request.
			return request;
		}
	}


	package org.springframework.web.multipart.commons;

	public class CommonsMultipartResolver extends CommonsFileUploadSupport
			implements MultipartResolver, ServletContextAware {

		public MultipartHttpServletRequest resolveMultipart(final HttpServletRequest request) throws MultipartException {
			Assert.notNull(request, "Request must not be null");
			if (this.resolveLazily) {
				return new DefaultMultipartHttpServletRequest(request) {
					@Override
					protected void initializeMultipart() {
						MultipartParsingResult parsingResult = parseRequest(request);
						setMultipartFiles(parsingResult.getMultipartFiles());
						setMultipartParameters(parsingResult.getMultipartParameters());
						setMultipartParameterContentTypes(parsingResult.getMultipartParameterContentTypes());
					}
				};
			}
			else {
				MultipartParsingResult parsingResult = parseRequest(request);
				return new DefaultMultipartHttpServletRequest(request, parsingResult.getMultipartFiles(),
						parsingResult.getMultipartParameters(), parsingResult.getMultipartParameterContentTypes());
			}
		}


		/**
		 * Parse the given servlet request, resolving its multipart elements.
		 * @param request the request to parse
		 * @return the parsing result
		 * @throws MultipartException if multipart resolution failed.
		 */
		@SuppressWarnings("unchecked")
		protected MultipartParsingResult parseRequest(HttpServletRequest request) throws MultipartException {
			String encoding = determineEncoding(request);
			FileUpload fileUpload = prepareFileUpload(encoding);
			try {
				List<FileItem> fileItems = ((ServletFileUpload) fileUpload).parseRequest(request);
				return parseFileItems(fileItems, encoding);
			}
			catch (FileUploadBase.SizeLimitExceededException ex) {
				throw new MaxUploadSizeExceededException(fileUpload.getSizeMax(), ex);
			}
			catch (FileUploadException ex) {
				throw new MultipartException("Could not parse multipart servlet request", ex);
			}
		}
	}

	package org.apache.commons.fileupload;

	public abstract class FileUploadBase {


		/**
	     * Processes an <a href="http://www.ietf.org/rfc/rfc1867.txt">RFC 1867</a>
	     * compliant <code>multipart/form-data</code> stream.
	     *
	     * @param request The servlet request to be parsed.
	     *
	     * @return A list of <code>FileItem</code> instances parsed from the
	     *         request, in the order that they were transmitted.
	     *
	     * @throws FileUploadException if there are problems reading/parsing
	     *                             the request or storing files.
	     */
	    @Override
	    public List<FileItem> parseRequest(HttpServletRequest request)
	    throws FileUploadException {
	        return parseRequest(new ServletRequestContext(request));
	    }


	    /**
	     * Processes an <a href="http://www.ietf.org/rfc/rfc1867.txt">RFC 1867</a>
	     * compliant <code>multipart/form-data</code> stream.
	     *
	     * @param ctx The context for the request to be parsed.
	     *
	     * @return A list of <code>FileItem</code> instances parsed from the
	     *         request, in the order that they were transmitted.
	     *
	     * @throws FileUploadException if there are problems reading/parsing
	     *                             the request or storing files.
	     */
	    public List<FileItem> parseRequest(RequestContext ctx)
	            throws FileUploadException {
	        List<FileItem> items = new ArrayList<FileItem>();
	        boolean successful = false;
	        try {
	            FileItemIterator iter = getItemIterator(ctx);
	            FileItemFactory fac = getFileItemFactory();
	            if (fac == null) {
	                throw new NullPointerException("No FileItemFactory has been set.");
	            }
	            while (iter.hasNext()) {
	                final FileItemStream item = iter.next();
	                // Don't use getName() here to prevent an InvalidFileNameException.
	                final String fileName = ((FileItemIteratorImpl.FileItemStreamImpl) item).name;
	                FileItem fileItem = fac.createItem(item.getFieldName(), item.getContentType(),
	                                                   item.isFormField(), fileName);
	                items.add(fileItem);
	                try {
	                    Streams.copy(item.openStream(), fileItem.getOutputStream(), true);
	                } catch (FileUploadIOException e) {
	                    throw (FileUploadException) e.getCause();
	                } catch (IOException e) {
	                    throw new IOFileUploadException(format("Processing of %s request failed. %s",
	                                                           MULTIPART_FORM_DATA, e.getMessage()), e);
	                }
	                final FileItemHeaders fih = item.getHeaders();
	                fileItem.setHeaders(fih);
	            }
	            successful = true;
	            return items;
	        } catch (FileUploadIOException e) {
	            throw (FileUploadException) e.getCause();
	        } catch (IOException e) {
	            throw new FileUploadException(e.getMessage(), e);
	        } finally {
	            if (!successful) {
	                for (FileItem fileItem : items) {
	                    try {
	                        fileItem.delete();
	                    } catch (Throwable e) {
	                        // ignore it
	                    }
	                }
	            }
	        }
	    }
	}

