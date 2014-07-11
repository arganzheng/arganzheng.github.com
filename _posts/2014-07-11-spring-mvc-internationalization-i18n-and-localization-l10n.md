---
title: Spring MVC国际化和本地化
layout: post
---

在你的mvc-dispatcher-servlet.xml中配置如下：

	<!-- i18n && l10n -->
	<bean id="messageSource"
		class="org.springframework.context.support.ReloadableResourceBundleMessageSource">
		<property name="basename" value="classpath:i18n/messages" />
		<property name="defaultEncoding" value="UTF-8" />
	</bean>

	<bean id="localeResolver"
		class="org.springframework.web.servlet.i18n.CookieLocaleResolver">
		<property name="defaultLocale" value="en" />
		<property name="cookieName" value="localeCookie" />
		<property name="cookieMaxAge" value="3600" />
	</bean>

	<!-- Configures Handler Interceptors -->
	<mvc:interceptors>
		<mvc:interceptor>
			<mvc:mapping path="/**" />
			<mvc:exclude-mapping path="/resources/**" />
			<bean class="org.springframework.web.servlet.i18n.LocaleChangeInterceptor">
				<property name="paramName" value="language" />
			</bean>
		</mvc:interceptor>
	</mvc:interceptors>


**说明**

1. 注意到我们这里的basename配置的是`classpath:i18n/messages`，那么Spring会加载classpath中的i18n目录下的所有messages_${locale}.xml文件。
2. LocaleChangeInterceptor的配置表示，如果用户用`language=xxx`查询字符串指定了locale，那么就会使用这个locale，这就是所谓的切换语言功能。

但是用户刚进来的时候语言是怎么确定的呢？一般有如下几种处理方式：

1. 显示en，因为毫无争议的，en是国际通用语言。
2. 尝试判断用户的locale。一般有如下两种方式：
	1. 根据HTTP的accept-language头部判断
	2. 根据IP得到country，再根据country判断language

根据accept-language，笔者试验了一下，不是很准确。根据IP会比较准确，但是country到language的映射比较痛苦。不过可以只覆盖重要的国家。

IP得到country可以通过这个得到：


	package me.arganzheng.study.springmvc.i18n.gateway;

	import java.io.IOException;
	import java.util.Map;

	import org.apache.log4j.Logger;
	import org.codehaus.jackson.map.ObjectMapper;
	import org.springframework.stereotype.Component;

	import me.arganzheng.study.springmvc.i18n.utils.WebUtils;

	/**
	 * API查询网关
	 * 
	 */
	@Component
	public class GeoipGateway {

		private static final Logger logger = Logger.getLogger(GeoipGateway.class);

		private static final String API_URL = "http://hwww.telize.com/geoip/";

		public String getCountryCodeByIp(String ip) {
			try {
				String url = API_URL + ip;
				String geoipJson = WebUtils.doGet(url, null, null);

				ObjectMapper mapper = new ObjectMapper();
				Map geoipMap = mapper.readValue(geoipJson, Map.class);
				return (String) geoipMap.get("country_code");
			} catch (IOException e) {
				logger.error("getCountryCodeByIp failed! Ip=" + ip, e);
			}
			return null;
		}

		public static void main(String[] args) throws IOException {
			GeoipGateway gw = new GeoipGateway();
			System.out.println(gw.getCountryCodeByIp("210.227.234.38"));
		}
	}

然后将country code转换为language:

	package me.arganzheng.study.springmvc.i18n.web;

	import java.io.File;
	import java.io.FileFilter;
	import java.io.FileInputStream;
	import java.io.FileNotFoundException;
	import java.io.IOException;
	import java.io.InputStream;
	import java.util.Arrays;
	import java.util.HashMap;
	import java.util.Map;

	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpServletResponse;

	import org.apache.commons.io.FilenameUtils;
	import org.apache.commons.io.IOUtils;
	import org.apache.commons.io.comparator.LastModifiedFileComparator;
	import org.apache.commons.io.filefilter.WildcardFileFilter;
	import org.apache.log4j.Logger;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.beans.factory.annotation.Value;
	import org.springframework.stereotype.Controller;
	import org.springframework.ui.Model;
	import org.springframework.util.StringUtils;
	import org.springframework.web.bind.annotation.RequestMapping;
	import org.springframework.web.bind.annotation.RequestParam;
	import org.springframework.web.bind.annotation.ResponseBody;
	import org.springframework.web.servlet.LocaleResolver;
	import org.springframework.web.servlet.support.RequestContextUtils;

	import me.arganzheng.study.springmvc.i18n.gateway.GeoipGateway;
	import me.arganzheng.study.springmvc.i18n.utils.HttpServletRequestUtils;

	/**
	 * 官网
	 * 
	 * @author zhengzhibin
	 */
	@Controller
	@RequestMapping(value = "/official")
	public class OfficialWebsiteController {
		private static final Logger logger = Logger.getLogger(OfficialWebsiteController.class);

		private static final String DEFAULT_LANGUAGE = "en";

		@Value("${cdnUrl}")
		private String cdnUrl = "http://s.mobile-global.baidu.com/mbrowser";

		private static final String classesDir = FilenameUtils.separatorsToSystem(getClassesDir());
		private static final String apkDir = FilenameUtils.separatorsToSystem(classesDir
				+ "/META-INF/apk");

		/**
		 * get class abs path
		 * 
		 * @return
		 */
		private static String getClassesDir() {
			String path = OfficialWebsiteController.class
					.getResource("OfficialWebsiteController.class").getFile();
			path = path.substring(0, path.length() - OfficialWebsiteController.class.getName().length()
					- 7);
			return new File(path).getAbsolutePath();
		}

		@Autowired
		private GeoipGateway geoipGateway;

		private static Map<String, String> countryCodeToLanguageMapping = new HashMap<String, String>();
		static {
			countryCodeToLanguageMapping.put("ID", "id");
			countryCodeToLanguageMapping.put("TH", "th");
			countryCodeToLanguageMapping.put("PT", "pt");
			countryCodeToLanguageMapping.put("BR", "pt");
			countryCodeToLanguageMapping.put("EG", "ar");
			countryCodeToLanguageMapping.put("SA", "ar");
			countryCodeToLanguageMapping.put("IN", "en_in");
			countryCodeToLanguageMapping.put("JP", "ja");
		}

		/**
		 * @param language 用户指定语言（一般是切换语言）
		 */
		@RequestMapping(value = "/home")
		public String viewOfficialWebsite(HttpServletRequest request, HttpServletResponse response,
				@RequestParam(value = "language", required = false) String language, Model model) {
			// 根据请求IP判断用户的locale
			if (StringUtils.isEmpty(language)) {
				language = DetermineLanguage(request);
				// Set locale to request context
				setLocale(request, response, language);
			}

			model.addAttribute("language", language);

			// 返回相应语言的官网页面
			return "official";
		}

		private void setLocale(HttpServletRequest request, HttpServletResponse response,
				String newLocale) {
			LocaleResolver localeResolver = RequestContextUtils.getLocaleResolver(request);
			if (localeResolver == null) {
				throw new IllegalStateException(
						"No LocaleResolver found: not in a DispatcherServlet request?");
			}
			localeResolver.setLocale(request, response, StringUtils.parseLocaleString(newLocale));
		}

		private String DetermineLanguage(HttpServletRequest request) {
			String ip = HttpServletRequestUtils.getClientIp(request);
			String countryCode = geoipGateway.getCountryCodeByIp(ip);
			String language = countryCodeToLanguageCode(countryCode);
			if (StringUtils.isEmpty(language)) {
				return DEFAULT_LANGUAGE;
			}
			return language;
		}

		private String countryCodeToLanguageCode(String countryCode) {
			return countryCodeToLanguageMapping.get(countryCode);
		}

		/**
		 * 下载APK。处理CDN的真正下载请求。其实应该是在静态资源服务器和nginx处理。
		 */
		@RequestMapping("/download")
		public void downloadApk(HttpServletResponse response) {
			File file = getLatestApkFile();
			if(file==null){
				logger.error("No apk file for download!!!");
				return;
			}
			
			InputStream inStream = null;
			try {
				inStream = new FileInputStream(file);
				String fileName = file.getName();
				response.setHeader("Content-Disposition", "attachment; filename=" + fileName); // HttpServletResponse
				response.addHeader("Content-Length", "" + inStream.available());
				response.setContentType("application/octet-stream; charset=UTF-8");
				// 后面再来考虑支持断点续传。
				IOUtils.copy(inStream, response.getOutputStream());
			} catch (FileNotFoundException e) {
				logger.error(e.getMessage() + "; FilePath=" + file.getName());
			} catch (IOException e) {
				logger.error("download apk failed!", e);
			} finally {
				IOUtils.closeQuietly(inStream);
			}
		}
		
		/**
		 * Google Play Download 打点
		 */
		@RequestMapping("/download/nop")
		@ResponseBody
		public boolean downloadApk() {
			// NOP
			logger.info("Jump to Google Play for download!");
			return true;
		}

		private File getLatestApkFile() {
			File dir = new File(apkDir);
			FileFilter fileFilter = new WildcardFileFilter("*.apk");
			File[] files = dir.listFiles(fileFilter);

			if (files == null || files.length == 0) {
				return null;
			}

			/** The newest file comes first **/
			Arrays.sort(files, LastModifiedFileComparator.LASTMODIFIED_REVERSE);

			return files[files.length - 1];
		}
	}

	public class HttpServletRequestUtils {

		public static String getClientIp(HttpServletRequest request) {
			String clientIp = request.getHeader("X-Real-IP");
			if (clientIp == null) {
				clientIp = request.getRemoteAddr();
			}
			if (clientIp == null) {
				clientIp = "127.0.0.1";
			}

			return clientIp;
		}
	}

参考文章
--------

1. [Spring MVC Internationalization (i18n) and Localization (L10n) Example](http://www.journaldev.com/2610/spring-mvc-internationalization-i18n-and-localization-l10n-example)