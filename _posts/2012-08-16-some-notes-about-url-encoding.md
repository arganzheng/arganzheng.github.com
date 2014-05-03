---
title: URL encoding学习笔记
layout: post
---


最近在搞一个Open API在线测试工具，出现一些问题，因此接触到了URL Encoding相关方面的东西。

### 首先要了解一下什么是URL Encoding：

> URL Encoding is the process of converting string into valid URL format.  Valid URL format means that the URL contains only what is termed "alpha | digit | safe | extra | escape" characters. 

### 然后了解一下为什么需要URL Encoding：

> URLs can only be sent over the Internet using the ASCII character-set.
> 
> Since URLs often contain characters outside the ASCII set, the URL has to be converted into a valid ASCII format.
>
> URL encoding replaces unsafe ASCII characters with a "%" followed by two hexadecimal digits.
> URLs cannot contain spaces. URL encoding normally replaces a space with a + sign.
>
> URL encoding is normally performed to convert data passed via html forms, because such data may contain special character, such as "/", ".", "#", and so on, which could either: a) have special meanings; or b) is not a valid character for an URL; or c) could be altered during transfer.   For instance, the "#" character needs to be encoded because it has a special meaning of that of an html anchor.   The <space> character also needs to be encoded because is not allowed on a valid URL format.   Also, some characters, such as "~" might not transport properly across the internet.


关键一点是要注意form POST其实也是需要URL encoding的。


### 如何URL encoding

在JavaScript中有内建的系统函数可以调用：`encodeURI(string)`。但是需要注意的是这个函数并不会编码'&'符号，因为它在URL中是一个合法字符，作为各个参数的分隔符。另外，它把空格编码为%20，而不是'+'（两者都是合法的）。

但是对于post参数，或者ajax请求，直接使用encodeURI()是不行的，需要调用`escape(string)`来进行url encoding，需要注意的是这个函数并不编码'/'和'+'符号（正如encodingURI不编码'&'一样），所以需要自己定义一下：

     function urlencoding()                  
     {                  
          var inputString=document.forms["TestEncodingForm"]["inputString"].value;
          var encodedInputString=escape(inputString);
          encodedInputString=encodedInputString.replace("+", "%2B");
          encodedInputString=encodedInputString.replace("/", "%2F");
          document.forms["TestEncodingForm"]["encodedInputString"].value=encodedInputString;
     }

实际上是两个步骤：

1. Convert the character string into a sequence of bytes using the UTF-8 encoding
2. Convert each byte that is not an ASCII letter or digit to %HH, where HH is the hexadecimal value of the byte

对于非Ajax的GET/POST请求，浏览器其实都是会自动做url encoding的：

     <html>
          <head></head>
         <body>
               <form action="/apitools/apiTools.xhtml" method="get">
                      First name: <input type="text" name="fname" /><br />
                      Last name: <input type="text" name="lname" /><br />
                      <input type="submit" value="Submit" />
               </form>
          </body>
     </html>

如果在第一个文本框输入`hello world`，在第二个输入`you & me`.

会请求如下URL：

     http://localhost/apitools/apiTools.xhtml?fname=hello+world&lname=you+%26+me

在这里：空格被编码为"+"，而'&'字符被编码为"%26"。基本上，字符URL编码后的格式就是%XY，其实X和Y都是数字。

POST请求浏览器也会对参数值进行encoding，否则会出现揭断问题。但是如果是ajax请求就不会了，需要自己处理。

### 在Java中如何正确解码用Javascript:escape()编码的中文字符

直接用URLDecoder.decode解码JS编码过的中文字符是会抛异常的，需要自己解析。

具体参见：http://blog.csdn.net/hbzyaxiu520/article/details/5607873

     package me.arganzheng.study.utils;

     /**
     * JavaScript escape/unescape 编码的 Java 实现 author jackyz keep this copyright info
     * while using this method by free
     */
     public class EscapeUtil {

          private static final String[] hex = { "00", "01", "02", "03", "04", "05",
                    "06", "07", "08", "09", "0A", "0B", "0C", "0D", "0E", "0F", "10",
                    "11", "12", "13", "14", "15", "16", "17", "18", "19", "1A", "1B",
                    "1C", "1D", "1E", "1F", "20", "21", "22", "23", "24", "25", "26",
                    "27", "28", "29", "2A", "2B", "2C", "2D", "2E", "2F", "30", "31",
                    "32", "33", "34", "35", "36", "37", "38", "39", "3A", "3B", "3C",
                    "3D", "3E", "3F", "40", "41", "42", "43", "44", "45", "46", "47",
                    "48", "49", "4A", "4B", "4C", "4D", "4E", "4F", "50", "51", "52",
                    "53", "54", "55", "56", "57", "58", "59", "5A", "5B", "5C", "5D",
                    "5E", "5F", "60", "61", "62", "63", "64", "65", "66", "67", "68",
                    "69", "6A", "6B", "6C", "6D", "6E", "6F", "70", "71", "72", "73",
                    "74", "75", "76", "77", "78", "79", "7A", "7B", "7C", "7D", "7E",
                    "7F", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
                    "8A", "8B", "8C", "8D", "8E", "8F", "90", "91", "92", "93", "94",
                    "95", "96", "97", "98", "99", "9A", "9B", "9C", "9D", "9E", "9F",
                    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "AA",
                    "AB", "AC", "AD", "AE", "AF", "B0", "B1", "B2", "B3", "B4", "B5",
                    "B6", "B7", "B8", "B9", "BA", "BB", "BC", "BD", "BE", "BF", "C0",
                    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CA", "CB",
                    "CC", "CD", "CE", "CF", "D0", "D1", "D2", "D3", "D4", "D5", "D6",
                    "D7", "D8", "D9", "DA", "DB", "DC", "DD", "DE", "DF", "E0", "E1",
                    "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "EA", "EB", "EC",
                    "ED", "EE", "EF", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
                    "F8", "F9", "FA", "FB", "FC", "FD", "FE", "FF" };
          private static final byte[] val = { 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x00, 0x01,
                    0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
                    0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F };

          public static String escape(String s) {
               StringBuffer sbuf = new StringBuffer();
               int len = s.length();
               for (int i = 0; i < len; i++) {
                    int ch = s.charAt(i);
                    if (ch == ' ') { // space : map to '+'
                         sbuf.append('+');
                    } else if ('A' <= ch && ch <= 'Z') { // 'A'..'Z' : as it was
                         sbuf.append((char) ch);
                    } else if ('a' <= ch && ch <= 'z') { // 'a'..'z' : as it was
                         sbuf.append((char) ch);
                    } else if ('0' <= ch && ch <= '9') { // '0'..'9' : as it was
                         sbuf.append((char) ch);
                    } else if (ch == '-'
                              || ch == '_' // unreserved : as it was
                              || ch == '.' || ch == '!' || ch == '~' || ch == '*'
                              || ch == '/' || ch == '(' || ch == ')') {
                         sbuf.append((char) ch);
                    } else if (ch <= 0x007F) { // other ASCII : map to %XX
                         sbuf.append('%');
                         sbuf.append(hex[ch]);
                    } else { // unicode : map to %uXXXX
                         sbuf.append('%');
                         sbuf.append('u');
                         sbuf.append(hex[(ch >>> 8)]);
                         sbuf.append(hex[(0x00FF & ch)]);
                    }
               }
               return sbuf.toString();
          }

          public static String unescape(String s) {
               StringBuffer sbuf = new StringBuffer();
               int i = 0;
               int len = s.length();
               while (i < len) {
                    int ch = s.charAt(i);
                    if (ch == '+') { // + : map to ' '
                         sbuf.append(' ');
                    } else if ('A' <= ch && ch <= 'Z') { // 'A'..'Z' : as it was
                         sbuf.append((char) ch);
                    } else if ('a' <= ch && ch <= 'z') { // 'a'..'z' : as it was
                         sbuf.append((char) ch);
                    } else if ('0' <= ch && ch <= '9') { // '0'..'9' : as it was
                         sbuf.append((char) ch);
                    } else if (ch == '-'
                              || ch == '_' // unreserved : as it was
                              || ch == '.' || ch == '!' || ch == '~' || ch == '*'
                              || ch == '/' || ch == '(' || ch == ')') {
                         sbuf.append((char) ch);
                    } else if (ch == '%') {
                         int cint = 0;
                         if ('u' != s.charAt(i + 1)) { // %XX : map to ascii(XX)
                              cint = (cint << 4) | val[s.charAt(i + 1)];
                              cint = (cint << 4) | val[s.charAt(i + 2)];
                              i += 2;
                         } else { // %uXXXX : map to unicode(XXXX)
                              cint = (cint << 4) | val[s.charAt(i + 2)];
                              cint = (cint << 4) | val[s.charAt(i + 3)];
                              cint = (cint << 4) | val[s.charAt(i + 4)];
                              cint = (cint << 4) | val[s.charAt(i + 5)];
                              i += 5;
                         }
                         sbuf.append((char) cint);
                    }
                    i++;
               }
               return sbuf.toString();
          }
     }


补记：2012-08-23 星期四 晴朗
--------------------------

在线测试工具发布之后，发现了一个bug：当请求form表单中字段包含中文时候，用post方式提交ajax请求，用httpclient3.0.1请求OpenAPI接口会出现乱码。改成httpclient4.0.1就没有问题。非常诡异。怀疑是老版本的一个bug。
expressName=顺丰 理论上来来说应该是编码成 expressName=%25E9%25A1%25BA%25E9%25A3%258E

但是返回的response确实返回如下信息

     <?xml version="1.0" encoding="UTF-8"?>
     <signShipV2>
     <errorCode>83</errorCode>
          <errorMessage>83 FILE:appauthoidbnao.cpp LN:932 FUN:CheckSig EN:83 EM:OIDBAccessAuto_0x720 Return Failed!RetCode = 83;ErrorMsg:????HMAC?????? / check sign from oidb faile!
     appIdOauth=700045336
     randomValue=92534
     sourceString=POST&%2Fdeal%2FsignShipV2.xhtml&accessToken%3Df54ecc636ba4095ac7914f47bcd6039e%26appOAuthID%3D700045336%26cooperatorId%3D795019780%26dealId%3D795019780-20120816-8997719%26expressCompanyId%3D003shunfeng%26expressDealId%3D1224-12345-123485%26expressName=%3F%3F%26format%3Dxml%26randomValue%3D92534%26timeStamp%3D1345440378678%26transportType%3D1%26uin%3D795019780
     timeStamp=1345440419
     </errorMessage>
     </signShipV2>

注意到expressName=%3F%3F%26，这就是为什么它老师sign验证失败的原因了。

我原来的使用httpClient发http请求的代码如下：

     package me.arganzheng.study.utils;

     import java.io.BufferedReader;
     import java.io.IOException;
     import java.io.InputStream;
     import java.io.InputStreamReader;
     import java.util.Map;
     import java.util.Map.Entry;

     import org.apache.commons.httpclient.Header;
     import org.apache.commons.httpclient.HttpClient;
     import org.apache.commons.httpclient.HttpMethod;
     import org.apache.commons.httpclient.NameValuePair;
     import org.apache.commons.httpclient.SimpleHttpConnectionManager;
     import org.apache.commons.httpclient.URI;
     import org.apache.commons.httpclient.methods.GetMethod;
     import org.apache.commons.httpclient.methods.PostMethod;
     import org.apache.log4j.Logger;

     /**
     *
     * @author arganzheng
     * @date 2012-08-10
     */
     public class HttpClientWrapper {

          public static final Logger logger = Logger
                    .getLogger(HttpClientWrapper.class);

          private static final String DEFAULT_ENCODING = "utf-8";

          // 让connectionmanager管理httpclientconnection时是否关闭连接
          // 参见：http://fuyun.org/2009/09/connection-close-in-httpclient/
          private static boolean alwaysClose = false;
          private final HttpClient client = new HttpClient(
                    new SimpleHttpConnectionManager(alwaysClose));

          public HttpClient getHttpClient() {
               return client;
          }

          /**
          * @param url
          *            请求的资源ＵＲＬ
          * @param header
          *            request请求时附带的头信息(header) 没有时传null
          * @param postData
          *            POST请求时form表单封装的数据 没有时传null
          * @param encoding
          *            response返回的信息编码格式 传null将采用utf-8编码
          * @return response返回的文本数据
          *
          * @note 不保留http连接，每次请求结束都会关闭链接。
          */
          public <K, V> String sendRequest(String url, Map<K, V> header,
                    Map<K, V> postData, String encoding) {
               String responseString = null;
               // 头部请求信息
               Header[] headers = null;
               if (header != null) {

                    headers = new Header[header.size()];
                    int i = 0;
                    for (Entry<K, V> e : header.entrySet()) {
                         headers[i++] = new Header(e.getKey().toString(), e.getValue()
                                   .toString());
                    }
               }
               URI uri = null;
               try {
                    uri = new URI(url.trim(), true);
                    if (postData == null) { // get方式
                         GetMethod getRequest = new GetMethod();
                         getRequest.setURI(uri);
                         if (headers != null) {
                              for (int i = 0; i < headers.length; i++) {
                                   getRequest.setRequestHeader(headers[i]);
                              }
                         }
                         responseString = this.executeMethodAndReleaseConnection(
                                   getRequest, encoding);
                    } else {// post方式
                         PostMethod postRequest = new PostMethod();
                         postRequest.setURI(uri);
                         if (headers != null) {
                              for (int i = 0; i < headers.length; i++) {
                                   postRequest.setRequestHeader(headers[i]);
                              }
                         }
                         int dataLength = postData.size();
                         NameValuePair[] params = new NameValuePair[dataLength];
                         int i = 0;
                         for (Entry<K, V> e : postData.entrySet()) {
                              params[i++] = new NameValuePair(e.getKey().toString(), e
                                        .getValue().toString());
                         }

                         postRequest.setRequestBody(params);

                         responseString = this.executeMethodAndReleaseConnection(
                                   postRequest, encoding);
                    }
               } catch (Exception e) {
                    throw new RuntimeException("sendRequest error!", e);
               }

               return responseString;
          }

          private String executeMethodAndReleaseConnection(HttpMethod request,
                    String encoding) {
               String responseString = null;
               try {
                    responseString = this.executeMethod(request, encoding);
               } catch (RuntimeException e) {
                    throw e;
               } finally {
                    request.releaseConnection();
               }
               return responseString;
          }

          private String executeMethod(HttpMethod request, String encoding) {
               String responseContent = null;
               InputStream responseStream = null;
               BufferedReader rd = null;
               try {
                    this.getHttpClient().executeMethod(request);
                    if (encoding == null) {
                         encoding = DEFAULT_ENCODING;
                    }

                    responseStream = request.getResponseBodyAsStream();
                    rd = new BufferedReader(new InputStreamReader(responseStream,
                              encoding));
                    String tempLine = rd.readLine();
                    StringBuffer tempStr = new StringBuffer();
                    String crlf = System.getProperty("line.separator");
                    while (tempLine != null) {
                         tempStr.append(tempLine);
                         tempStr.append(crlf);
                         tempLine = rd.readLine();
                    }
                    responseContent = tempStr.toString();

                    Header locationHeader = request.getResponseHeader("location");
                    // 返回代码为302,301时，表示页面己经重定向，则重新请求location的url，这在
                    // 一些登录授权取cookie时很重要
                    if (locationHeader != null) {
                         String redirectUrl = locationHeader.getValue();
                         this.sendRequest(redirectUrl, null, null, null);
                    }
               } catch (Exception e) {
                    throw new RuntimeException(e);
               } finally {
                    if (rd != null)
                         try {
                              rd.close();
                         } catch (IOException e) {
                              logger.error("");
                         }
                    if (responseStream != null)
                         try {
                              responseStream.close();
                         } catch (IOException e) {
                              throw new RuntimeException(e.getMessage());

                         }
               }
               return responseContent;
          }
     }

POST请求的关键代码展开就是对commons-httpclient3.1的如下几个调用：

     PostMethod postRequest = new PostMethod();
     postRequest.setURI(uri);
     // 设置header..
     postRequest.setRequestBody(params);

     this.getHttpClient().executeMethod(request);
    
     /// ... 根据encoding抽取结果 ...
         
但是发现commons-httpclients3.1的executeMethod(PostMethod)并没有对params正确编码，因为它没有地方让你指定encoding，你需要自己先对form表单进行url encoding，就像发送get请求一样。但是笔者发现即使对其用utf-8编码传递，仍然有问题，貌似它会再次按照ISO-8859-1进行编码，这样进过两次编码之后就变成乱码了。

改成httpclient-4.0.1的UrlEncodedFormEntity就没有问题了，它可以让你指定encoding：

     package me.arganzheng.study.utils;

     import java.io.BufferedReader;
     import java.io.IOException;
     import java.io.InputStream;
     import java.io.InputStreamReader;
     import java.util.ArrayList;
     import java.util.Arrays;
     import java.util.List;
     import java.util.Map;
     import java.util.Map.Entry;
     import java.util.Set;

     import org.apache.commons.lang.StringUtils;
     import org.apache.http.Header;
     import org.apache.http.HttpEntity;
     import org.apache.http.HttpResponse;
     import org.apache.http.HttpStatus;
     import org.apache.http.NameValuePair;
     import org.apache.http.client.HttpClient;
     import org.apache.http.client.entity.UrlEncodedFormEntity;
     import org.apache.http.client.methods.HttpGet;
     import org.apache.http.client.methods.HttpPost;
     import org.apache.http.impl.client.DefaultHttpClient;
     import org.apache.http.message.BasicHeader;
     import org.apache.http.message.BasicNameValuePair;
     import org.apache.log4j.Logger;

     /**
     *
     * @author arganzheng
     * @date 2012-08-10
     */
     public class HttpClientWrapper {

          public static final Logger logger = Logger
                    .getLogger(HttpClientWrapper.class);

          private static final String DEFAULT_ENCODING = "utf-8";

          // 让connectionmanager管理httpclientconnection时是否关闭连接
          // 参见：http://fuyun.org/2009/09/connection-close-in-httpclient/
          private final HttpClient client = new DefaultHttpClient();

          public HttpClient getHttpClient() {
               return client;
          }

          /**
          * @param url
          *            请求的资源ＵＲＬ
          * @param header
          *            request请求时附带的头信息(header) 没有时传null
          * @param postData
          *            POST请求时form表单封装的数据 没有时传null
          * @param encoding
          *            response返回的信息编码格式 传null将采用utf-8编码
          * @return response返回的文本数据
          *
          * @note 不保留http连接，每次请求结束都会关闭链接。
          */
          public <K, V> String sendRequest(String url, Map<String, String> header,
                    Map<K, V> postData, String encoding) {
               if (StringUtils.isBlank(encoding)) {
                    encoding = DEFAULT_ENCODING;
               }

               String responseString = null;
               try {
                    // 头部请求信息
                    Header[] headers = getHeader(header);

                    if (postData == null) { // get方式
                         HttpGet getRequest = new HttpGet(url);
                         getRequest.setHeaders(headers);

                         HttpResponse response = this.getHttpClient()
                                   .execute(getRequest);
                         if (response.getStatusLine().getStatusCode() != HttpStatus.SC_OK) {
                              logger.error("sendRequest error for url: " + url);
                              return null;
                         }
                         responseString = getResponseStringFromResponse(response,
                                   encoding);
                    } else {// post方式
                         HttpPost post = new HttpPost(url);
                         post.setHeaders(headers);

                         List<NameValuePair> parameters = getPostParametersFromPostData(postData);
                         HttpEntity entity = new UrlEncodedFormEntity(parameters,
                                   encoding);
                         post.setEntity(entity);

                         HttpResponse response = this.getHttpClient().execute(post);

                         if (response.getStatusLine().getStatusCode() != HttpStatus.SC_OK) {
                              logger.error("sendRequest error for url: " + url);
                              return null;
                         }

                         HttpEntity responseEntity = response.getEntity();
                         responseString = getResponseStringFromResponse(response,
                                   encoding);
                    }
               } catch (Exception e) {
                    throw new RuntimeException("sendRequest error!", e);
               } finally {
                    // 关闭链接，释放资源
                    this.getHttpClient().getConnectionManager().shutdown();
               }
               return responseString;
          }

          private <K, V> List<NameValuePair> getPostParametersFromPostData(
                    Map<K, V> postData) {
               List<NameValuePair> parameters = new ArrayList<NameValuePair>();
               NameValuePair pair;
               Set<String> keySet = (Set<String>) postData.keySet();
               for (String key : keySet) {
                    Object obj = postData.get(key);
                    if (obj instanceof Arrays) {
                         String arr[] = (String[]) obj;
                         for (String value : arr) {
                              pair = new BasicNameValuePair(key, value);
                              parameters.add(pair);
                         }
                    } else if (obj instanceof String) {
                         pair = new BasicNameValuePair(key, (String) obj);
                         parameters.add(pair);
                    } else {
                         throw new RuntimeException("http get not support parameter");
                    }
               }
               return parameters;
          }

          private String getResponseStringFromResponse(HttpResponse response,
                    String encoding) {
               if (response == null) {
                    return null;
               }
               String responseString = null;
               HttpEntity responseEntity = response.getEntity();

               InputStream responseStream = null;
               BufferedReader rd = null;

               try {
                    responseStream = responseEntity.getContent();
                    rd = new BufferedReader(new InputStreamReader(responseStream,
                              encoding));
                    String tempLine = rd.readLine();
                    StringBuffer tempStr = new StringBuffer();
                    String crlf = System.getProperty("line.separator");
                    while (tempLine != null) {
                         tempStr.append(tempLine);
                         tempStr.append(crlf);
                         tempLine = rd.readLine();
                    }
                    responseString = tempStr.toString();
               } catch (Exception e) {
                    throw new RuntimeException(e);
               } finally {
                    try {
                         responseStream.close();
                    } catch (IOException e) {
                    }
               }
               return responseString;
          }

          private Header[] getHeader(Map<String, String> header) {
               Header[] headers = null;
               if (header != null) {
                    headers = new Header[header.size()];
                    int i = 0;
                    for (Entry<String, String> e : header.entrySet()) {
                         headers[i++] = new BasicHeader(e.getKey(), e.getValue());
                    }
               }
               return headers;
          }
     }

可以看到如下代码：

     HttpEntity entity = new UrlEncodedFormEntity(parameters, encoding);
     post.setEntity(entity);

深入到源码中可以其实是做了这么一件事情：

     public class UrlEncodedFormEntity extends StringEntity {
      
         /**
          * Constructs a new {@link UrlEncodedFormEntity} with the list
          * of parameters in the specified encoding.
          *
          * @param parameters list of name/value pairs
          * @param encoding encoding the name/value pairs be encoded with
          * @throws UnsupportedEncodingException if the encoding isn't supported
          */
         public UrlEncodedFormEntity (
             final List <? extends NameValuePair> parameters,
             final String encoding) throws UnsupportedEncodingException {
             super(URLEncodedUtils.format(parameters, encoding), encoding);
             setContentType(URLEncodedUtils.CONTENT_TYPE + HTTP.CHARSET_PARAM +
                     (encoding != null ? encoding : HTTP.DEFAULT_CONTENT_CHARSET));
         }
          ...
     }

它其实是做了这么两件事情：

1. 对请求参数进行encoding：`URLEncodedUtils.format(parameters, encoding)`
2. 设置请求编码的header：`Content-Type: application/x-www-form-urlencoded; charset=${encoding}`

这样就没有问题了。

