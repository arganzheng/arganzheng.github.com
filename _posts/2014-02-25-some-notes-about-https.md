---
title: HTTPS原理
layout: post
---

HTTPS通讯流程
-------------

![HTTPS通讯时序图](/media/images/https.png)

1. 客户端发出安全会话请求（会同时将自己支持的一套加密规则发送给服务器）
2. 服务器从中选出一组加密算法与HASH算法，并将自己的身份信息以证书的形式发回给客户端。证书里面包含了服务器地址，加密公钥，以及证书的颁发机构等信息。 
3. 客户端获得服务器CA证书之后浏览器要做以下工作： 
	1. 用已知的CA列表来验证证书的合法性（颁发证书的机构是否合法，证书中包含的网站地址是否与正在访问的地址一致等），如果证书受信任，则浏览器栏里面会显示一个小锁头，否则会给出证书不受信的提示，让用户选择自担风险。 
	2. 如果证书受信任，或者是用户接受了不受信的证书，浏览器会生成一串随机对称密钥，并用证书中提供的公钥加密。 
	3. 使用约定好的HASH算法计算握手消息，并使用生成的随机对称密钥对消息进行加密，最后将之前生成的所有信息发送给服务器。 
4. 服务器接收客户端发来的数据之后要做以下的操作： 
	1. 使用自己的私钥将信息解密取出密码，使用密码解密浏览器发来的握手消息，并验证HASH是否与浏览器发来的一致。 
	2. 使用对称密钥加密一段握手消息，发送给客户端。 
5. 客户端解密并计算握手消息的HASH，如果与服务端发来的HASH一致，此时握手过程结束，之后所有的通信数据将由之前客户端生成的随机对称密钥并利用对称加密算法进行加密。


说明
----

#### SSL和TLS

* SSL1, SSL2
* SSL3 = TLS1 双向证书交换


#### 证书的合法性保证

证书的作用：访问https://blog.arganzheng.me，确认它就是blog.arganzheng.me。
证书的合法性是通过证书的颁发机构权威性保证的。这引出了一个问题：怎么确保是合法的颁发机构呢？

1. 系统有配置所谓的权威机构，就是CA root，这是安装系统时就配置在电脑上。（所以不要随便安装盗版操作系统，也不要随便安装不受信任的证书[/呲牙]）
2. 权威机构有级联性，即，如果证书是被机构B认证的，而机构B的证书被CA root认证，则证书也是受信的。这就是所谓的证书链（certification chain）


#### 证书如何作废？

* CRL (certification revocation list)
* CDP (certification distribution point)
* OCSP (online certificate status protocol)

#### 证书的常见格式

* PKCS标准体系
* X509为最常见的实现
* 证书最常见的后者名：cer, p12, crt, p7b

#### HTTPS一般使用的加密与HASH算法

* 对称加密算法：AES，RC4，3DES。速度快，但需要双方交换或者保存密钥，风险大。
* 非对称加密算法：RSA，DSA/DSS。不需要双方交换或者保存密钥，但是速度奇慢。
* HASH算法：MD5，SHA1，SHA256。

一般来说服务器使用RSA非对称加密算法来生成公钥和私钥（CA证书）。但是RSA性能是非常低的，原因在于寻找大素数、大数计算、数据分割需要耗费很多的CPU周期，所以一般的HTTPS连接只在第一次握手时使用RSA非对称加密，通过握手交换对称加密密钥，在之后的通信走对称加密。


#### 证书的组成

* 版本: V1
* 序列号
* 签名算法: ad2RSA
* 签名哈希算法：ad2
* 颁发者：Class 3 Public Primary ...
* 有效期
* 使用者：Class 3 Public Primary ...
* 公钥：RSA (2048 Bits)
* 指纹算法：sha1
* 指纹

#### 用openssl制作证书

1. 生成CA密钥（这里用RSA加密算法）

	openssl genrsa -des3 -out server.key 1024

2. 用密钥对创建证书请求(CSR)文件
	
	openssl req -new -key server.key -out server.csr

	期间要求输入：国家代码、地区名称、城市名称、组织名称、组织单位名称、常用名称、电子邮件、密码、名称。

3. 查看生成的密钥和请求文件

	openssl rsa -noout -text -in server.key
	openssl req -noout -text -in server.csr

4. 备份私钥并提交证书请求，得到证书


#### HTTPS的安全性讨论


HTTPS其实是两个过程来的

1. 客户端通过CA证书认证服务器是否合法
2. 双方交换对称密钥的过程（由客户端发起，具体是前面提到的5个步骤）

在每一个步骤，都有可能被伪造和欺骗。


第一个过程可以伪造证书来欺骗，所以强烈建议只安装信任机构颁发的证书（浏览器不会提醒风险的那种）。比如12306要求安装它自己制造的证书，如果安装了，12306就能伪装google，不装12306，他就伪装不了。所以是有风险的。

第二个过程如果服务端私钥泄漏了，那么也交换私钥的过程虽然是用服务器公钥加密，也相当于明文传输了，这时候证书就可能被伪造。随后的通信过程当然也无安全可言。此时，CA需要更换私钥和公钥，并申明所有下级CA和使用者的证书过期（revocation list）。随后，把所有过期证书重新用新的私钥签名。

在第二个过程中，很容易出错的一个步骤是步骤三：

>**3. 客户端获得服务器CA证书之后浏览器要做以下工作：**
>	 1. 用已知的CA列表来验证证书的合法性（颁发证书的机构是否合法，证书中包含的网站地址是否与正在访问的地址一致等），如果证书受信任，则浏览器栏里面会显示一个小锁头，否则会给出证书不受信的提示，让用户选择自担风险。 
> 	 2. 如果证书受信任，或者是用户接受了不受信的证书，浏览器会生成一串随机对称密钥，并用证书中提供的公钥加密。 
>	 3. 使用约定好的HASH算法计算握手消息，并使用生成的随机对称密钥对消息进行加密，最后将之前生成的所有信息发送给服务器。


这个公司内网有个分享，阿甘这里觉得没有什么需要保密的，所以就公开出来了。希望所有开发同学都能注意一下，特别是手机APP开发。

--- 

*注* 以下是引用内容，稍微有修改。

在google的官方文档中，详细给出了若干种Android平台中使用https的方法。开发小伙伴在使用了这些代码开发测试自己产品的https功能时，会发现发生很多种类型的https异常，相信不少有经验的同学也遇到过类似的问题。简单来说，根本原因是google的API会检查https证书进行合法性。而开发或者测试环境的https证书，基本上都无法通过合法性检查。
API的检查内容包括以下4方面的内容：

1. 签名CA是否合法；
2. 域名是否匹配；
3. 是不是自签名证书；
4. 证书是否过期。

一旦发现任何异常，则会终止请求并抛出相应的异常。那小伙伴们在产品开发或者测试时怎么办呢？绝大多数产品都采用了覆盖google默认的证书检查机制（X509TrustManager）的方式来解决这个问题。一个很典型的解决方案如下所示 [EasyX509TrustManager.java](https://github.com/Unidata/thredds/blob/master/cdm/src/main/java/ucar/nc2/util/net/EasyX509TrustManager.java)：


	/*
	 * ====================================================================
	 *
	 *  Copyright 2002-2004 The Apache Software Foundation
	 *
	 *  Licensed under the Apache License, Version 2.0 (the "License");
	 *  you may not use this file except in compliance with the License.
	 *  You may obtain a copy of the License at
	 *
	 *      http://www.apache.org/licenses/LICENSE-2.0
	 *
	 *  Unless required by applicable law or agreed to in writing, software
	 *  distributed under the License is distributed on an "AS IS" BASIS,
	 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	 *  See the License for the specific language governing permissions and
	 *  limitations under the License.
	 * ====================================================================
	 *
	 * This software consists of voluntary contributions made by many
	 * individuals on behalf of the Apache Software Foundation.  For more
	 * information on the Apache Software Foundation, please see
	 * <http://www.apache.org/>.
	 *
	 */

	package ucar.nc2.util.net;

	import thredds.logs.ServletLogParser;
	import ucar.nc2.util.rc.RC;

	import java.security.KeyStore;
	import java.security.KeyStoreException;
	import java.security.NoSuchAlgorithmException;
	import java.security.cert.CertificateException;
	import java.security.cert.X509Certificate;

	import javax.net.ssl.TrustManagerFactory;
	import javax.net.ssl.TrustManager;
	import javax.net.ssl.X509TrustManager;

	/**
	 * <p/>
	 * EasyX509TrustManager unlike default {@link X509TrustManager} accepts
	 * self-signed certificates.
	 * </p>
	 * <p/>
	 * This trust manager SHOULD NOT be used for productive systems
	 * due to security reasons, unless it is a concious decision and
	 * you are perfectly aware of security implications of accepting
	 * self-signed certificates
	 * </p>
	 *
	 * @author <a href="mailto:adrian.sutton@ephox.com">Adrian Sutton</a>
	 * @author <a href="mailto:oleg@ural.ru">Oleg Kalnichevski</a>
	 *         <p/>
	 *         <p/>
	 *         DISCLAIMER: AbstractHttpClient developers DO NOT actively support this component.
	 *         The component is provided as a reference material, which may be inappropriate
	 *         for use without additional customization.
	 *         </p>
	 */

	public class EasyX509TrustManager implements X509TrustManager {
	  private X509TrustManager standardTrustManager = null;

	  static public org.slf4j.Logger logger = HTTPSession.log;

	  /**
	   * Constructor for EasyX509TrustManager.
	   */
	  public EasyX509TrustManager(KeyStore keystore) throws NoSuchAlgorithmException, KeyStoreException {
	    super();
	    TrustManagerFactory factory = TrustManagerFactory.getInstance("SunX509");
	    factory.init(keystore);
	    TrustManager[] trustmanagers = factory.getTrustManagers();
	    if (trustmanagers.length == 0) {
	      throw new NoSuchAlgorithmException("SunX509 trust manager not supported");
	    }
	    this.standardTrustManager = (X509TrustManager) trustmanagers[0];
	  }

	  /**
	   * see com.sun.net.ssl.X509TrustManager#getAcceptedIssuers()
	   */
	  public X509Certificate[] getAcceptedIssuers() {
	    return this.standardTrustManager.getAcceptedIssuers();
	  }

	  /**
	   * see com.sun.net.ssl.X509TrustManager#isClientTrusted(X509Certificate[])
	   */
	  public void checkClientTrusted(X509Certificate[] certificates, String authType)
	        throws CertificateException
	  {
	    this.standardTrustManager.checkClientTrusted(certificates, authType);
	  }

	  /**
	   * see com.sun.net.ssl.X509TrustManager#isServerTrusted(X509Certificate[])
	   */
	  public void checkServerTrusted(X509Certificate[] certificates, String authType)
	        throws CertificateException
	  {
	    if ((certificates != null) && logger.isDebugEnabled()) {
	      logger.debug("Server certificate chain:");
	      for (int i = 0; i < certificates.length; i++) {
	        logger.debug("X509Certificate[" + i + "]=" + certificates[i]);
	      }
	    }

	    // The certificate checking rules are as follows:
	    // 1. If !RC.getVerifyServer()
	    //    then just return (indicating success)
	    // 2. If certificates.length > 1 || !RC.getAllowSelfSigned() then 
	    //    call standardTrustManager.checkServerTrusted() to
	    //    see if this is a valid certificate chain.
	    // 3. Otherwise, see if this looks like a self signed certificate.

	    if(RC.getVerifyServer()) {
	        if(RC.getAllowSelfSigned() && certificates != null && certificates.length == 1) {
	            X509Certificate certificate = certificates[0];
	                certificate.checkValidity(); // check that current date is within the certficates valid dates
	                // See if this looks like a self-signed cert
	                if(!certificate.getIssuerDN().equals(certificate.getSubjectDN())) {
	                    // apparently not self-signed so check certificate chain
	                    standardTrustManager.checkServerTrusted(certificates,authType);
	            }
	        } else // Do a complete certificates check
	            standardTrustManager.checkServerTrusted(certificates,authType);
	    }
	    return;
	  }
	  
	}


相信许多同学看到这段代码，已经发现问题在哪里了：覆盖默认的证书检查机制后，检查证书是否合法的责任，就落到了我们自己的代码上。但绝大多数app在选择覆盖了默认安全机制后，却没有对证书进行应有的安全性检查，直接接受了所有异常的https证书，不提醒用户存在安全风险，也不终止这次危险的连接。实际上，现在所有的网页浏览器，都会对这类https异常进行处理并提醒用户存在安全风险。


想要利用这个漏洞进行攻击，我们需要能够进行流量劫持，去截获并修改https握手时数据包：将握手时的服务器下发的证书，替换成我们伪造的假证书。随后，全部的https数据都在我们的监控之下，如果需要，甚至可以随意篡改数据包的内容。

可以采用下面的方式进行攻击：

1. 伪造公众wifi进行劫持：具体方式这里不展开
2. 城域网、DNS等其他形式的流量劫持：具体方式这里不展开


参考文章
--------

1. [HTTPS工作原理和TCP握手机制](http://www.cnblogs.com/ttltry-air/archive/2012/08/20/2647898.html)
2. [图解HTTPS](http://kb.cnblogs.com/page/155287/)
3. [浏览器如何验证SSL证书？](http://www.aitrust.com/browser-how-to-verify-ssl/)
4. [HTTPS的七个误解（译文）](http://www.ruanyifeng.com/blog/2011/02/seven_myths_about_https.html)
5. [新型攻击30秒内可从加密通信中提取机密信息](http://www.oschina.net/translate/gone-in-30-seconds-new-attack-plucks-secrets-from-https-protected-pages)
6. [Security with HTTPS and SSL](http://developer.android.com/training/articles/security-ssl.html)
7. [Android软件安全开发实践](http://www.programmer.com.cn/15036/)