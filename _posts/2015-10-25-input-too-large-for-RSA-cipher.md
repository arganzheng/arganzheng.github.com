---
title: input too large for RSA cipher
layout: post
---


自从我们的某些接口用上加密解密后，总是会收到如下错误报警：

	2015-10-22 16:41:23,453 [WARN] [catalina-exec-178] me.arganzheng.study.nantianmen.utils.ParamsHelper (ParamsHelper.java:132): Can not get decrypted params. encryptedParams=3ekss73ymx96en5vrrai58ux8ibifleidyayo2ewpry09sk0/v42cgs8cvquevbqvb/ecmnfyluxgw8ybu1sk2a
		w9edu1titloweefeehbcant3xtiqytckh2bdsadz4jvmjyza1rmlt9ec1byd68ojfmgx3zqteerpegc6oi8kiw5y810nem3t6ugfn0ck8a6o2sayuok9y1hn3qvaju/ltsklldfe/foep7mht/rotqkrluz1vw2ijyjl5aml7jukzpg8eb7c7thh1/o3r6ylxrgzlzeehoolpj3ohksbzzytzmmqwmonworvs7edbgsr07qbxu0t+z0ukoezyb5q==
		org.bouncycastle.crypto.DataLengthException: input too large for RSA cipher.
		        at org.bouncycastle.crypto.engines.RSACoreEngine.convertInput(Unknown Source)
		        at org.bouncycastle.crypto.engines.RSABlindedEngine.processBlock(Unknown Source)
		        at org.bouncycastle.jcajce.provider.asymmetric.rsa.CipherSpi.engineDoFinal(Unknown Source)
		        at javax.crypto.Cipher.doFinal(Cipher.java:2121)
		        at me.arganzheng.study.toolkit.blade.cipher.DecryptionUtils.decryptByRSAInPieces(DecryptionUtils.java:90)
		        at me.arganzheng.study.toolkit.blade.cipher.DecryptionUtils.decryptString(DecryptionUtils.java:275)
		        at me.arganzheng.study.nantianmen.utils.ParamsHelper.getDecryptedParams(ParamsHelper.java:130)
		        at me.arganzheng.study.nantianmen.convertor.SkinConvertor.convertSkinParams(SkinConvertor.java:61)
		        at me.arganzheng.study.nantianmen.web.SkinController.channel(SkinController.java:185)
		        at sun.reflect.GeneratedMethodAccessor973.invoke(Unknown Source)
		        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
		        at java.lang.reflect.Method.invoke(Method.java:483)
		        at org.springframework.web.method.support.InvocableHandlerMethod.invoke(InvocableHandlerMethod.java:215)


而且关键是bouncycastle的源码没有line number信息，无法跳进去debug，导致查起问题困难很多。直接谷歌没有发现明了的答案，只能先从原理入手了。


RSA是什么
--------

关于RSA，阮一峰在其博文《RSA算法原理》中做了通俗易懂的介绍。简单来说就是不对称加密算法的基本原理就是单向函数。而其中一种单向函数是整数分解的问题：即给定两个大素数p 、q，计算他们的乘积n很容易；但是反过来知道n的情况下，想要分解出p、q却非常困难。在RSA算法中，n是公钥的一部分，是公开的，而p、q是保密的，否则根据p、q就可以很容易推导出私钥d。

总结来说RSA用到如下六个数字：`p, q, n, φ(n), e, d`。

* 其中p和q是选定的两个不相等的质数p和q。实际应用中，这两个质数越大，就越难破解。（因为相乘得到的n会很大 => n的长度越大 => 密钥长度越大）
* n是p和q的乘积。需要注意的是n的二进制位数长度就是密钥长度。实际应用中，RSA密钥一般是1024位，重要场合则为2048位。
* φ(n)是n的欧拉函数，根据欧拉函数公式：φ(n) = (p-1)(q-1)可以轻易算得。
* e是随机选择一个整数e，条件是 1 < e < φ(n)，且e与φ(n)互质。实际应用中，常常选择65537。
* d是e对于φ(n)的模反元素。所谓 [模反元素](http://zh.wikipedia.org/wiki/%E6%A8%A1%E5%8F%8D%E5%85%83%E7%B4%A0) 就是指有一个整数d，可以使得ed被φ(n)除的余数为1。即满足`ed - 1 = kφ(n)`，在已知e和φ(n)的情况下，这个方程可以用 [扩展欧几里得算法](http://zh.wikipedia.org/wiki/%E6%89%A9%E5%B1%95%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95) 求解，取其中的一组整数解作为结果即可。


公钥(n, e)，私钥(n, d)。可以看到公私钥其实是一对数值来的。但是我们常见的是一个长长的字符串，这是因为在实际应用中，公钥和私钥的数据都采用ASN.1格式表示。

可以看到其实其他几个数值都是根据p和q计算出来的。所以p和q如果泄漏就意味着公私密钥都泄漏。但是p和q的乘积n确实公开的，这就是前面说过的RSA可靠性的基础：即给定两个大素数p 、q，计算他们的乘积n很容易；但是反过来知道n的情况下，想要分解出p、q却非常困难。目前，除了暴力破解，还没有发现别的有效方法。目前人类已经分解的最大整数（232个十进制位，768个二进制位）。比它更大的因数分解，还没有被报道过，因此目前被破解的最长RSA密钥就是768位。所以现在一般都会采用1024位的RSA密钥。也就是说需要找到p和q使得他们的乘积n在2的1024次方左右。


从上面分析可以知道，RSA的加密和解密都是在整数环内完成的。假设要加密的明文是m，那么要把m看成是一个位数很多的整数，其整数值在集合内，所以m表示的二进制必须小于n。

但是在实际应用中，通常需要加密的明文都是任意长度的文件或者报文。这就带来两个问题：

第一个问题是如何将报文转换为整数。这个好办，因为在计算机的世界里任何编码其实都是二进制，而二进制本身就是数值。

第二个问题复杂一些：公钥(n,e) 只能加密大于0小于n的整数m，那么如果要加密大于n的整数，该怎么办呢？

解决方案是把长信息分割成若干段短消息，每段分别加密。常用的分组模式有ECB（电子密码本模式）、CBC（密码分组链接模式）、CFB（密码反馈模式）、OFB（输出反馈模式）、CTR（计数器模式）等。

不过这又会带来两个问题：

1. 分组的大小
2. 最后一个分组长度问题，因为明文长度不可能每次都刚好是分组的整数倍。

第一个问题比较好解决。因为0<m<n，所以分组的大小上限其实就是n的位数。假设密钥长度是1024，那么分组的大小就可以是1024-1，即127个字节：


 	/**
     * Return the maximum size for an input block to this engine.
     * For RSA this is always one byte less than the key size on
     * encryption, and the same length as the key size on decryption.
     *
     * @return maximum size for an input block.
     */
    public int getInputBlockSize()
    {
        int     bitSize = key.getModulus().bitLength();

        if (forEncryption)
        {
            return (bitSize + 7) / 8 - 1;
        }
        else
        {
            return (bitSize + 7) / 8;
        }
    }


第二个问题也不难解决，事实上我们在前面介绍Base64的也是遇到一样的情况，所有的分组处理（编码、加密等）的算法都会有这个问题。解决方案也非常的统一明了，就是padding(填充)。
不过因为RSA实际上是操作正整数，所以padding的方式是在分组的前面填充标志位和多个0，直到明文刚好为整数倍的分组大小。

但是在实际应用中为了增加安全性，一般还会对每个分组进行填充(padding)混淆。填充方案非常重要，实现不好，RSA实现就不安全。它在加密明文前将一个随机结构嵌入到明文中。有点类似对称密钥算法中的CBC等模式。padding的一个副作用就是每个分组的大小变小了。也就是说每个分组的长度(block length)不仅仅跟密钥长度(key length)相关，还跟所使用的填充模式相关。

	
常见的填充模式有：

1、PKCS1PADDING: 这是最常用的模式，也是JDK默认的填充模式。

要求:
输入: 必须 比 RSA 钥模长(modulus) 短至少11个字节, 也就是　RSA_size(rsa) – 11
如果输入的明文过长，必须切割，然后填充。
输出: 和modulus一样长
根据这个要求，对于1024bit的密钥，block length = 1024/8 – 11 = 117 字节

2、RSA_PKCS1_OAEP_PADDING

RSA_size(rsa) – 41

3、RSA_NO_PADDING　　不填充


**NOTES**

1、JDK7的RAS Cipher实现(`com.sun.crypto.provider.RSACipher extends javax.crypto.CipherSpi`)默认的RSA的模式(Mode)和填充方式(Padding)是：`RSA/ECB/PKCS1Padding`。而BouncyCastle的实现(`org.bouncycastle.jcajce.provider.asymmetric.rsa.CipherSpi`)则是`RSA/NONE/NoPadding`。

这就是为什么Java 默认的 RSA 加密实现不允许明文长度超过密钥长度减去11(单位是字节，也就是 byte)，否则会抛诸如`javax.crypto.IllegalBlockSizeException: Data must not be longer than 117 bytes` 的异常，而BC的明文可以是密钥长度，因为BC默认不padding。

不过不管是JDK或者BC的RSA加密，都不能超过密钥长度，一般的做法就是自己分组加密。事实上，你如果看BC的代码，会发现其实他对于加密的blockSize是 密钥长度-1（见上面的getInputBlockSize方法），因为用尽密钥空间有可能会出现m>n的情况。


2、注意，android平台上使用`Cipher.getInstance("RSA")`使用的其实是`org.bouncycastle.jce.provider.BouncyCastleProvider`，而不是JDK自带的SunJCE Provider。为了避免出现不同实现加解密的不同，最好就是指定providerName或者指定`Algorithm/Mode/Padding`。

另外需要注意的是，SunJCE Provider并不支持`RSA/NONE/NoPadding`模式，所以使用`Cipher.getInstance("RSA/NONE/NoPadding")`也会使用BC Provider而不是SunJCE Provider。



**TIPS** 

因为不对称加密的计算相对比较耗时，所以一般不会直接用来对长文笔进行加密解密，而是用于一开始的对称密钥交换过程加密。一旦成功交换了对称加密密钥，后续的通讯就都是走对称加密了。


原理性的东西介绍了那么多。那么上面的`input too large for RSA cipher`原因是什么呢？看一下出错的代码：


 	public BigInteger convertInput(
        byte[]  in,
        int     inOff,
        int     inLen)
    {
        if (inLen > (getInputBlockSize() + 1))
        {
            throw new DataLengthException("input too large for RSA cipher.");
        }
        else if (inLen == (getInputBlockSize() + 1) && !forEncryption)
        {
            throw new DataLengthException("input too large for RSA cipher.");
        }

        byte[]  block;

        if (inOff != 0 || inLen != in.length)
        {
            block = new byte[inLen];

            System.arraycopy(in, inOff, block, 0, inLen);
        }
        else
        {
            block = in;
        }

        BigInteger res = new BigInteger(1, block);
        if (res.compareTo(key.getModulus()) >= 0)
        {
            throw new DataLengthException("input too large for RSA cipher.");
        }

        return res;
    }

由于没有行号，所以不知道是那一段抛出异常了。根据分析和验证，知道出错的是最后一个判断：

	BigInteger res = new BigInteger(1, block);
    if (res.compareTo(key.getModulus()) >= 0)
    {
        throw new DataLengthException("input too large for RSA cipher.");
    }

这看起来像是m>n的情况。但是事实上这是发生在解密的时候，也就是说此时input不是明文m，而是密文c！难道密文c也必须小于m?
根据公式：

	encrypt(m=>c): c = me mod n
	decrypt(c=>m): m = cd mod n

可以看出，不管是明文m还是密文c，都是n的余数，所以m和c都必须小n。

对于明文m我们是很好控制的，只要保证加密的时候分组大小不要大于密文大小就可以了。但是对于密文c，就比较难办了。因为c是根据公式计算出来的，理论上是不可能出现大于n的情况，但是实际上由于填充补位的原因，不管m是几位，加密后的c总是会被填充为密钥长度（如128字节），既然都是128字节，所以就有可能会导致某些情况下c会大于n。这能解释为什么有时候可行，有时候报错。但是笔者觉得如果这样就太挫了，期待高手解答。


参考文档
-------

1. [RSA算法原理（一）](http://www.ruanyifeng.com/blog/2013/06/rsa_algorithm_part_one.html)
2. [RSA算法原理（二）](http://www.ruanyifeng.com/blog/2013/07/rsa_algorithm_part_two.html)
3. [Number Theory – A Primer](http://jeremykun.com/2011/07/30/number-theory-a-primer/)
4. [Encryption & RSA](http://jeremykun.com/2011/07/29/encryption-rsa/)
5. [RSA (cryptosystem)](https://en.wikipedia.org/wiki/RSA_(cryptosystem))
6. [[Just a door] java web对RSA的使用](http://www.justabug.net/door-1-rsa/)
7. [Problem with BouncyCastle RSA](https://community.oracle.com/thread/1557359?start=0&tstart=0)
8. [crypto incompatibilities between pycrypto and bouncycastle](https://winswitch.org/trac/ticket/162)
9. [Too much data for RSA block fail. What is PKCS#7?](http://stackoverflow.com/questions/2579103/too-much-data-for-rsa-block-fail-what-is-pkcs7)
