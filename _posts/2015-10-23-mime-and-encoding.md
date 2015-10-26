---
title: MIME和编码学习笔记
layout: post
---


Base64
------

所谓Base64，就是说从ASCII码中选出64个字符----大写字母A-Z、小写字母a-z、数字0-9、符号"+"、"/"（再加上作为填充字符的"="，实际上是65个字符），作为一个基本字符集。然后，其他所有符号都转换成这个字符集中的字符。

具体来说，转换方式可以分为四步：

1. 将每3个字节作为一组，一共是24个二进制位: 3x8=24
2. 将这24个二进制位分为4组，每个组有6个二进制位: 24/4=6
3. 在每组前面加两个00，扩展成32个二进制位，即3个字节: 4x(6+2)=32
4. 根据下表，得到扩展后的每个字节的对应符号，这就是Base64的编码值

**说明**

1. 因为转换后的每个字符的最高两位都是0，所以实际有效位数是6位，也就是2^6=64个字符就可以覆盖所有的编码。
2. 如果剩下的字符不足3个字节，则用0填充，输出字符使用'='，因此编码后输出的文本末尾可能会出现1或2个"="。
3. 因为Base64将3个字节转化成4个字节，因此Base64编码后的文本，会比原文本大出三分之一左右。

标准的Base64编码表如下：
		
		Value Encoding  Value Encoding  Value Encoding  Value Encoding
         0 A            17 R            34 i            51 z
         1 B            18 S            35 j            52 0
         2 C            19 T            36 k            53 1
         3 D            20 U            37 l            54 2
         4 E            21 V            38 m            55 3
         5 F            22 W            39 n            56 4
         6 G            23 X            40 o            57 5
         7 H            24 Y            41 p            58 6
         8 I            25 Z            42 q            59 7
         9 J            26 a            43 r            60 8
        10 K            27 b            44 s            61 9
        11 L            28 c            45 t            62 +
        12 M            29 d            46 u            63 /
        13 N            30 e            47 v
        14 O            31 f            48 w         (pad) =
        15 P            32 g            49 x
        16 Q            33 h            50 y

可以看到其实还是蛮有规律的，就是大写字母A-Z、小写字母a-z、数字0-9、符号"+"、"/"一共64个字符，编号（码值）从0到63，还有一个用于填充的'='。

### Padding

Base64是三个字节(Bytes)作为一组(24-bit block)的编码转换，如果字节数不是三的倍数，那么就会出最后一组只有一个或者两个字节的情况，那么讲会做这样的处理:

1. 一个字节的情况：将这一个字节的8个二进制位，按照上面的规则转成二组，最后一组除了前面加二个0以外，后面再加4个0。这样得到一个二位的Base64编码，再在末尾补上两个"="号。
比如，"M"这个字母是一个字节，二进制是01001101，可以转化为二组00010011、00010000，对应的Base64值分别为T、Q，再补上二个"="号，因此"M"的Base64编码就是TQ==。
2. 二个字节的情况：将这二个字节的一共16个二进制位，按照上面的规则，转成三组，最后一组除了前面加两个0以外，后面也要加两个0。这样得到一个三位的Base64编码，再在末尾补上一个"="号。
比如，"Ma"这个字符串是两个字节，二进制表示是01001101 01100001，可以转化成三组00010011、00010110、00000100以后，对应Base64值分别为T、W、E，再补上一个"="号，因此"Ma"的Base64编码就是TWE=。

**TIPS**

后缀的'='用于表示最后一组缺失的字节数。'=='表示最后一组只包含一个字节，'='表示最后一组包含两个字节。



### Base64的应用

标准的Base64并不适合直接放在URL里传输，根据网络标准[RFC1738](http://www.ietf.org/rfc/rfc1738.txt)的规定：only alphanumerics, the special characters "$-_.+!*'(),", and reserved characters used for their reserved purposes may be used unencoded within a URL.

	RFC 1738            Uniform Resource Locators (URL)        December 1994


	alpha          = lowalpha | hialpha
	digit          = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" |
	                 "8" | "9"
	safe           = "$" | "-" | "_" | "." | "+"
	extra          = "!" | "*" | "'" | "(" | ")" | ","
	national       = "{" | "}" | "|" | "\" | "^" | "~" | "[" | "]" | "`"
	punctuation    = "<" | ">" | "#" | "%" | <">


	reserved       = ";" | "/" | "?" | ":" | "@" | "&" | "="
	hex            = digit | "A" | "B" | "C" | "D" | "E" | "F" |
	                 "a" | "b" | "c" | "d" | "e" | "f"
	escape         = "%" hex hex

	unreserved     = alpha | digit | safe | extra
	uchar          = unreserved | escape
	xchar          = unreserved | reserved | escape
	digits         = 1*digit


**NOTES**

因为"%"用于URL转义，所以它本身被URL编码为"%25"。

下面是常见的URL保留字以及相应的编码：

	!	#	$	&	'	(	)	*	+	,	/	:	;	=	?	@	[	]
	%21	%23	%24	%26	%27	%28	%29	%2A	%2B	%2C	%2F	%3A	%3B	%3D	%3F	%40	%5B	%5D

回到正题，因为'/'和'='是URI的保留字符，对于URL有特殊的意义。所以标准的做法就是对Base64编码过的URL进行 [URL encoding](https://en.wikipedia.org/wiki/Percent-encoding) ——（'+' => '%2B', '/' => '%2F'，'=' => '%3D'），但是这样子会导致URL无意义的变长，而且多了一个URL编解码步骤。为了避免这个问题，出现了一种用于URL的改进Base64编码变种，它其实就是简单的把标准Base64中的'+'和'/'分别改成了'-'和'_'。对于填充字符'='，有些变种是把它直接去掉，有些则是把它替换成'.'。



参考文章
-------

1. [MIME笔记](http://www.ruanyifeng.com/blog/2008/06/mime.html)
2. [Base64笔记](http://www.ruanyifeng.com/blog/2008/06/base64.html)
3. [Base64](https://en.wikipedia.org/wiki/Base64)
4. [Percent encoding](https://en.wikipedia.org/wiki/Percent-encoding)
5. [关于URL编码](http://www.ruanyifeng.com/blog/2010/02/url_encoding.html)
6. [字符编码笔记：ASCII，Unicode和UTF-8](http://www.ruanyifeng.com/blog/2007/10/ascii_unicode_and_utf-8.html)
