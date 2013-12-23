---
title: 多次编解码导致的的奇怪乱码问题
layout: post
---

今天遇到一个诡异的编码问题：我们这边一个客户端A调用某个网关B，网关调用某个服务C，服务C又调用服务D。结果发现收到的数据有中文乱码。但是这个乱码和用错编码的乱码很不同：大部分都是正确的，就是几个字符是乱码（基本都是'?'）。在A直接调用D，编码没有问题。说明A和D都是同一种编码（确定是utf-8编码）。那么就是中间的B和C编码出问题了。于是确定了一下B和C的编码，果然B是utf-8，C是GBK。C的编码与大家都不一致，所以会出现这个问题。

但是这里有个诡异的问题，就是在一个调用串中，即使每个调用方的编解码都是约定好的，只要这条调用串的编码不一致（而且是多字节编码方式的），就会出现乱码问题。看下面这个例子：

    public class MultipleEncodingTest{

        public static void main(String[] args) throws UnsupportedEncodingException {
            String str = "广东5天内确诊第4例人感染H7N9病例";

            String encoding = "UTF-8";
            String intermediateEncoding = "GBK";

            System.out.println(str);

            // 发送前编码
            byte[] bs = str.getBytes(encoding);

            // 中间服务解码再編码
            str = new String(bs, intermediateEncoding);
            bs = str.getBytes(intermediateEncoding);

            // 接收端解码
            str = new String(bs, encoding);

            System.out.println(str);

        }
    }    


打印结果：

    广东5天内确诊第4例人感染H7N9病例
    广东5天内确诊�?例人感染H7N9病例


如果把中间服务的编解码改成单字节的`iso8859`就不会有问题了：

    String intermediateEncoding = "iso8859-1";


    广东5天内确诊第4例人感染H7N9病例
    广东5天内确诊第4例人感染H7N9病例


**结论** 多次编解码，一定要保证整个链条中的每个节点都是同样的编码，或者中间节点采用单字节的编解码方式（如iso8859-1）。

