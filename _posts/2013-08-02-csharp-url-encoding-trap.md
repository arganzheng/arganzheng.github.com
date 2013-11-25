---
layout: post
title: C# Url Encoding的一些问题
---


最近API的C#版本SDK又出问题了，总是报sign计算错误。Sign计算问题一般是源串生成（参数排序、URL Encoding之类的）不一致，另一个就是签名算法（一般是HMAC或者MD5）不一致。首先看一下原串生成是不是有问题。

用Java和C#对同样的请求参数进行源串生成，发现果然不一样。对比了一下，发现是由于左右括号引起的。

    public static void main(String[] args) throws UnsupportedEncodingException {
        String itemDesc = "样品材质：材质棉+聚脂纤维        颜色描述：蓝色  绿色  卡其(样品为蓝色)";
        String encodingString = URLEncoder.encode(itemDesc, "UTF-8");
        System.out.println(encodingString);
    }

输出结果如下：

    %E6%A0%B7%E5%93%81%E6%9D%90%E8%B4%A8%EF%BC%9A%E6%9D%90%E8%B4%A8%E6%A3%89%2B%E8%81%9A%E8%84%82%E7%BA%A4%E7%BB%B4++++++++%E9%A2%9C%E8%89%B2%E6%8F%8F%E8%BF%B0%EF%BC%9A%E8%93%9D%E8%89%B2++%E7%BB%BF%E8%89%B2++%E5%8D%A1%E5%85%B6%28%E6%A0%B7%E5%93%81%E4%B8%BA%E8%93%9D%E8%89%B2%29

但是在C#中：

    namespace me.arganzheng.study
    {
        class Program
        {
            static void Main(string[] args)
            {
                String itemDesc = "样品材质：材质棉+聚脂纤维        颜色描述：蓝色  绿色  卡其(样品为蓝色)";
                String encodingString = HttpUtility.UrlEncode(itemDesc, Encoding.UTF8);
                Console.WriteLine(encodingString);
            }
        }
    }

输出结果是：

    %e6%a0%b7%e5%93%81%e6%9d%90%e8%b4%a8%ef%bc%9a%e6%9d%90%e8%b4%a8%e6%a3%89%2b%e8%81%9a%e8%84%82%e7%ba%a4%e7%bb%b4++++++++%e9%a2%9c%e8%89%b2%e6%8f%8f%e8%bf%b0%ef%bc%9a%e8%93%9d%e8%89%b2++%e7%bb%bf%e8%89%b2++%e5%8d%a1%e5%85%b6(%e6%a0%b7%e5%93%81%e4%b8%ba%e8%93%9d%e8%89%b2)
    
很明显的区别在于左右括号。简单replace一下就可以了：
     
    public string UrlEncode(string str, Encoding e)
    {
        if (str == null)
        {
            return null;
        }

        return HttpUtility.UrlEncode(str, e).Replace("(", "%28").Replace(")", "%29");
    }

再测试，发现还是不一样，仔细对比查看才发现C# encoding字符都是小写的，而java的是大写的。如`/`在Java是转移为`%2F`，在C#是`%2f`。所以还要做一下大小写转换。另外，查看了一下URL转换规范，`+`和`*`也要转移的，这点Java和C#都没有转，所以一并处理。最终encoding方法如下：

    /**
     * C#的URL encoding有三个问题：
     * 1.左右括号没有转移（Java的URLEncoder.encode有）
     * 2.转移符合都是小写的，而Java是大写的
     * 3.+、*号 Java/C#都没有编码
     */
    public static Regex REG_URL_ENCODING = new Regex(@"%[a-f0-9]{2}");
    public string UrlEncode(string str, Encoding e)
    {
        if (str == null)
        {
            return null;
        }

        String stringToEncode = HttpUtility.UrlEncode(str, e).Replace("+", "%20").Replace("*", "%2A").Replace("(", "%28").Replace(")", "%29");
        return REG_URL_ENCODING.Replace(stringToEncode, m => m.Value.ToUpperInvariant());
    }

再测试一下，这下子完全一样了。

## 参考资料

1. [HTML URL Encoding Reference](http://www.w3schools.com/tags/ref_urlencode.asp)