---
title: Java中如何把InputStream转化为String
layout: post
---


### 法一：使用InputStreamReader(with charset)作为source，StringWriter或者StringBuilder作为destination。


	import java.io.BufferedReader;
	import java.io.FileInputStream;
	import java.io.IOException;
	import java.io.InputStream;
	import java.io.InputStreamReader;
	import java.io.StringWriter;

	public class InputStreamToStringTest {

	    public static final String CHARSET = "UTF-8";

	    public static void main(String args[]) throws IOException {
	        InputStream inputStream = new FileInputStream("D:/test.txt");

	        String string = getStreamAsString(inputStream, CHARSET);

	        System.out.println(string);
	    }

	    // InputStream => InputStreamReader(with charset) => String
	    private static String getStreamAsString(InputStream stream, String charset) throws IOException {
	        try {
	            BufferedReader reader = new BufferedReader(new InputStreamReader(stream, charset));
	            StringWriter writer = new StringWriter();

	            char[] chars = new char[256];
	            int count = 0;
	            while ((count = reader.read(chars)) > 0) {
	                writer.write(chars, 0, count);
	            }

	            return writer.toString();
	        } finally {
	            if (stream != null) {
	                stream.close();
	            }
	        }
	    }
	}


**说明** 1. 这里BufferdReader并不是必须的; 2. 也可以使用StringBuilder作为destination。

    private static String getStreamAsString(InputStream stream, String charset) throws IOException {
        try {
            Reader reader = new InputStreamReader(stream, charset);
            StringBuilder sb = new StringBuilder();

            char[] chars = new char[256];
            while (reader.read(chars) > 0) {
                sb.append(chars);
            }

            return sb.toString();
        } finally {
            if (stream != null) {
                stream.close();
            }
        }
    }


### 法二：使用Apache IOUtils工具类


	import java.io.FileInputStream;
	import java.io.IOException;
	import java.io.InputStream;

	import org.apache.commons.io.IOUtils;

	public class InputStreamToStringTest {

	    public static final String CHARSET = "UTF-8";

	    public static void main(String args[]) throws IOException {
	        InputStream inputStream = new FileInputStream("D:/test.txt");

	        String string = getStreamAsString(inputStream, CHARSET);

	        System.out.println(string);
	    }

	    // InputStream => InputStreamReader(with charset) => String
	    private static String getStreamAsString(InputStream stream, String charset) throws IOException {
	        try {
	            return IOUtils.toString(stream, charset);
	        } finally {
	            if (stream != null) {
	                stream.close();
	            }
	        }
	    }
	}

够简单，就一个接口调用：`IOUtils.toString(stream, charset);`搞定。但是其实IOUtils底层也是采用我们法一的做法进行的：

	package org.apache.commons.io;

	public class IOUtils {

		public static String toString(InputStream input, String encoding)
	            throws IOException {
	        StringWriter sw = new StringWriter();
	        copy(input, sw, encoding);
	        return sw.toString();
	    }

	    public static void copy(InputStream input, Writer output, String encoding)
	            throws IOException {
	        if (encoding == null) {
	            copy(input, output);
	        } else {
	            InputStreamReader in = new InputStreamReader(input, encoding);
	            copy(in, output);
	        }
	    }

	    public static void copy(InputStream input, Writer output)
	            throws IOException {
	        InputStreamReader in = new InputStreamReader(input);
	        copy(in, output);
	    }

	    public static int copy(Reader input, Writer output) throws IOException {
	        long count = copyLarge(input, output);
	        if (count > Integer.MAX_VALUE) {
	            return -1;
	        }
	        return (int) count;
	    }

	    public static long copyLarge(Reader input, Writer output) throws IOException {
	        char[] buffer = new char[DEFAULT_BUFFER_SIZE];
	        long count = 0;
	        int n = 0;
	        while (-1 != (n = input.read(buffer))) {
	            output.write(buffer, 0, n);
	            count += n;
	        }
	        return count;
	    }
	}


所以也可以使用IOUtils.copy方法：

    private static String getStreamAsString(InputStream stream, String charset) throws IOException {
        try {
            StringWriter writer = new StringWriter();
            IOUtils.copy(stream, writer, charset);
            return writer.toString();
        } finally {
            if (stream != null) {
                stream.close();
            }
        }
    }


### 法三：使用Google的guava库

用法与Apache的IOUtils十分类似：

	import java.io.FileInputStream;
	import java.io.IOException;
	import java.io.InputStream;
	import java.io.InputStreamReader;

	import com.google.common.io.CharStreams;

	public class InputStreamToStringTest {

	    public static final String CHARSET = "UTF-8";

	    public static void main(String args[]) throws IOException {
	        InputStream inputStream = new FileInputStream("D:/test.txt");

	        String string = getStreamAsString(inputStream, CHARSET);

	        System.out.println(string);
	    }

	    // InputStream => InputStreamReader(with charset) => String
	    private static String getStreamAsString(InputStream stream, String charset) throws IOException {
	        try {
	            return CharStreams.toString(new InputStreamReader(stream, charset));
	        } finally {
	            if (stream != null) {
	                stream.close();
	            }
	        }
	    }
	}

### 法四：使用Java 5 Scanner


Here we have used delimiter as "\A" which is boundary match for beginning of  the input as declared in java.util.regex.Pattern and that's why Scanner is returning whole String form InputStream.


	import java.io.FileInputStream;
	import java.io.IOException;
	import java.io.InputStream;
	import java.util.Scanner;

	public class InputStreamToStringTest {

	    public static final String CHARSET = "UTF-8";

	    public static void main(String args[]) throws IOException {
	        InputStream inputStream = new FileInputStream("D:/test.txt");

	        String string = getStreamAsString(inputStream, CHARSET);

	        System.out.println(string);
	    }

	    // InputStream => InputStreamReader(with charset) => String
	    private static String getStreamAsString(InputStream stream, String charset) throws IOException {
	        try {
	            Scanner scanner = new java.util.Scanner(stream, charset).useDelimiter("\\A");
	            return scanner.hasNext() ? scanner.next() : "";
	        } finally {
	            if (stream != null) {
	                stream.close();
	            }
	        }
	    }
	}


参考文章
--------

1. [Java IO概述](http://blog.arganzheng.me/posts/java-io.html)
