---
title: velocity的foreach
layout: post
---


背景：在velocity模版文件中为一个java类生成toString()方法。

原来的做法：

    @Override    
    public String toString(){
        StringBuilder sb = new StringBuilder();       
        #foreach( $field in ${response.fields} )
        sb.append("${field.name}=").append(this.$field.name).append(",");
        #end       
        String s = sb.toString();
        if (s.endsWith(",")) {
            s = s.substring(0, s.length() - 1);
        }       
        return s;
    }

liuhe看了之后，说不好，不要用subString，性能不好。直接判断是否是最后一个特殊处理一下就可以了。于是Google了一下，发现Velocity的foreach其实已经很强大了，支持很多判断：

    #foreach ($foo in $bar)    
        hasNext: $foreach.hasNext
        count: $foreach.count
        index: $foreach.index
        first: $foreach.first 
        last:  $foreach.last
    #end

使用`$foreach.hasNext`和`$foreach.last`可以做到对for循环的最后一个元素特殊处理的效果。

在StackOverFlow上也有人提出这个问题了：[velocity: do something except in last loop iteration](http://stackoverflow.com/questions/8196828/velocity-do-something-except-in-last-loop-iteration)。而且还指出了需要注意的版本问题：
> $foreach variable is only available in velocity 1.7, if you're using an earlier version you can use the $velocityHasNext.

看了一下pom文件，我们使用的是：

    <dependency>
         <groupId>velocity</groupId>
         <artifactId>velocity</artifactId>
         <version>1.4</version>
    </dependency>

于是使用$velocityHasNext。但是一直不生效，直接打印$velocityHasNext，显示的是空值。

很奇怪，将pom中的版本改成1.7发现下载失败，说找不到。最后发现原来velocity在mvnrepository中有两个位置。就是groupId不一样，应该使用后者。
    
    <dependency>
         <groupId>velocity</groupId>
         <artifactId>velocity</artifactId>
         <version>1.5</version>
    </dependency>

    <dependency>
         <groupId>org.apache.velocity</groupId>
         <artifactId>velocity</artifactId>
         <version>1.7</version>
    </dependency>       

改成后者就可以直接使用$foreach.hasNext了：

    @Override   
    public String toString(){
        StringBuilder sb = new StringBuilder();

        sb.append("${response.className}[");       
        #foreach( $field in ${response.fields} )
        sb.append("${field.name}=").append(this.$field.name)#if( $foreach.hasNext ).append(", ")#end;
        #end

        sb.append(']');       
        return sb.toString();
    }
