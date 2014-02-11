---
title: 一个有意思的Scala函数式编程例子
layout: post
---

问题
----

输入一个Map[String, Any]，要求过滤掉这个map中所有key="version"的元素。

示例输入：

    val subMap = Map("version" -> "3", "hello" -> "world")
    val map = Map("version" -> 1, "query" -> "arganzheng",
      "pageNumber" -> "1", "resultPerPage" -> "10", "data" -> subMap)
      
非函数式的写法
--------------

查看了一下scala的集合操作，是有一个filter的操作，但是这个函数只对当前map的元数据进行过滤，并没有递归处理子map。

    object MapFilterTest {
    
      def main(args: Array[String]) {
        val subMap = Map("version" -> "3", "hello" -> "world")
        val map = Map("version" -> 1, "query" -> "arganzheng",
          "pageNumber" -> "1", "resultPerPage" -> "10", "data" -> subMap)
    
        println(map)
        println(map.filter(_._1 != "version"))
      }
    }

输出结果如下：

    Map(data -> Map(version -> 3, hello -> world), query -> arganzheng, version -> 1, resultPerPage -> 10, pageNumber -> 1)
    Map(data -> Map(version -> 3, hello -> world), query -> arganzheng, resultPerPage -> 10, pageNumber -> 1)

并不满足我们的要求，我们需要对map下的子map也调用filter操作。

于是笔者只能自己动手写一个了，像这种需求一般都是简单的递归处理：

    object MapFilterTest {
        def main(args: Array[String]) {
            val subMap = Map("version" -> "3", "hello" -> "world")
            val map = Map("version" -> 1, "query" -> "arganzheng",
              "pageNumber" -> "1", "resultPerPage" -> "10", "data" -> subMap)
            
            println(map)
            
            val resultMap = visit(map, (key, value) => {
              !key.equalsIgnoreCase("version")
            })

            println(resultMap)
        }
      
        def visit(tree: Map[String, Any], accept: (String, Any) => Boolean): Map[String, Any] = {
            var resultMap = Map[String, Any]();
        
            for (elem <- tree) {
              elem._2 match {
                case x: Map[String, Any] => {
                  resultMap = resultMap.updated(elem._1, visit(x, accept))
                }
                case _ => {
                  if (accept(elem._1, elem._2)) resultMap = resultMap + (elem._1 -> elem._2)
                }
              }
            }
            return resultMap
        }
    }
    
输出结果

    Map(data -> Map(version -> 3, hello -> world), query -> arganzheng, version -> 1, resultPerPage -> 10, pageNumber -> 1)

是正确的，但是看起来很别扭，因为用到了可变对象。

标准函数式的写法
----------------

于是询问了一个scala大神walt哥，大神给了一个标准的FP写法：

    object MapFilterTest {
        def main(args: Array[String]) {
            val subMap = Map("version" -> "3", "hello" -> "world")
            val map = Map("version" -> 1, "query" -> "arganzheng",
              "pageNumber" -> "1", "resultPerPage" -> "10", "data" -> subMap)
            
            println(map)
            
            val resultMap = visit(map, (key, value) => {
              !key.equalsIgnoreCase("version")
            })

            println(resultMap)
        }
      
        def visit(tree: Map[String, Any], accept: (String, Any) => Boolean): Map[String, Any] = {
            tree
              .filter { case (k, v) => accept(k, v) }
              .map {
                case (k, map: Map[String, Any]) => (k, visit(map, accept))
                case (k, v) => (k, v)
              }
        }
    }
    
一看就是典型的FP写法：没有变量、没有可变对象，只是简单的值的变换。

仔细看一下他的实现，其实真的有点类似于写shell管道。scala内建的filter只能对当前的map进行过滤，我们需要对map下的子map也调用filter操作，这不就是map提供的功能吗？不过这里确实避免不了要需要递归。

补记
----

上面的实现有个bug，就是对于map的value为Map数组/列表的没有过滤，修正了一下：


    def visit(tree: Map[String, Any], accept: (String, Any) => Boolean): Map[String, Any] = {
        tree
          .filter { case (k, v) => accept(k, v) }
          .map {
            case (k, map: Map[String, Any]) => (k, visit(map, accept))
            case (k, v: List[Map[String, Any]]) => (k, v.map(travel(_, accept)))
            case (k, v: Vector[Map[String, Any]]) => (k, v.map(travel(_, accept)))
            case (k, v) => (k, v)
          }
    }



    