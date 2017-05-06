---
title: java 8学习笔记
layout: post
catalog: true
---


Lambda表达式
-----------

Java8主要在原来的面向对象的基础上增加了函数式编程的能力。

例子：比较字符串大小的代码块：

	(String first, String second) -> Integer.compare(first.length(), second.length());

Java中lambda表达式格式：(参数列表) -> 代码块

	(String first, String second) -> {
		if(first.length() < second.length()) return -1;
		else if( first.length() > second.length()) return 1;
		else return 0;
	}

如果lambda表达式没有参数，可以写为：

	() -> { for (int i=0;i<1000;i++) doWork(); }

如果一个lambda表达式的参数类型是可以被推断的，那么可以省略它们的类型，例如：

	Comparator<String> comp = (first, second) -> Integer.compare(first.length(), second.length());

如果某个方法只含有一个参数，并且该参数的类型可以被推断出来，你甚至可以省略小括号：

	EventHandler<ActionEvent> listener = event // 无需 (event) 或者 (ActionEvent event)
										-> System.out.println("Thanks for clicking!");

										