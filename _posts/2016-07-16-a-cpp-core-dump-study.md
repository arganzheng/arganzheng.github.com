---
title: 记一个诡异的C++问题
layout: post
---

昨天联通一直遇到一个诡异的问题：服务某个接口一接受到请求就core dump了。用gdb查看core文件，也没有看出个所以然:

![cy core dump.png](/img/in-post/cy-core.png)

最后实在没有办法，只能采用排除法，把可疑的代码逐行注释掉，检查是不是还有core。因为是一跑就core，所以其实还是很快就定位到问题代码：

```cpp
class StrategyData {
    public:
	StrategyData();
	inline std::string get_search_id() {return _search_id;}
	inline std::string set_search_id(const std::string& search_id){
	    _search_id = search_id;
	}

    public:
	....

    private:
	std::string _search_id;
};
```

set_search_id是一个set函数，应该返回void，但是却不小心定义成return string。在java中这个是会报错的，直接Eclipse就会提示红点。但是在强类型的C++中反而没有问题。。顺利编译通过了。。然后一调用这个函数就core了。。

为了验证是不是这个问题，写一个测试文件：

```cpp
#include <iostream>

inline std::string foo(int i){
    std::cout << i <<  "~~~foo~~~~~" << std::endl;
}

int main(){
    std::cout << "hello world" << std::endl;

    for(int i=0;i<100000000;i++){
	foo(i);
    }

    return 0;
}
```

编译运行一下：

```bash
[arganzheng@st01-bda-asp04.st01.baidu.com ~]$ g++ test.cpp
[arganzheng@st01-bda-asp04.st01.baidu.com ~]$ ./a.out
hello world
0~~~foo~~~~~
Segmentation fault (core dumped)
```

果然屡试不爽。看core堆栈信息也非常莫名其妙：

![test core dump.png](/img/in-post/test-core.png)

奇怪的是把返回值改成int就没有问题。只有string有问题。

但是如果增加`-Wall`编译选项就会提示：

```bash
[arganzheng@st01-bda-asp04.st01.baidu.com ~]$ g++ -Wall test.cpp
test.cpp: In function `std::string foo(int)':
test.cpp:5: warning: no return statement in function returning non-void
test.cpp:5: warning: control reaches end of non-void function
```

所以还是要打开所有warning编译选项：

```bash
-Wall -Werror -Wno-unused-parameter -Wformat -Wconversion -Wdeprecated
```

然后认真认真对待每个warning啊。。


具体原理不懂，期待高手解答。。



