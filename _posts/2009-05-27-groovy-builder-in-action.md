---
title: Groovy元编程——使用invokeMethod和闭包构建DSL和Builder
layout: post
---


Since 1.0, Groovy supports the ability to intercept all method and property access via the invokeMethod and get/setProperty hooks. If you only want to intercept failed method/property access take a look at [Using methodMissing and propertyMissing](http://groovy.codehaus.org/Using+methodMissing+and+propertyMissing).


invokeMethod介绍
----------------

invokeMethod(String name, Object args) is at the heart of Groovy metaprogramming. Every method call on an object is intercepted by invoke-Method. The name parameter is the method call. The args parameter is an Object array that catches all subsequent parameters . 

In any Groovy class you can override invokeMethod which will essentially intercept all method calls (to intercept calls to existing methods, the class additionally has to implement the GroovyInterceptable interface). This makes it possible to construct some quite interesting DSLs and builders.

InvokeMethod将会拦截Groovy对象的所有定义或者未定义的方法。如果要拦截已经定义的方法，还需要实现GroovyInterceptable接口。如果只是要拦截未定义方法，那么可以考虑使用`methodMissing`和`propertyMissing`。

下面让我们看一个简单又强大的例子——使用invokeMethod和闭包实习一个简单的XmlBuilder，这个XmlBuilder允许用户采用声明式的语法构建一个XML文件：

For example a trivial XmlBuilder could be written as follows (note Groovy ships with much richer XML APIs and this just serves as an example):

    class XmlBuilder {
        def out
        XmlBuilder(out) { this.out = out }
        def invokeMethod(String name, args) {
            out << "<$name>"
            if(args[0] instanceof Closure) { 
                  args[0].delegate = this  //委托给自己（实际上是设置闭包的上下文）
                  args[0].call() //调用该闭包，由于是委托给自己，所以有点递归的味道，实际上不是。
            }
            else {
                out << args[0].toString()
            }
            out << "</$name>"
        }
    }

    class HelloGroovy{
        /**
         * @param args
         */
        public static void main(def args){
            // TODO Auto-generated method stub
            def xml = new XmlBuilder (new StringBuilder());
            xml.html{
                head{
                    title "Hello World"
                }
                body{
                    p "Welcome!"
                }
            }
            println xml.out;
        }
    }


运行这个程序，将产生如下输出：

    <html><head><title>Hello World</title></head><body><p>Welcome!</p></body></html>

从这个简单的例子足以让我们见识到MOP（MetaProgramming Oriented Programming，面向元编程)与闭包（Closure）结合所带来的巨大生产力。

下面让我们举一个现实的例子。

需求：根据msg的消息类型MessageType和status确定其操作，也就是实现如下表的映射：

![消息路由](/media/images/builder-for-message-routing.jpg)

常规做法——使用Java做if-else条件判断：
    
    MessageType type = msg.getMessageType();
    int status = msg.getStatus();
    int actionFlag = ACTION_DROP;

    // set operation type according to rules
    if (type == ADD) {
      actionFlag = ADD;
    } else if (type == CREATE_REJECT) {
      actionFlag = DELETE;
    } else if ((type == CREATE_TEST) || (type == MessageType.UPDATE)) {
      actionFlag = UPDATE;
    } else if ((status == STATUS_NORMAL) || (status == STATUS_PRE_WITHDRAW)) {
      actionFlag = UPDATE;
    } else if (status == STATUS_WITHDRAW) {
      actionFlag = DELETE;
    }

    return actionFlag;
  
现在让我们看一下Groovy的做法：

    def type = msg.getMessageType();
    def status = msg.getStatus();
    def target = msg.getTarget();
        
    def builder = new OptypeMappingBuilder(target, type, status);
        
    builder{
        pismp4roaming{
            A(ADD);
            D(CREATE_REJECT);
            U(CREATE_TEST);
            U(UPDATE);
            U(END, STATUS_NORMAL);
            U(END, STATUS_PRE_WITHDRAW);
            D(END, STATUS_WITHDRAW);
        }
    }
    return builder.getActionFlag();

这个做法与我们上面的XmlBuilder有异曲同工之处。正是利用了InvokeMethod和闭包（也许还有递归的思想在里面）。我们来看看OptypeMappingBuilder{}的定义：

    package processors.common;
    import org.apache.log4j.Logger;
    class OptypeMappingBuilder {
        //Properties
        def target=null;
        def List values=null;
       
        //私有内部变量
        private final static Logger logger = Logger.getLogger(OptypeMappingBuilder.class);
        
        private int actionFlag=-1;
        private String action="";
        private map=["":-1,"A":1,"U":2,"D":3];//actionFlag maping
        
        //构造函数
        OptypeMappingBuilder() { ;actionFlag=-1;}
        OptypeMappingBuilder(target) { this.target=target;}
        OptypeMappingBuilder(target,Object[] values) {
            this.target=target;
            this.values=values;
            if(logger.isDebugEnabled()){
                StringBuilder sb = new StringBuilder();
                sb.append("Optype mapping for target [").append(target).append("], criterions [").append(values.toString()).append("]"); //使用$value语法会更简略
                logger.debug(sb);
            }        
        }
        
        //MOP函数：其中name是被拦截的函数名，args是该函数的参数
        def invokeMethod(String name, args) {
            if(["A","D","U","CALL",target.toUpperCase()].contains(name.toUpperCase())){
                //closure without parameter
                if (args.length == 1 && args[0] instanceof Closure) {
                    args[0].delegate = this;
                    args[0].call();
                } else {
                    if (args.length >= 1 && ["A", "D", "U"].contains(name.toUpperCase()) && actionFlag == -1) {
                        //respensent a specific rule, if values agree with args, the rule is matched
                        def boolean equal = true;
                        int argv = args.size();
                        if (argv <= values.size()) {
                            for (def i = 0; i < argv; ++i) {
                                if (args[i]!=null) {//args[i] is not null or empty string
                                    if (!(args[i].equals(values[i]))) {
                                        equal = false;
                                        break;
                                    }
                                }
                            }
                            if (equal) {
                                actionFlag = map.get(name);
                                if(logger.isDebugEnabled()){
                                    StringBuilder sb = new StringBuilder();
                                    sb.append("Optype mapping rule [")
                                        .append(target).append(" ")
                                        .append(name).append(args.toString())
                                        .append("] matched. ActionFlag = [")
                                        .append(actionFlag).append("]");
                                    logger.debug(sb);
                                }
                                return;
                            }
                        }
                    }
                }
            }
        }
     
        public getActionFlag(){
            return actionFlag;
        }
    }
 