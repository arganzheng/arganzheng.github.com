---
title: 一个诡异的Antlr4语法问题
layout: post
category: [技术]
tags: [antlr, parser]
catelog: true
---


背景
---


图数据库对外要提供灵活的查询接口，可以有三种层次的实现形式：

* 自然语言: unicorn(facebook)
* 结构化查询语言: Gremlin(Titan), AQL(ArangoDB), Cypher(Neo4j), etc.
* 根据需要预定义的接口: 如一度关系接口，实体查询接口等。

自然语言实现起来过于复杂，准确度不高，不好现实；预定义接口之前在做通用知识图谱支持图搜广告的时候采用过，好处就是高性能，使用简单，但是缺点就是不够灵活；综合来说第二种方式是最合适的。面向图谱的结构化查询语言也有很多，比如AQL(ArangoDB)，OrientDB SQL dialect(OrientDB)，Cypher(Neo4j)，还有Gremlin(Titan)，前面的几种语言都是各个产品特有的查询语言，而Gremlin则是一个开源的图查询语言规范，类似于关系型数据库中的SQL。所以综合考虑，我们决定也采用Gremlin作为我们的查询语言。


**TIPS** 关于Gremlin

图数据库中比较主流的一种

gremlin是一个开源的比较通用的图操作DSL (Domain Specific Language) ，它是类似函数式的，`Path-based`的结构化查询语言，它可以用来检索、维护、和分析图谱。具体语法可以参见：[Apache TinkerPop](http://tinkerpop.apache.org/gremlin.html) 和 [GremlinDocs](http://gremlindocs.spmallette.documentup.com)。


问题
----

之前基于SimpleDB(部门内部的一个分布式KV系统)实现的图数据库是用C++实现的，经过调研之后决定采用 [Spirit](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/index.html) 来实现语法解析这一层的功能，Spirit是boost提供的一个PEG(Parsing Expression Grammar EBNF的一个衍伸)解析器，相对于其他的parser，不需要引用额外的库，而且有很多内建的parser和lexer。

但是这一次GDB是用Java编写的，经过重新调研，决定使用antlr4，它支持语法和解析逻辑分离的方式更利于维护，而且支持多种目标语言。

很快就将语法写好了，并且把visitor也写好了，但是测试的时候，发现在解析范围查询的时候没办法解析出数字。把有问题的语法单独抽取出来，写了一个单元测试，很快就发现问题所在了：


```java
package com.baidu.bdg.kg.gdb.gql;

import java.io.IOException;
import java.io.StringReader;
import java.util.Collections;
import java.util.List;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Parser;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenSource;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.atn.PredictionMode;
import org.antlr.v4.runtime.tree.ParseTree;
import org.junit.Test;

public class TestAntlr4 {

    @Test
    public void test() throws IOException {
        String input = "20";

        CharStream inputCharStream = new ANTLRInputStream(new StringReader(input));
        // create a lexer that feeds off of input CharStream
        TokenSource tokenSource = new TestAntlr4Lexer(inputCharStream);
        // create a buffer of tokens pulled from the lexer
        TokenStream inputTokenStream = new CommonTokenStream(tokenSource);

        // create a parser that feeds off the tokens buffer
        TestAntlr4Parser parser = new TestAntlr4Parser(inputTokenStream);

        parser.removeErrorListeners(); // remove ConsoleErrorListener
        parser.addErrorListener(new VerboseListener());

        parser.getInterpreter().setPredictionMode(PredictionMode.LL_EXACT_AMBIG_DETECTION);

        ParseTree tree = parser.propValue(); // begin parsing at init rule
        System.out.println(tree.toStringTree(parser)); // print LISP-style tree
        System.out.println(tree.toString());
    }

    class VerboseListener extends BaseErrorListener {
        @Override
        public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine,
                String msg, RecognitionException e) {
            List<String> stack = ((Parser) recognizer).getRuleInvocationStack();
            Collections.reverse(stack);
            System.err.println("rule stack: " + stack);
            System.err.println("line " + line + ":" + charPositionInLine + " at " + offendingSymbol + ": " + msg);
        }
    }

    class UnderlineListener extends BaseErrorListener {
        public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine,
                String msg, RecognitionException e) {
            System.err.println("line " + line + ":" + charPositionInLine + " " + msg);
            underlineError(recognizer, (Token) offendingSymbol, line, charPositionInLine);
        }

        protected void underlineError(Recognizer recognizer, Token offendingToken, int line, int charPositionInLine) {
            CommonTokenStream tokens = (CommonTokenStream) recognizer.getInputStream();
            String input = tokens.getTokenSource().getInputStream().toString();
            String[] lines = input.split("\n");
            String errorLine = lines[line - 1];
            System.err.println(errorLine);
            for (int i = 0; i < charPositionInLine; i++) {
                System.err.print(" ");
            }
            int start = offendingToken.getStartIndex();
            int stop = offendingToken.getStopIndex();
            if (start >= 0 && stop >= 0) {
                for (int i = start; i <= stop; i++) {
                    System.err.print("^");
                }
            }
            System.err.println();
        }
    }

}
```

运行会报错如下错误：

	rule stack: [propValue]
	line 1:0 at [@0,0:1='20',<3>,1:0]: no viable alternative at input '20'
	(propValue 20)
	[]

只有int会报错，double(如30.0)和quotedString(如'hello')都可以解析成功。

出错的语法定义如下：

```
grammar TestAntlr4 ;


propValue
	: number
	| quotedString ;

number
	: INT
	| DOUBLE ;

quotedString
	: '"' STRING '"'
	| '\''   STRING  '\'' ;

STRING
	: [a-zA-Z0-9_/-]+ ;

DOUBLE
	: ('+'|'-')? INTEGER '.' INTEGER? ;

INT
	: ('+'|'-')? INTEGER ;


WS
	: [ \t\r\n]+ -> skip ; // skip spaces, tabs, newlines

fragment INTEGER
	: '0'
	| '1'..'9' ('0'..'9')* ;

fragment DIGIT
	: [0-9] ;
```

经过对语法进行删减测试，发现将quotedString和STRING的定义往后面放就没有问题了：


```
grammar TestAntlr4 ;


propValue
	: number
	| quotedString ;

number
	: INT
	| DOUBLE ;


DOUBLE
	: ('+'|'-')? INTEGER '.' INTEGER? ;

INT
	: ('+'|'-')? INTEGER ;


quotedString
	: '"' STRING '"'
	| '\''   STRING  '\'' ;

STRING
	: [a-zA-Z0-9_/-]+ ;
	
WS
	: [ \t\r\n]+ -> skip ; // skip spaces, tabs, newlines

fragment INTEGER
	: '0'
	| '1'..'9' ('0'..'9')* ;

fragment DIGIT
	: [0-9] ;
```

这样子所有的例子就可以解析成功了。


原因很简单，在我们的语法定义中，数字20可以同时被`STRING`和`INT`匹配。如果STRING定义在INT前面，那么number就没有LEXER feed它了。


所有antlr的语法定义顺序还是蛮重要的。一般来说，特例的语法要定义在前面，否则被前面通用的定义匹配了就没有机会匹配了。


**TIPS**

1、antlr的语法调试并不是很容易，出错信息并不太直观。建议定义语法的时候可以以 top-down 的方式编写，但是调试的时候以 buttom-up 的方式调试。

2、buttom-up的方式调试并不意味着你必须一点点的添加语法，事实上你可以在你的测试代码中指定解析的入口rule:

```java
public void testGqlIsValid() throws IOException {
    String query = "30";
    // String query = "g.V('person').has('name', eq, 'argan')";
    // String query = "g.V('Person').has('name', eq, 'argan').out('friend').has('name', eq, 'argan')";

    CharStream inputCharStream = new ANTLRInputStream(new StringReader(query));
    // create a lexer that feeds off of input CharStream
    TokenSource tokenSource = new GqlLexer(inputCharStream);
    // create a buffer of tokens pulled from the lexer
    TokenStream inputTokenStream = new CommonTokenStream(tokenSource);

    // create a parser that feeds off the tokens buffer
    GqlParser parser = new GqlParser(inputTokenStream);

    parser.removeErrorListeners(); // remove ConsoleErrorListener
    parser.addErrorListener(new VerboseListener());

    parser.getInterpreter().setPredictionMode(PredictionMode.LL_EXACT_AMBIG_DETECTION);

	// ParseTree tree = parser.gremlinQuery();
  	ParseTree tree = parser.number();
    System.out.println(tree.toStringTree(parser)); // print LISP-style tree
    System.out.println(tree.toString());
}
```

上面的代码中你指定了解析的入口规则是number（`ParseTree tree = parser.number()`），那么只有number及其下面的rule会被解析和测试到，然后你再一层层的往上走，直到根rule。


