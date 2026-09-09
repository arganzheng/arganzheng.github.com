---
title: Boost Spirit学习笔记
layout: post
catalog: true
---


Spirit基本介绍
-------------

Spirit is a header file only library. There are no libraries to link to. 

Spirit contains five sub-libraries plus a 'support' module where common support classes are placed:

* Classic: The Old Parsing Library
* Qi: Parsing Library
* Karma: Generating Library
* Lex: A Lexer
* Phoenix: 
* Support


Spirit的基本概念
---------------

### [Nonterminal](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/qi/reference/parser_concepts/nonterminal.html)

A Nonterminal is a symbol in a Parsing Expression Grammar production that represents a grammar fragment. Nonterminals may self reference to specify recursion. This is one of the most important concepts and the reason behind the word "recursive" in recursive descent parsing.

**签名**

Nonterminals can have both synthesized and inherited attributes. The Nonterminal's Signature specifies both the synthesized and inherited attributes. The specification uses the function declarator syntax:

	RT(A0, A1, A2, ..., AN)

where RT is the Nonterminal's synthesized attribute and A0 ... AN are the Nonterminal's inherited attributes.

**TIPS**

The Nonterminal models a C++ function. The Nonterminal's synthesized attribute is analogous to the function return value and its inherited attributes are analogous to function arguments. The inherited attributes (arguments) can be passed in just like any Lazy Argument, e.g.:

	r(expr) // Evaluate expr at parse time and pass the result to the Nonterminal r



### [属性](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/abstracts/attributes/nonterminal_attributes.html)

Spirit具有两种类型的属性：

1. synthesized attributes: return value，grammar或者rule匹配的结果
2. inherited attributes: parameters，grammar或者rule接受的参数，在parser被调用的时候作为参数传递。

Both, the types of the inherited and the synthesized attributes have to be explicitly specified while defining the particular grammar or the rule (the Spirit Repository additionally has subrules which conform to a similar interface). As an example, the following code declares a Spirit.Qi rule exposing an int as its synthesized attribute, while expecting a single double as its inherited attribute (see the section about the Spirit.Qi Rule for more information):

	qi::rule<Iterator, int(double)> r;


**Attributes of Primitive Components**

当match成功的时候匹配的token会被标注：synthesized attribute，例如int_parser如果匹配了一个int输入，这个int值可以通过属性得到。

**Attributes of Composed Components**

attribute type propagation rules for all compound parsers and generators, such as sequences, alternatives, Kleene star, etc. 

The main attribute propagation rule for a sequences is for instance:

	a: A, b: B --> (a >> b): tuple<A, B>


### Phoenix placeholder in Spirit::Qi

spirit + phoenix提供了几种形式的属性占位符：

* _1, _2, ... : Nth attribute of the parser
* _r1, _r2, ... : The enclosing rule’s Nth inherited attribute (for rule<>’s)
	* 在声明rule的时候的local模板参数声明
* _a, _b, ..., _j : The enclosing rule’s local variables. (for rule<>’s)
* _val : The enclosing rule’s synthesized attribute(left hand side’s attribute)
* _pass : Assign false to force parser failure.

可以通过下面这例子学习一下：

	int result = 100;
	std::string input( "12 - 8" );

	rule<iter_t, int(int), space_type> mult, div, add, sub; // => 1
	rule<iter_t, int(), locals<int>, space_type> binary_op;	// => 2

	mult= ’*’>>int_ [_val=_r1*_1]; 
	div = ’/’>>int_ [_val=_r1/_1]; 
	add = ’+’>>int_ [_val=_r1+_1]; 
	sub = ’-’>>int_ [_val=_r1-_1];

	binary_op =     int_				[ _a = _1 ]		// => 3
	             >> (   add(_a)			// => 4
	                  | sub(_a)			// => 4
	                  | mult(_a)		// => 4
	                  | div(_a) 		// => 4
	                )					[ _val = _1 ];	// => 5

	phrase_parse( iter, end,
	              binary_op,
				  space, result );

**说明**

1. int(int) 表示接受一个int参数(inherited attributes)，返回一个int值(synthesized attributes)
2. locals<int> 表示声明一个局部变量(使用_a占位符)
3. 将第一个参数赋值给局部变量_a
4. rule是可以传递参数的，这里就是将前面的局部变量_a作为参数传递给parser，这样在这些parser就可以使用_r1得到传递的参数了。
5. _val表示整个规则解析得到的值(返回值)，这里也就是binary_op规则解析的结果，然后在phrase_parse()函数赋值给了result参数。

再举一个例子说明一下 "_1, _2, ... : Nth attribute of the parser" 这个：

	#include <boost/spirit/include/qi.hpp>
	#include <boost/spirit/include/phoenix.hpp>
	 
	namespace qi = boost::spirit::qi;
	namespace phoenix = boost::phoenix;
	 
	class Point3
	{
	    double x_, y_, z_;
	public:
	    Point3() : x_(0), y_(0), z_(0) {}
	    Point3(double x, double y, double z) : x_(x), y_(y), z_(z) {}
	};
	 
	Point3 pt3;
	std::string input("1.0,2.0,3.0");

	qi::rule<std::string::iterator, Point3()> r =
	    (qi::double_ >> ',' >> qi::double_ >> ',' >> qi::double_)
	    [
	        qi::_val = phoenix::construct<Point3>(qi::_1, qi::_2, qi::_3)
	    ];
	qi::parse(input.begin(), input.end(), r, pt3);

再比如下面这个代码片段：

	primary =
        real                   [_val =  _1]
        |   '(' >> expression  [_val =  _1] >> ')'
        |   ('-' >> primary    [_val = -_1])
        |   ('+' >> primary    [_val =  _1])
        |   (no_case[ufunc] >> '(' >> expression >> ')')
                               [_val = lazy_ufunc(_1, _2)]
        |   (no_case[bfunc] >> '(' >> expression >> ','
                                   >> expression >> ')')
                               [_val = lazy_bfunc(_1, _2, _3)]
        |   no_case[constant]  [_val =  _1]
        ;

现在相信大家应该知道_1,_2,_3分别代表说明了吧~。


### Parser

#### 1. defining parser(语法)

spirit内建了许多[parser](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/qi/quick_reference/qi_parsers.html)，通过组合这些简单类型的内建parser，我们可以写出更复杂的解析器。

building(predefined) Parsers包括如下这些：

* Character Parsers
	* ch: Matches ch
	* lit(ch): Matches ch
	* char_: Matches any character
	* char_(ch): Matches ch
	* char_("c"): Matches a single char string literal, c
	* char_(ch, ch2): Matches a range of chars from ch to ch2 (inclusive)
	* char_(charset): Matches a character set charset
	* alnum: Matches a character based on the equivalent of std::isalnum in the current character set
	* alpha: Matches a character based on the equivalent of std::isalpha in the current character set
	* blank: Matches a character based on the equivalent of std::isblank in the current character set
	* cntrl: Matches a character based on the equivalent of std::iscntrl in the current character set
	* digit: Matches a character based on the equivalent of std::isdigit in the current character set
	* graph: Matches a character based on the equivalent of std::isgraph in the current character set
	* print: Matches a character based on the equivalent of std::isprint in the current character set
	* punct: Matches a character based on the equivalent of std::ispunct in the current character set
	* space: Matches a character based on the equivalent of std::isspace in the current character set
	* xdigit: Matches a character based on the equivalent of std::isxdigit in the current character set
	* lower: Matches a character based on the equivalent of std::islower in the current character set
	* upper: Matches a character based on the equivalent of std::isupper in the current character set
* Numeric Parsers
	* float_: Parse a floating point number into a float
	* float_(num): Parse a floating point number into a float, a number is matched only if it's num
	* double_: Parse a floating point number into a double
	* double_(num): Parse a floating point number into a double, a number is matched only if it's num
	* long_double: Parse a floating point number into a long double
	* long_double(num): Parse a floating point number into a long double, a number is matched only if it's num
	* bin: Parse a binary integer into an unsigned
	* oct: Parse an octal integer into an unsigned
	* hex: Parse a hexadecimal integer into an unsigned
	* ushort_: Parse an unsigned short integer
	* ushort_(num): Parse an unsigned short integer, a number is matched only if it's num
	* ulong_: Parse an unsigned long integer
	* ulong_(num): Parse an unsigned long integer, a number is matched only if it's num
	* uint_: Parse an unsigned int
	* uint_(num): Parse an unsigned int, a number is matched only if it's num
	* ulong_long: Parse an unsigned long
	* ulong_long(num): Parse an unsigned long long, a number is matched only if it's num
	* short_: Parse a short integer
	* short_(num): Parse a short integer, a number is matched only if it's num
	* long_: Parse a long integer
	* long_(num): Parse a long integer, a number is matched only if it's num
	* int_: Parse an int
	* int_(num): Parse an int, a number is matched only if it's num
	* long_long: Parse a long long
	* long_long(num): Parse a long long, a number is matched only if it's num
* String Parsers
	* str: Matches str, with Unused Attribute
	* lit(str): Matches str, with Unused Attribute
	* string(str): Matches str, with Str Attribute
	* symbols<>: Declare a symbol table, sym. Ch is the underlying char type of the symbol table keys. T is the data type associated with each key.
	* sym.add (str1, val1) (str2, val2) /*...more...*/ ; : Add symbols into a symbol table, sym. val1 and val2 are optional data of type T, the data type associated with each key.
	* sym: Matches entries in the symbol table, sym. If successful, returns the data associated with the key
* Auxiliary Parsers
	* eol: Matches the end of line (\r or \n or \r\n)
	* eoi: Matches the end of input (first == last)
	* eps: Match an empty string
	* eps(b): If b is true, match an empty string
	* lazy(fp): Invoke fp at parse time, returning a parser p which is then called to parse.
	* fp: Equivalent to lazy(fp)
	* attr(attrib): Doesn't consume/parse any input, but exposes the argument attrib as its attribute.

composite Parsers(Spirit sequence parsers), eg:

	double_ >> *(char_(',') >> double_)

#### 2. invoking parser

There are a couple of ways to do this.

1、phrase_parse函数:

* An iterator pointing to the start of the input
* An iterator pointing to one past the end of the input
* The parser object
* Another parser called the skip parser

例子：

	template <typename Iterator>
	bool parse_numbers(Iterator first, Iterator last)
	{
	    using qi::double_;
	    using qi::phrase_parse;
	    using ascii::space;

	    bool r = phrase_parse(
	        first,                          // start iterator
	        last,                           // end iterator
	        double_ >> *(',' >> double_),   // the parser
	        space                           // the skip parser
	    );
	    if (first != last) // fail if we did not get a full match
	        return false;
	    return r;
	}

#### 3. Semantic Actions

前面的例子我们定义了一个parser，但是仅仅用它来匹配输入数据(match input data)，并没有做任何其他的操作。

语法:

	P[F]

The expression above links F to the parser, P。其中 P是parser，F是C++ function或者function object. If p is successful, call semantic action, F. The function or function object F is provided the attribute value parsed by the parser p, plus some more context information and a mutable bool flag which can be used to fail parsing.

function/function object 的签名取决于他关联的解析器的类型，例如double_ 解析器会返回匹配的double值，所以它关联的语义操作函数签名如下：

	void F(double n);	

There are actually 2 more arguments being passed (the parser context and a reference to a boolean 'hit' parameter). 


目前有好几种方式可以将解析器和语义操作关联起来(attach semantic actions to parser):

* Using simple function object
* Using Boost.Bind with a plain function
* Using Boost.Bind with a member function
* Using Boost.Lambda
* [Phoenix](http://www.boost.org/doc/libs/1_62_0/libs/spirit/phoenix/doc/html/index.html)

示例：

	namespace client
	{
	    namespace qi = boost::spirit::qi;

	    // A plain function
	    void print(int const& i)
	    {
	        std::cout << i << std::endl;
	    }

	    // A member function
	    struct writer
	    {
	        void print(int const& i) const
	        {
	            std::cout << i << std::endl;
	        }
	    };

	    // A function object
	    struct print_action
	    {
	        void operator()(int const& i, qi::unused_type, qi::unused_type) const
	        {
	            std::cout << i << std::endl;
	        }
	    };
	}

假设我们要解析的数据格式为："{integer}"，那么可以这么样子定义解析器：

attach a plain function:

	parse(first, last, '{' >> int_[&print] >> '}');

attach a simple function object:

	parse(first, last, '{' >> int_[print_action()] >> '}');

或者使用[Boost.Bind](http://www.boost.org/doc/libs/1_62_0/libs/bind/index.html):

	writer w;
	parse(first, last, '{' >> int_[boost::bind(&writer::print, &w, _1)] >> '}');

Boost.Bind也可以绑定普通的函数：

	parse(first, last, '{' >> int_[boost::bind(&print, _1)] >> '}');

还可以使用[Boost.Lambda](http://www.boost.org/doc/libs/1_62_0/libs/lambda/index.html):

	parse(first, last, '{' >> int_[std::cout << _1 << '\n'] >> '}');


假设我们要解析复数，格式为：

	(123.45, 987.65)
	(123.45)
	123.45

那么首先定义语法：

	'(' >> double_ >> -(',' >> double_) >> ')' | double_

然后就可以这样子将解析结果保存起来：

	namespace client
	{
	    template <typename Iterator>
	    bool parse_complex(Iterator first, Iterator last, std::complex<double>& c)
	    {
	        using boost::spirit::qi::double_;
	        using boost::spirit::qi::_1;
	        using boost::spirit::qi::phrase_parse;
	        using boost::spirit::ascii::space;
	        using boost::phoenix::ref;

	        double rN = 0.0;
	        double iN = 0.0;
	        bool r = phrase_parse(first, last,

	            //  Begin grammar
	            (
	                    '(' >> double_[ref(rN) = _1]
	                        >> -(',' >> double_[ref(iN) = _1]) >> ')'
	                |   double_[ref(rN) = _1]
	            ),
	            //  End grammar

	            space);

	        if (!r || first != last) // fail if we did not get a full match
	            return false;
	        c = std::complex<double>(rN, iN);
	        return r;
	    }
	}

**说明**

ref(n) = _1，This assigns the parsed result (actually, the attribute of double_) to n. ref(n) tells Phoenix that n is a mutable reference. _1 is a Phoenix placeholder for the parsed result attribute.

再举一个例子，对数据进行累加：

	template <typename Iterator>
	bool adder(Iterator first, Iterator last, double& n)
	{
	    bool r = qi::phrase_parse(first, last,

	        //  Begin grammar
	        (
	            double_[ref(n) = _1] >> *(',' >> double_[ref(n) += _1])
	        )
	        ,
	        //  End grammar

	        space);

	    if (first != last) // fail if we did not get a full match
	        return false;
	    return r;
	}

还可以通过将解析结果存放到vector中：

	template <typename Iterator>
	bool parse_numbers(Iterator first, Iterator last, std::vector<double>& v)
	{
	    using qi::double_;
	    using qi::phrase_parse;
	    using qi::_1;
	    using ascii::space;
	    using phoenix::push_back;

	    bool r = phrase_parse(first, last,

	        //  Begin grammar
	        (
	            double_[push_back(phoenix::ref(v), _1)]
	                >> *(',' >> double_[push_back(phoenix::ref(v), _1)])
	        )
	        ,
	        //  End grammar

	        space);

	    if (first != last) // fail if we did not get a full match
	        return false;
	    return r;
	}


**TIPS** p % d

对于以 d 分隔的语法，spirit提供了一个简便的语法：p % d。

例如以逗号分隔的double数组，正常语法是：

	double_ >> *(',' >> double_)

可以简化为：

	double_ % ','

read as: a list of doubles separated by ','。


**TIPS** composed parser attributes

在spirit中，任何parser都有属性，所以组合的parser也是有属性的。例如，前面提到的list parser: p % d，它的属性是一个 std::vector of the attribute of p，所以对于 double_ % ','，其对应的属性是 std::vector<double>。所以如果我们只是想取出parser的属性，phrase_parse是支持返回parser属性的：

	template <typename Iterator>
	bool parse_numbers(Iterator first, Iterator last, std::vector<double>& v)
	{
	    using qi::double_;
	    using qi::phrase_parse;
	    using qi::_1;
	    using ascii::space;

	    bool r = phrase_parse(first, last,

	        //  Begin grammar
	        (
	            double_ % ','
	        )
	        ,
	        //  End grammar

	        space, v);

	    if (first != last) // fail if we did not get a full match
	        return false;
	    return r;
	}

### 符号表(Symbol Table)

Traditionally, symbol table management is maintained separately outside the BNF grammar through semantic actions. Contrary to standard practice, the Spirit symbol table class symbols is a parser. An object of which may be used anywhere in the EBNF grammar specification. It is an example of a dynamic parser. A dynamic parser is characterized by its ability to modify its behavior at run time. Initially, an empty symbols object matches nothing. At any time, symbols may be added or removed, thus, dynamically altering its behavior.

Each entry in a symbol table has an associated mutable data slot. In this regard, one can view the symbol table as an associative container (or map) of key-value pairs where the keys are strings.

The symbols class expects two template parameters. The first parameter specifies the character type of the symbols. The second specifies the data type associated with each symbol: its attribute.

下面是一个使用了the symbol table的parser。Keep in mind that the data associated with each slot is the parser's attribute (which is passed to attached semantic actions).

	struct ones_ : qi::symbols<char, unsigned>
	{
	    ones_()
	    {
	        add
	            ("I"    , 1)
	            ("II"   , 2)
	            ("III"  , 3)
	            ("IV"   , 4)
	            ("V"    , 5)
	            ("VI"   , 6)
	            ("VII"  , 7)
	            ("VIII" , 8)
	            ("IX"   , 9)
	        ;
	    }

	} ones;


### 规则(Rules)

A parser expression can be assigned to what is called a "rule". 

签名摘要:

	template <typename Iterator, typename A1, typename A2, typename A3>
	struct rule;

可以看出，规则声明可以有下面两类模板参数：

* Iterator: The underlying iterator type that the rule is expected to work on. 必填，输入流的迭代器类型。
* A1, A2, A3: 可以是 Signature, Skipper 或者 Locals in any order. 
	* Signature: 可选，规定规则的属性，包括synthesized (return value) 和 inherited attributes (arguments)。具体参见:[Nonterminal](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/qi/reference/parser_concepts/nonterminal.html)
	* Skipper: 可选，指定规则的skipper parser。如果没有指定，那么规则就不会skip空格。只能应用于parse函数，因为parse函数就是不需要做skipping。
	* Locals: 可选，指定规则的局部变量，具体参见: [Nonterminal](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/qi/reference/parser_concepts/nonterminal.html)

虽然Signature, Skipper 和 Locals可以以任何顺序出现，不过在一个项目中强烈建议以统一的顺序声明，一般来说推荐使用下面这个声明：
 
	rule<Iterator, Signature, Skipper, Locals> r

一旦声明好了一个规则，就可以给它具体的语法表达式：

	r = double_ >> *(',' >> double_);

### 语法(Grammars)

语法包含多条规则。语法的模板参数跟规则是一样的。

1. deriving a struct (or class) from the grammar class template
3. declare one or more rules as member variables
3. initialize the base grammar class by giving it the start rule (its the first rule that gets called when the grammar starts parsing)
4. initialize your rules in your constructor

具体参考: [Roman Numerals](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/qi/tutorials/roman_numerals.html)。


实战——gremlin语法解析
-------------------

假设我们有一个byId查询语句，作用就是用节点的id查找相关数据，这种查询直接查询正排索引，不会查询倒排索引，速度较快，相当于一次简单的k-v查询。语法如下: 

	g.v(id1[:score1], id2[:score2], id3[:socre3], ...)

中括号内score为可选项，表示人工干预查询的结果的score排名。

假设id是ulong类型，score是double类型，那么我们可以定义如下规则:

    qi::rule<Iterator, std::vector<gremlin::IdInfoPtr>(), encode::space_type, qi::locals<unsigned long, double> > id_info;

具体的语法是：

	id_info = lit(".v") >> '(' >> (((ulong_[_a = _1] >> ( (':' >> double_[_b = _1])
	      |eps[_b = -1]))[push_back(_val, static_cast_<IdInfoPtr>(construct<IdInfo>(_a, _b)))]) % ',') >> ')';

因为id列表是一个逗号隔开的语法，这里使用了`p % d`的简便语法，所以_val返回的属性值类型是 std::vector<>。

	class IdInfo;

	typedef IdInfo* IdInfoPtr;

	class IdInfo {
	private:
		uint64 id;
		double score;
	public:
		...
	}

然后在看看gremlin_query的语法：g.query().has('gender', MATCH, '女').has('type',MATCH,'Engineer').with('*')

	gremlin_query = (char_('g') >> (id_info[_d = _1]
		            |(-(lit(".query") >> '(' >> ')')>> -(prop_query[_a = _1])>> ommit)) >>
		            (*(travel_query[push_back(_b, _1)])) >>
		            -('.' >> (op_query[push_back(_c, _1)] % '.')))[_val = construct<GremlinAST>(_a, _b, _c, _d)] >> eoi;

对应的rule声明如下：

   	qi::rule<Iterator, GremlinAST(), encode::space_type, qi::locals<gremlin::TopPropQueryPtr,
                                                                    ::sofa::vector<gremlin::TravelFunPtr>,
                                                                    ::sofa::vector<gremlin::OpFunPtr>,
                                                                    ::sofa::vector<gremlin::IdInfoPtr> > > gremlin_query;
                                                                    
最后整个gremlin查询语句被解析成一个GremlineAST对象。可以很直观的看出，Gremlin包括pop_query, traval_query, op_query三部分：

struct GremlinAST {
	
	// inv index 处理部分
	TopPropQuery index_query;

	// 遍历部分
	vector<TravelFun> travel_funs;

	// 遍历后操作部分
	vector<OpFun> op_funs;

	// tree/path/graph
	RetType ret_method;

	// 是否需要满足完成的travel_fun路径才算结果返回
	bool full = false;

	// 结果属性with
	PropValue with;

	// 结果属性without
	PropVallue without;

};

参考文档
-------

1. [Nonterminal](http://www.boost.org/doc/libs/1_62_0/libs/spirit/doc/html/spirit/qi/reference/parser_concepts/nonterminal.html)

