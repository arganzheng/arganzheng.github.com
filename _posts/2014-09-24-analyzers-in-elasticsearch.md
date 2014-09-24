---
title: ElasticSearch的Analyzer
layout: post
---


Analyzer在构建索引的时候，还有分析查询字符串(query string)时候会用到。

Lucene的Analyzer是一个[pineline](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-analyzers.html)的机制，由一个 [Tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-tokenizers.html) + N个 [TokenFilter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-tokenfilters.html) 构成，N>=0。Tokenizer之前还可以配置N个 [CharFilter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-charfilters.html)。

其中各个部件的职责如下：

### [character filters](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-charfilters.html)

Character filters are used to preprocess the string of characters before it is passed to the tokenizer. A character filter may be used to strip out HTML markup, , or to convert "&" characters to the word "and".

### [tokenizers](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-tokenizers.html)

Tokenizers are used to break a string down into a stream of terms or tokens. A simple tokenizer might split the string up into terms wherever it encounters whitespace or punctuation.

### [token filters](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-tokenfilters.html#analysis-tokenfilters)	

Token filters accept a stream of tokens from a tokenizer and can modify tokens (eg lowercasing), delete tokens (eg remove stopwords) or add tokens (eg synonyms).

ES或者说Lucene大概有如下组件：

* [character filters](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-charfilters.html)
	* [mapping char filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-mapping-charfilter.html)
	* [html strip char filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-htmlstrip-charfilter.html)
	* [pattern replace char filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-pattern-replace-charfilter.html)
* [tokenizers](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-tokenizers.html)
	* [standard tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-standard-tokenizer.html)
	* [edge ngram tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html)				
	* [keyword tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-keyword-tokenizer.html)
	* [letter tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-letter-tokenizer.html)
	* [lowercase tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-lowercase-tokenizer.html)
	* [ngram tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-ngram-tokenizer.html)
	* [whitespace tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-whitespace-tokenizer.html)
	* [pattern tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-pattern-tokenizer.html)
	* [uax email url tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-uaxurlemail-tokenizer.html)
	* [path hierarchy tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-pathhierarchy-tokenizer.html)
	* [classic tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-classic-tokenizer.html)
	* [thai tokenizer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-thai-tokenizer.html)
* [token filters](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-tokenfilters.html#analysis-tokenfilters)	
	* [standard token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-standard-tokenfilter.html)
	* [ascii folding token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-asciifolding-tokenfilter.html)
	* [length token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-length-tokenfilter.html)
	* [lowercase token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-lowercase-tokenfilter.html)
	* [uppercase token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-uppercase-tokenfilter.html)
	* [ngram token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-ngram-tokenfilter.html)
	* [edge ngram token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenfilter.html)
	* [porter stem token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-porterstem-tokenfilter.html) 词干过滤器，必须在lowercase filter/tokenizer之后
	* [shingle token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-shingle-tokenfilter.html)
	* [stop token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-stop-tokenfilter.html)
	* [word delimiter token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-word-delimiter-tokenfilter.html)
	* [stemmer token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-stemmer-tokenfilter.html)
	* [stemmer override token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-stemmer-override-tokenfilter.html)
	* [keyword marker token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-keyword-marker-tokenfilter.html)
	* [keyword repeat token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-keyword-repeat-tokenfilter.html)
	* [kstem token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-kstem-tokenfilter.html)
	* [snowball token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-snowball-tokenfilter.html)
	* [phonetic token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-phonetic-tokenfilter.html)
	* [synonym token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html) 同义词
	* [compound word token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-compound-word-tokenfilter.html)
	* [reverse token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-reverse-tokenfilter.html)
	* [elision token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-elision-tokenfilter.html)
	* [truncate token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-truncate-tokenfilter.html)
	* [unique token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-unique-tokenfilter.html)
	* [pattern capture token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-pattern-capture-tokenfilter.html)
	* [pattern replace token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-pattern_replace-tokenfilter.html)
	* [trim token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-trim-tokenfilter.html)
	* [limit token count token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-limit-token-count-tokenfilter.html)
	* [hunspell token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-hunspell-tokenfilter.html)
	* [common grams token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-common-grams-tokenfilter.html)
	* [normalization token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-normalization-tokenfilter.html)
		* [arabic_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/ar/ArabicNormalizer.html)
		* [german_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/de/GermanNormalizationFilter.html)
		* [hindi_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/hi/HindiNormalizer.html)
		* [indic_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/in/IndicNormalizer.html)
		* [sorani_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/ckb/SoraniNormalizer.html)
		* [persian_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/fa/PersianNormalizer.html)
		* [scandinavian_normalization](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/miscellaneous/ScandinavianNormalizationFilter.html), [scandinavian_folding](http://lucene.apache.org/core/4_9_0/analyzers-common/org/apache/lucene/analysis/miscellaneous/ScandinavianFoldingFilter.html)
	* [cjk width token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-cjk-width-tokenfilter.html)
	* [cjk bigram token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-cjk-bigram-tokenfilter.html)
	* [delimited payload token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-delimited-payload-tokenfilter.html)
	* [keep words token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-keep-words-tokenfilter.html)
	* [keep types token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-keep-types-tokenfilter.html)
	* [classic token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-classic-tokenfilter.html)
	* [apostrophe token filter](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-apostrophe-tokenfilter.html)


在ES配置自定义的analyzer是很简单事情，下面是一个例子：

	index :
	    analysis :
	        analyzer :
	            myAnalyzer2 :
	                type : custom
	                tokenizer : myTokenizer1
	                filter : [myTokenFilter1, myGreekLowerCaseFilter]
	                char_filter : [my_html]
	        tokenizer :
	            myTokenizer1 :
	                type : standard
	                max_token_length : 900
	        filter :
	            myTokenFilter1 :
	                type : stop
	                stopwords : [stop1, stop2, stop3, stop4]
	            myGreekLowerCaseFilter :
	                type : lowercase
	                language : greek
	        char_filter :
	              my_html :
	                type : html_strip
	                escaped_tags : [xxx, yyy]
	                read_ahead : 1024

另一个例子：

	index :
	    analysis :
	        analyzer :
	            standard :
	                type : standard
	                stopwords : [stop1, stop2]
	            myAnalyzer1 :
	                type : standard
	                stopwords : [stop1, stop2, stop3]
	                max_token_length : 500
	            # configure a custom analyzer which is
	            # exactly like the default standard analyzer
	            myAnalyzer2 :
	                tokenizer : standard
	                filter : [standard, lowercase, stop]
	        tokenizer :
	            myTokenizer1 :
	                type : standard
	                max_token_length : 900
	            myTokenizer2 :
	                type : keyword
	                buffer_size : 512
	        filter :
	            myTokenFilter1 :
	                type : stop
	                stopwords : [stop1, stop2, stop3, stop4]
	            myTokenFilter2 :
	                type : length
	                min : 0
	                max : 2000


ES已经内建了很多analyzer，一般情况下可以直接使用，不需要自定义。事实上，这些buildin analyzer也是由上面的charfilter、tokenizer和tokenfilter构成的。而另一方面，如果内建的analyzer不符合你的要求，可以很方便的通过custom analyzers进行自定义。

* [standard  analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-standard-analyzer.html)
	* Standard Tokenizer => Standard Token Filter => Lower Case Token Filter => Stop Token Filter
* [simple analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-simple-analyzer.html)
	* Lower Case Tokenizer
* [whitespace analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-whitespace-analyzer.html)
	* Whiterspace Tokenizer
* [stop analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-stop-analyzer.html)
	* Lower Case Tokenizer => Stop Token Filter
* [keyword analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-keyword-analyzer.html)	
* [pattern analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-pattern-analyzer.html)
* [snowball analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-snowball-analyzer.html)
	* standard tokenizer => standard filter => lowercase filter => stop filter => snowball filter
* [language analyzers](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html)
	* stopwords
	* excluding words from stemming
	* The built-in language analyzers can be reimplemented as custom analyzers (as described below) in order to customize their behaviour.	
	* arabic, armenian, basque, brazilian, bulgarian, catalan, chinese, cjk, czech, danish, dutch, english, finnish, french, galician, german, greek, hindi, hungarian, indonesian, irish, italian, latvian, norwegian, persian, portuguese, romanian, russian, sorani, spanish, swedish, turkish, thai.
* [custom analyzer](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-custom-analyzer.html)
	* tokenizer: The logical / registered name of the tokenizer to use.
	* filter: An optional list of logical / registered name of token filters.
	* char_filter: An optional list of logical / registered name of char filters.


**TIPS**

上面列出的都是ES内建的组件，如果不能满足的你要求，可以找一下有没有相关的插件可以使用。
比如这个 [icu analysis plugin](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/analysis-icu-plugin.html]。


### 我们会用到的语言分析器

#### 1. arabic analyzer

The arabic analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "arabic_stop": {
	          "type":       "stop",
	          "stopwords":  "_arabic_" 
	        },
	        "arabic_keywords": {
	          "type":       "keyword_marker",
	          "keywords":   [] 
	        },
	        "arabic_stemmer": {
	          "type":       "stemmer",
	          "language":   "arabic"
	        }
	      },
	      "analyzer": {
	        "arabic": {
	          "tokenizer":  "standard",
	          "filter": [
	            "lowercase",
	            "arabic_stop",
	            "arabic_normalization",
	            "arabic_keywords",
	            "arabic_stemmer"
	          ]
	        }
	      }
	    }
	  }
	}


#### 2. cjk analyzer

The cjk analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "english_stop": {
	          "type":       "stop",
	          "stopwords":  "_english_" 
	        }
	      },
	      "analyzer": {
	        "cjk": {
	          "tokenizer":  "standard",
	          "filter": [
	            "cjk_width",
	            "lowercase",
	            "cjk_bigram",
	            "english_stop"
	          ]
	        }
	      }
	    }
	  }
	}


我们主要是针对日语。如果cjk不好用，可以考虑使用这个分析器插件：[Japanese (Kuromoji) Analysis plugin.](https://github.com/elasticsearch/elasticsearch-analysis-kuromoji)

#### 3. english analyzer

The english analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "english_stop": {
	          "type":       "stop",
	          "stopwords":  "_english_" 
	        },
	        "english_keywords": {
	          "type":       "keyword_marker",
	          "keywords":   [] 
	        },
	        "english_stemmer": {
	          "type":       "stemmer",
	          "language":   "english"
	        },
	        "english_possessive_stemmer": {
	          "type":       "stemmer",
	          "language":   "possessive_english"
	        }
	      },
	      "analyzer": {
	        "english": {
	          "tokenizer":  "standard",
	          "filter": [
	            "english_possessive_stemmer",
	            "lowercase",
	            "english_stop",
	            "english_keywords",
	            "english_stemmer"
	          ]
	        }
	      }
	    }
	  }
	}

#### 4. hindi analyzer

The hindi analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "hindi_stop": {
	          "type":       "stop",
	          "stopwords":  "_hindi_" 
	        },
	        "hindi_keywords": {
	          "type":       "keyword_marker",
	          "keywords":   [] 
	        },
	        "hindi_stemmer": {
	          "type":       "stemmer",
	          "language":   "hindi"
	        }
	      },
	      "analyzer": {
	        "hindi": {
	          "tokenizer":  "standard",
	          "filter": [
	            "lowercase",
	            "indic_normalization",
	            "hindi_normalization",
	            "hindi_stop",
	            "hindi_keywords",
	            "hindi_stemmer"
	          ]
	        }
	      }
	    }
	  }
	}


#### 5. indonesian analyzer

The indonesian analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "indonesian_stop": {
	          "type":       "stop",
	          "stopwords":  "_indonesian_" 
	        },
	        "indonesian_keywords": {
	          "type":       "keyword_marker",
	          "keywords":   [] 
	        },
	        "indonesian_stemmer": {
	          "type":       "stemmer",
	          "language":   "indonesian"
	        }
	      },
	      "analyzer": {
	        "indonesian": {
	          "tokenizer":  "standard",
	          "filter": [
	            "lowercase",
	            "indonesian_stop",
	            "indonesian_keywords",
	            "indonesian_stemmer"
	          ]
	        }
	      }
	    }
	  }
	}

#### 6. portuguese analyzer

The portuguese analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "portuguese_stop": {
	          "type":       "stop",
	          "stopwords":  "_portuguese_" 
	        },
	        "portuguese_keywords": {
	          "type":       "keyword_marker",
	          "keywords":   [] 
	        },
	        "portuguese_stemmer": {
	          "type":       "stemmer",
	          "language":   "light_portuguese"
	        }
	      },
	      "analyzer": {
	        "portuguese": {
	          "tokenizer":  "standard",
	          "filter": [
	            "lowercase",
	            "portuguese_stop",
	            "portuguese_keywords",
	            "portuguese_stemmer"
	          ]
	        }
	      }
	    }
	  }
	}


#### 7. thai analyzer

The thai analyzer could be reimplemented as a custom analyzer as follows:

	{
	  "settings": {
	    "analysis": {
	      "filter": {
	        "thai_stop": {
	          "type":       "stop",
	          "stopwords":  "_thai_" 
	        }
	      },
	      "analyzer": {
	        "thai": {
	          "tokenizer":  "thai",
	          "filter": [
	            "lowercase",
	            "thai_stop"
	          ]
	        }
	      }
	    }
	  }
	}



