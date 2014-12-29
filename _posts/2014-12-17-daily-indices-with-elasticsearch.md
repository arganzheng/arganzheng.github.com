---
title: ElasticSearch如何实现按天翻滚索引
layout: post
---

最近在做集中式日志，将应用的日志保存到Elasticsearch中，结合kibana实现集中化日志监控和报警。在设计ES存储的时候。考虑到日志的特殊性，打算采用Daily Indices方式。名称为：log-${application}-YYYY.MM.DD。每天会自动创建一个新的索引，类似于log4j的DailyRollingFileAppender，一来减少活跃索引的大小，二来方便存档和删除。

但是这个不是ES内建的功能，需要应用自己实现，方法有很多种。结合ES提供的功能，最简单的方式是采用如下方式：

1. [indices templates](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/indices-templates.html) 为 log-* 的index应用同样的settings、mappings和alias。
2. 启动 [automatic index creation](http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/docs-index_.html#index-creation)，而且支持黑白名单，这样就不需要每次先创建索引了。

具体步骤如下：

#### 1. 配置 elasticsearch.yml，开启自动创建索引白名单，但是仍然关闭动态mapper：

	action.auto_create_index: +log*,-*
	index.mapper.dynamic: false

#### 2. 为log-*的索引添加如下index-templates:

	PUT /_template/log_template
	{
	    "template" : "log-*",
	    "mappings" : {
			"loggingMessage": {
				"properties": {
					"postDate": {
						"type": "date",
          				"format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"
					},
					"createTime": {
						"type": "date",
						"format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"
					},
					"timestamp": {
						"type": "long"
					},
					"env": {
						"type": "string",
						"index": "not_analyzed"
					},
					"type": {
						"type": "string",
						"index": "not_analyzed"
					},
					"host": {
						"type": "string",
						"index": "not_analyzed"
					},
					"application": {
						"type": "string",
						"index": "not_analyzed"
					},
					"threadName": {
						"type": "string"
					},
					"locationInfo": {
						"type": "string"
					},
					"level": {
						"type": "string",
						"index": "not_analyzed"
					},
					"message": {
						"type": "string"
					},
					"stackTrace": {
						"type": "string"
					}
				}
			}
		}
	}

#### 3. Java代码

	/**
	 * <pre>
     *	由于日志的特殊性，这里默认采用Daily Indices方式。名称为：log-${application}-YYYY.MM.DD。
 	 * 	每天会自动创建一个新的索引，类似于log4j的DailyRollingFileAppender，一来减少活跃索引的大小，二来方便存档和删除。
	 * </pre>
	 */
	private String getIndex(String application, Date createTime) {
		if (StringUtils.isEmpty(application) || "null".equalsIgnoreCase(application)) {
			return getIndex(createTime);
		}
		return "log-" + application + "-" + getTimeString(createTime);
	}

	private String getIndex(Date createTime) {
		return "log-" + getTimeString(createTime);
	}

	private String getTimeString(Date date) {
		return DateFormatUtils.format(new Date(), "yyyy.MM.dd");
	}


**NOTE** 关闭auto_create_index和dindex.mapper.dynamic对Kibana的影响

根据ES的推荐，建议将auto_create_index和dindex.mapper.dynamic都关闭。

	action.auto_create_index: +log*,-*
	index.mapper.dynamic: false

但是Kibana的Dashboard Schema也是存储在ES索引中，而且是动态创建的。这就会导致Kibana的Dashboard无法保存。简单而粗暴的解决方案就是手动创建kibana的索引和mapping：

	PUT /kibana-int/dashboard/_mapping
	{
	    "dashboard": {
	        "properties": {
	            "title": {
	                "type":            "string"
	            },
	            "user": {
	                "type":            "string"
	            },
	            "dashboard": {
	                "type":            "string"
	            },
	            "group": {
	                "type":            "string"
	            }
	        }
	    }
	}

网上也有人遇到这样的问题：[Kibana 3 fails to save anything to ElasticSearch if action.auto_create_index=false](https://github.com/elasticsearch/kibana/issues/112)。