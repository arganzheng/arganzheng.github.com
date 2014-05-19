---
title: 一个简单分页查询组件实现
layout: post
---


背景
----

分页查询是非常常见的需求。所以有必要做成一个比较公用的组件，避免大家重复实现分页功能。


实现
----

分页组件分为前台部门和后台部分。这里先前台说起。

### 前台分页组件

因为我们的网站用的是bootstrap的样式，所以找了一个基于bootstrap的前台分页组件——[bootstrap-paginator](https://github.com/lyonlai/bootstrap-paginator)。使用非常简单，这里关键是用velocity的宏把它定义成一个组件：


	#macro(showPaginator $page)
	    <div id="paginator"></div>
		<form id="form-paginator" style="display:none;" method="POST">
			<input type="hidden" id="pageIndex" name="pageIndex"/>
			<input type="hidden" id="pageSize" name="pageSize"/>
		</form>
	    <script type='text/javascript'>
	        var options = {
	            currentPage: $page.pageIndex,
	            totalPages: $page.pageCount,
	    		numberOfPages: '$!{page.pageSize}' || 10,
	    		size: "small",
	    		alignment: "right",
				itemContainerClass: function (type, page, current) {
	                return (page === current) ? "active" : "pointer-cursor";
	            },
	    		onPageClicked: function(e,originalEvent,type,page){
	    			$('#pageIndex').val(page)
	        		$('#pageSize').val('$!{page.pageSize}' || 10)
	        		$("#form-paginator").submit()
	            }
	        }
	    
	        $('#paginator').bootstrapPaginator(options);
	    	// 增加totalRecord信息
	    	$("#paginator>ul").append('<li><a href="javascript:void(0);">共$page.pageCount页, $page.recordCount条记录</a></li>');
	    	
	    </script>
	#end

然后在需要分页的页面直接这样使用就可以了：


	<script src="$request.getContextPath()/resources/bootstrap/js/bootstrap-paginator.min.js"></script>

	#if($newsList.isNotEmpty())

	#foreach($news in $newsList.getRecords())
	    #parse("partial/news.vm")
	#end

	#showPaginator($newsList)

	#end


### 后端分页组件

后端也蛮简单，不过类比较多，一个个说起。

首先是我们的分页结果类：


	package me.arganzheng.study.pagination.common;

	import java.util.Collections;
	import java.util.List;

	import org.apache.commons.collections.CollectionUtils;

	import me.arganzheng.study.pagination.criteria.PagingCriteria;

	public class Page<T> {

	    private int               recordCount = 0;
	    private int               pageIndex   = 1;
	    private int               pageCount   = 0;
	    private int               pageSize    = 10;
	    private List<T>           records     = Collections.emptyList();

	    @SuppressWarnings("rawtypes")
	    private static final Page emptyPage   = new Page();

	    public boolean isNotEmpty() {
	        return recordCount > 0;
	    }

	    public int getRecordCount() {
	        return recordCount;
	    }

	    public boolean isMultiplePages() {
	        return pageCount > 1;
	    }

	    public boolean isNotLastPage() {
	        return pageIndex != pageCount;
	    }

	    public boolean isNotFirstPage() {
	        return pageIndex != 1;
	    }

	    public int getPageIndex() {
	        return pageIndex;
	    }

	    public int getPageCount() {
	        return pageCount;
	    }

	    public List<T> getRecords() {
	        return records;
	    }

	    public static <T> Page<T> createInstance(List<T> items, int recordCount, int pageIndex, int pageSize) {
	        if (pageSize < 1 || recordCount < 0 || pageIndex < 1) {
	            throw new IllegalArgumentException("Invalid page parameter: recordCount: " + recordCount + ", pageIndex:"
	                                               + pageIndex + ", pageSize: " + pageSize);
	        }
	        int pageCount = calculatePageCount(recordCount, pageSize);
	        Page<T> page = new Page<T>();
	        if (CollectionUtils.isNotEmpty(items)) {
	            page.records = items;
	        }
	        page.recordCount = recordCount;
	        page.pageIndex = pageIndex;
	        page.pageCount = pageCount;
	        page.pageSize = pageSize;
	        return page;
	    }

	    public static <T1, T2> Page<T1> createInstanceFrom(Page<T2> templatePage, List<T1> items) {
	        Page<T1> page = new Page<T1>();
	        if (CollectionUtils.isNotEmpty(items)) {
	            page.records = items;
	        }
	        page.recordCount = templatePage.recordCount;
	        page.pageIndex = templatePage.pageIndex;
	        page.pageCount = templatePage.pageCount;
	        page.pageSize = templatePage.pageSize;
	        return page;
	    }

	    @SuppressWarnings("unchecked")
	    public static <T> Page<T> emptyPage() {
	        return (Page<T>) emptyPage;
	    }

	    public static int calculatePageCount(int recordCount, int pageSize) {
	        int result = recordCount / pageSize;
	        if (recordCount % pageSize != 0) {
	            result++;
	        }
	        return result;
	    }

	    public static <T> Page<T> createInstance(List<T> recordsInAPage, int recordCount, PagingCriteria pagingCriteria) {
	        return Page.createInstance(recordsInAPage, recordCount, pagingCriteria.getPageIndex(),
	                                   pagingCriteria.getPageSize());
	    }

	    public int getPageSize() {
	        return pageSize;
	    }

	    public void setPageSize(int pageSize) {
	        this.pageSize = pageSize;
	    }

	}

然后是我们的分页查询类：

	package me.arganzheng.study.pagination.criteria;

	public class PagingCriteria {

	    public static final int DEFAUT_PAGE_SIZE = 10;
	    private int             pageIndex        = 1;
	    private int             pageSize         = DEFAUT_PAGE_SIZE;

	    public PagingCriteria(){

	    }

	    public PagingCriteria(int pageIndex, int pageSize){
	        this.setPageIndex(pageIndex);
	        this.setPageSize(pageSize);
	    }

	    public PagingCriteria(Integer pageIndex, Integer pageSize){
	        if (pageSize == null || pageSize <= 0) {
	            pageSize = DEFAUT_PAGE_SIZE;
	        }
	        if (pageIndex == null || pageIndex <= 0) {
	            pageIndex = 1;
	        }
	        this.setPageIndex(pageIndex);
	        this.setPageSize(pageSize);
	    }

	    public PagingCriteria(int pageIndex){
	        this.setPageIndex(pageIndex);
	    }

	    public int getFromRecord() {
	        return pageSize * (pageIndex - 1);
	    }

	    public int getPageIndex() {
	        return pageIndex;
	    }

	    public void setPageIndex(int pageIndex) {
	        if (pageIndex < 1) {
	            throw new IllegalArgumentException("Invalid pageIndex: " + pageIndex);
	        }

	        this.pageIndex = pageIndex;
	    }

	    public int getPageSize() {
	        return pageSize;
	    }

	    public void setPageSize(int pageSize) {
	        if (pageSize < 1 || pageSize > 20) {
	            throw new IllegalArgumentException("Invalid pageSize: " + pageSize);
	        }

	        this.pageSize = pageSize;
	    }

	    public static PagingCriteria createFirstPageCriteria(PagingCriteria prototype) {
	        PagingCriteria result = new PagingCriteria(1, prototype.getPageSize());
	        return result;
	    }

	    @Override
	    public int hashCode() {
	        final int prime = 31;
	        int result = 1;
	        result = prime * result + pageIndex;
	        result = prime * result + pageSize;
	        return result;
	    }

	    @Override
	    public boolean equals(Object obj) {
	        if (this == obj) return true;
	        if (obj == null) return false;
	        if (getClass() != obj.getClass()) return false;
	        PagingCriteria other = (PagingCriteria) obj;
	        if (pageIndex != other.pageIndex) return false;
	        if (pageSize != other.pageSize) return false;
	        return true;
	    }

	    @Override
	    public String toString() {
	        return "PagingCriteria [pageIndex=" + pageIndex + ", pageSize=" + pageSize + "]";
	    }

	}


具体的业务分页查询类需要继承这个类：

	package me.arganzheng.study.pagination.news.criteria;

	import java.util.Date;

	import me.arganzheng.study.pagination.PagingCriteria;

	/**
	 * 新闻分页查询对象
	 * 
	 * @author arganzheng
	 * 
	 */
	public class NewsPagingCriteria extends PagingCriteria {
		private String id;

		private Date addTime;

		private String language;

		private String category;

		public Date getAddTime() {
			return addTime;
		}

		public NewsPagingCriteria setAddTime(Date addTime) {
			this.addTime = addTime;
			return this;
		}

		public String getLanguage() {
			return language;
		}

		public NewsPagingCriteria setLanguage(String language) {
			this.language = language;
			return this;
		}

		public String getCategory() {
			return category;
		}

		public NewsPagingCriteria setCategory(String category) {
			this.category = category;
			return this;
		}

		public String getId() {
			return id;
		}

		public void setId(String id) {
			this.id = id;
		}

	}


**说明** 我们这里没有偷懒让PagingCriteria直接extends HashMap，而是要求每个具体因为类都要继承PagingCriteria定义查询条件，目的是这样可以一看这个查询类就知道支持哪些条件查询。

最后是我们的基于MyBatis的一个BaseDao:


	package me.arganzheng.study.pagination.dao;

	import java.util.List;

	import me.arganzheng.study.pagination.common.Page;
	import me.arganzheng.study.pagination.criteria.PagingCriteria;

	import org.apache.ibatis.session.SqlSession;
	import org.apache.log4j.Logger;
	import org.springframework.beans.factory.annotation.Autowired;

	/**
	 * 分页、批量操作在这里进行。
	 * 
	 * @author arganzheng
	 */
	public abstract class BaseDao {

	    protected Logger        logger         = Logger.getLogger(getClass());

	    @Autowired
	    protected SqlSession    sqlSession;

	    public static final int MAX_BATCH_SIZE = 10000;

	    /**
	     * @param countStatementName
	     * @param queryStatementName
	     * @param query
	     * @return 分页结果
	     */
	    protected Page queryForPagination(String queryStatementName, String countStatementName,
	                                      PagingCriteria pagingCriteria) {
	        Integer totalCount = sqlSession.selectOne(countStatementName, pagingCriteria);

	        if (totalCount != null && totalCount.intValue() > 0) {
	            List items = sqlSession.selectList(queryStatementName, pagingCriteria);

	            if (items != null && !items.isEmpty()) {
	                return Page.createInstance(items, totalCount, pagingCriteria);
	            }
	        }

	        return Page.emptyPage();
	    }

	    /**
	     * <pre>
	     * 返回Mapper。如：
	     * 
	     *   BlogMapper mapper = session.getMapper(BlogMapper.class); 
	     *   Blog blog = mapper.selectBlog(101);
	     * </pre>
	     * 
	     * @param clazz
	     * @return
	     */
	    protected <T> T getMapper(Class<T> clazz) {
	        return sqlSession.getMapper(clazz);
	    }
	}


然后是MyBatis的XML映射文件，定义一个CommonsMapper.xml，定义分页SQL语句：

	<?xml version="1.0" encoding="UTF-8" ?>
	<!DOCTYPE mapper
	  PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
	  "http://mybatis.org/dtd/mybatis-3-mapper.dtd">

	<mapper namespace="me.arganzheng.study.pagination.commons">
		<sql id="Pagination">
			limit #{fromRecord}, #{pageSize}
		</sql>
	</mapper>

这样后台如果要使用分页功能，只需要这样定义：

	package me.arganzheng.study.pagination.dao;

	import java.util.List;

	import me.arganzheng.study.pagination.common.Page;
	import me.arganzheng.study.pagination.criteria.NewsPagingCriteria;

	import org.springframework.stereotype.Repository;

	/**
	 * @author arganzheng
	 */
	@Repository
	public class NewsDao extends BaseDao {

	    @SuppressWarnings("unchecked")
	    public Page<News> listMyBookOwnership(NewsPagingCriteria pagingCriteria) {
	        return queryForPagination("me.arganzheng.study.pagination.mapper.NewsMapper.listNews",
	                                  "me.arganzheng.study.pagination.mapper.NewsMapper.countNews",
	                                  pagingCriteria);
	    }


	}


然后在具体的Mapper中定义相应的select语句和count语句：

	<select id="queryNews"
		parameterType="me.arganzheng.study.pagination.criteria.NewsPagingCriteria"
		resultMap="BaseResultMap">
		select
		<include refid="Base_Column_List" />
		from
		<include refid="News_Table_Name" />
		where 1=1
		<choose>
			<when test="id != null and id != '' ">
				and id=#{id}
			</when>
			<otherwise>
				<if test="language != null and language != '' ">
					and language=#{language}
				</if>
				<if test="category != null and category != '' ">
					and category=#{category}
				</if>
				<if test="addTime != null">
			        <![CDATA[   
			            and add_time >= #{addTime}
			        ]]>
				</if>
			</otherwise>
		</choose>
		order by add_time
		desc
		<include
			refid="me.arganzheng.study.pagination.commons.Pagination" />
	</select>

	<select id="countNews"
		parameterType="me.arganzheng.study.pagination.criteria.NewsPagingCriteria"
		resultType="int">
		select count(0)
		from
		<include refid="News_Table_Name" />
		where 1=1
		<choose>
            <when test="id != null and id != '' ">
                and id=#{id}
            </when>
            <otherwise>
                <if test="language != null and language != '' ">
                    and language=#{language}
                </if>
                <if test="category != null and category != '' ">
                    and category=#{category}
                </if>
                <if test="addTime != null">
                    <![CDATA[   
                        and add_time >= #{addTime}
                    ]]>
                </if>
            </otherwise>
        </choose>
		order by add_time
		desc
	</select>	


