---
title: 如何编写高质量单元测试
layout: post
catalog: true
---


### TestingUtils

The org.springframework.test.util package contains several general purpose utilities for use in unit and integration testing.

* ReflectionTestUtils
* AopTestUtils
* AopUtils
* AopProxyUtils


### 上下文管理


* @ContextConfiguration
* @WebAppConfiguration
* @ContextHierarchy
* @ActiveProfiles
* @TestPropertySource
* @DirtiesContext
* @TestExecutionListeners
* @Commit
* @Rollback
* @BeforeTransaction
* @Transactional
* @AfterTransaction
* @SqlConfig
* @SqlGroup


@Autowired
@Qualifier
@Resource (javax.annotation) if JSR-250 is present
@Inject (javax.inject) if JSR-330 is present
@Named (javax.inject) if JSR-330 is present
@PersistenceContext (javax.persistence) if JPA is present
@PersistenceUnit (javax.persistence) if JPA is present
@Required
@Transactional

### 配置项


### 事务




### 数据准备

org.springframework.test.jdbc

* JdbcTestUtils
* AbstractTransactionalJUnit4SpringContextTests 
* AbstractTransactionalTestNGSpringContextTests 


@Sql
Used to annotate a test class or test method to configure SQL scripts to be executed against a given database during integration tests.
@Test
@Sql({"/test-schema.sql", "/test-user-data.sql"})
public void userTest {
    // execute code that relies on the test schema and test data
}

### Mock Objects


1. Environment

MockEnvironment & MockPropertySource

另一种常用做法是使用maven的src/test/resources目录。


### Web测试

org.springframework.test.web

* WebApplicationContext
* ModelAndViewAssert
* MockHttpServletRequest
* MockHttpSession


参考文章
-------

1. [Spring Framework Reference Documentation：14. Unit Testing](http://docs.spring.io/spring/docs/4.3.0.BUILD-SNAPSHOT/spring-framework-reference/htmlsingle/#unit-testing) 关于Spring单元测试官方介绍，非常详细，强烈推荐！
