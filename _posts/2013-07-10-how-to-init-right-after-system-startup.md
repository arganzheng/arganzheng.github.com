---
layout: post
title: 如何在系统启动时完成资源加载
---


## 背景

在我们的应用中，经常有这样的场景，就是需要在应用启动的时候就预先加载某些资源，如果加载失败，应用都不需要启动了。这些资源的加载，我们都会把他们封装在一个类的loadResource方法中。关键是这个方法如何在系统初始化的时候被调用。一般有两种方式。

### 法一：通过Spring触发loadResource方法

如果这个类是通过Spring加载的，那么可以让Spring初始化之后直接调用该类的loadResource方法。因为Spring框架基本都是系统启动的时候就加载的，而且对于ApplicationContext来说，其bean都是pre-loading的[Lazy loading vs. pre-loading beans with Spring Framework](http://springtips.blogspot.com/2008/09/lazy-loading-vs-pre-loading-beans-with.html)。

方法有如下四种

##### 1. Spring的@PostConstruct注解

这个注解会导致Spring在初始化完Bean之后，调用该bean的@PostConstruct注解的方法。系统启动时候加载Spring，Spring的ApplicationContext会pre-loading所有的bean，@PostConstruct会在bean loading完成后触发init方法。

    @Component
    public class MetadataService {
    
        @Value("${metadata.source}")
        private String                       source         = SOURCE_YIXUN;
    
        public static final String           SOURCE_YIXUN   = "yixun";
        public static final String           SOURCE_WANGGOU = "wanggou";
        public static final String           SOURCE_PAIPAI  = "paipai";
    
        private volatile ApiMetadataDTO      apiMetadata    = null;
        private volatile Map<String, ApiDTO> apiMetadataMap = null;
    
        @PostConstruct
        public void loadMetadata() {
            // 调用IDL得到最新元数据
            if (apiMetadataMap == null) {
                synchronized (this) {
                    if (apiMetadataMap == null) {
                        try {
                            ApiMetadataIdlClient apiMetadataIdlCilent = new ApiMetadataIdlClient(StringUtils.EMPTY, TIMEOUT);
    
                            apiMetadata = apiMetadataIdlCilent.getMetadata(source, true, true);
                            apiMetadataMap = transformToMap(apiMetadata);
                            if (CollectionUtils.isEmpty(apiMetadataMap)) {
                                logger.warn("apiMetadataMap for source " + source + " is Empty!");
                            } else {
                                logger.info("success loading api metadata to map! source: " + source + ",MapSize: "
                                            + apiMetadataMap.size());
                            }
                        } catch (Exception ex) {
                            logger.error("getMetadata failed for source: " + source, ex);
                            System.exit(-1); // 获取不到元数据，相当于废物一个，直接应用退出。
                            // throw new RuntimeException("getMetadata failed！", ex);
                        }
                    }
                }
            }
        }
    }
    

>By default, Spring will not aware of the @PostConstruct and @PreDestroy annotation. To enable it, you have to either register `CommonAnnotationBeanPostProcessor` or specify the `<context:annotation-config />` in bean configuration file,    

##### 2. Spring Bean的init-method

    <bean class="me.arganzheng.study.api.service.MetadataService" init-method="loadMetadata" />

这个跟上面的`@PostConstruct`方式一样，前提是你的bean是通过xml配置的。

##### 3. Spring的InitializingBean接口

    @Component
    public class MetadataService implements InitializingBean {

        @Value("${metadata.source}")
        private String                       source         = SOURCE_YIXUN;
    
        /**
         * 元数据Map。key: methodName
         */
        private volatile ApiMetadataDTO      apiMetadata    = null;
        private volatile Map<String, ApiDTO> apiMetadataMap = null;
    
        public void loadMetadata() {
            // 调用IDL得到最新元数据
            if (apiMetadataMap == null) {
                synchronized (this) {
                    if (apiMetadataMap == null) {
                        try {
                            ApiMetadataIdlClient apiMetadataIdlCilent = new ApiMetadataIdlClient(StringUtils.EMPTY, TIMEOUT);
    
                            apiMetadata = apiMetadataIdlCilent.getMetadata(source, true, true);
                            apiMetadataMap = transformToMap(apiMetadata);
                            if (CollectionUtils.isEmpty(apiMetadataMap)) {
                                logger.warn("apiMetadataMap for source " + source + " is Empty!");
                            } else {
                                logger.info("success loading api metadata to map! source: " + source + ",MapSize: "
                                            + apiMetadataMap.size());
                            }
                        } catch (Exception ex) {
                            logger.error("getMetadata failed for source: " + source, ex);
                            System.exit(-1); // 获取不到元数据，相当于废物一个，直接应用退出。
                            // throw new RuntimeException("getMetadata failed！", ex);
                        }
                    }
                }
            }
        }
    
        @Override
        public void afterPropertiesSet() throws Exception {
            loadMetadata();
        }
    }

##### 4. 通过`BeanPostProcessor`接口

这个其实就是上面`@PostConstruct`注解的底层实现机制，Spring通过`CommonAnnotationBeanPostProcessor`来实现`@PostConstruct`和`@PreDestroy`：

>`org.springframework.beans.factory.config.BeanPostProcessor` implementation that supports common Java annotations out of the box, in particular the JSR-250 annotations in the javax.annotation package. These common Java annotations are supported in many Java EE 5 technologies (e.g. JSF 1.2), as well as in Java 6's JAX-WS. 
>
This post-processor includes support for the `javax.annotation.PostConstruct` and `javax.annotation.PreDestroy` annotations - as init annotation and destroy annotation, respectively - through inheriting from `InitDestroyAnnotationBeanPostProcessor` with pre-configured annotation types. 

    public class InitMetadataServicePostProcessor implements BeanPostProcessor {
    
        @Override
        public Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
            return bean;
        }
    
        @Override
        public Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {    
            if (bean instanceof MetadataService) {
                System.out.println("postProcessAfterInitialization with MetadataService!");

                MetadataService metaDataService = (MetadataService) bean;
                metaDataService.loadMetadata();
            }
            return bean;
        }
    }

需要在xml中注册一下：

    <bean class="me.arganzheng.study.api.tools.common.InitMetadataServicePostProcessor" />

**TIPS** Spring的注解和AOP基本都是通过实现`BeanPostProcessor`接口完成的，可以在bean初始化之后做一些事情。
    
### 法二：通过servlet来完成资源初始化

如果你的bean不是通过Spring容器加载（一般情况下是通过static块来执行资源加载，但是加载的时间是这个类第一次被引用的时候（即被classloader加载的时候）），那么可以通过servlet来完成资源初始化。因为web容器在启动的时候会加载servlet，可以通过servlet来完成应用启动时的资源加载。（其实法一的Spring框架就是通过servlet/ContextListener完成加载的）。
    
    public class MetadataService {
    
        private volatile ApiMetadataDTO       apiMetadata    = null;
        private volatile Map<String, ApiDTO>  apiMetadataMap = null;
        private volatile Map<String, Service> idlMetadataMap = null;
    
        private MetadataService(String source){
            checkUpdate(source);
        }
    
        private static MetadataService instance = new MetadataService(SOURCE_YIXUN);
    
        public static MetadataService getInstance() {
            return instance;
        }
    
        private void checkUpdate(String source) {
            if (apiMetadataMap == null || idlMetadataMap == null) {
                synchronized (this) {
                    if (apiMetadataMap == null || idlMetadataMap == null) {
                        try {
                            ApiMetadataIdlClient apiMetadataIdlCilent = new ApiMetadataIdlClient(StringUtils.EMPTY, TIMEOUT);
                            IdlMetadataIdlClient idlMetadataIdlClient = new IdlMetadataIdlClient(StringUtils.EMPTY, TIMEOUT);
                            apiMetadata = apiMetadataIdlCilent.getMetadata(source, true, true);
                            apiMetadataMap = transformToMap(apiMetadata);
                            if (CollectionUtils.isEmpty(apiMetadataMap)) {
                                logger.warn("apiMetadataMap is Empty!");
                            } else {
                                logger.info("success loading api metadata to map! MapSize: " + apiMetadataMap.size());
                            }
                            GetIdlMetadataResp idlMetadata = idlMetadataIdlClient.getMetadata(source, 0, Integer.MAX_VALUE);
                            idlMetadataMap = trasformToMap(idlMetadata);
                            if (CollectionUtils.isEmpty(idlMetadataMap)) {
                                logger.warn("idlMetadataMap is Empty!");
                            } else {
                                logger.info("success loading idl metadata to map! MapSize: " + idlMetadataMap.size());
                            }
                        } catch (Exception ex) {
                            logger.error("getMetadata failed for source: " + source, ex);
                            throw new RuntimeException("getMetadata failed！", ex);
                        }
                    }
                }
            }
        }
    }


## 参考文章

1. [Lazy loading vs. pre-loading beans with Spring Framework](http://springtips.blogspot.com/2008/09/lazy-loading-vs-pre-loading-beans-with.html)
2. [Spring @PostConstruct And @PreDestroy Example](http://www.mkyong.com/spring/spring-postconstruct-and-predestroy-example/)
3. [Spring Init-Method And Destroy-Method Example](http://www.mkyong.com/spring/spring-init-method-and-destroy-method-example/)
4. [3.13 BeanPostProcessors & BeanFactoryPostProcessors](http://springindepth.com/book/in-depth-ioc-bean-post-processors-and-beanFactory-post-processors.html)