---
title: Java8时间处理
layout: post
---

用Java处理过时间的基本都会有一个感觉，就是麻烦，甚至恶心。

老的Java 时间API的核心类和方法比较少，如下：

* System.currentTimeMillis(): static method that returns the current date and time as milliseconds since January 1st 1970.
* java.util.Date: 表示一个特定的时间，精度是毫秒(millisecond)。很多方法都是deprecated..
* java.sql.Date: 表示一个日期，不包含时间信息，定位是用于JDBC，但是实际上基本不用。。
* java.sql.Timestamp: 继承java.util.Date，表示一个date time。同样定位是用于JDBC，但是实际上基本不用。。
* java.util.Calendar: 表示一个日历时间，比如年月日星期。有时区的概念。包含对于日期的算术操作，比如加减日期。
	* java.util.GregorianCalendar: representing the Gregorian calendar which is used in most of the western world today.
* java.text.DateFormat: 格式化时间
	* java.text.SimpleDateFormat
* java.util.TimeZone: 表示一个time zone偏移，同时会处理daylight，在做不同时区的日历算术时很有用。



可以看到Java8之前对时间日期的封装是非常混乱的，分散在各个package中，而且对时间日期的抽象也是有所欠缺的。其中SimpleDateFormat还是非线程安全的。

所以很多时候我们都会求救于第三方库，比如`org.apache.commons.lang`的DateUtils和DateFormatUtils，或者[Joda Time](http://www.joda.org/joda-time/)库。

Java显然也意识到这个问题了。所以在Java 8引入了一套全新的时间日期API，借鉴了Joda库的一些优点，比如将人和机器对时间日期的理解区分开的(readable date VS machine time(unix timestamp))，这对已经习惯使用Joda时间日期库的社区而言也是件好事。这个新的时间日期库的最大的优点就在于它清楚的定义了时间日期相关的一些概念，例如，瞬时时间(Instant)，持续时间(duration)，日期(date)，时间(time)，时区(time-zone)以及时间段(Period)。Java 8仍然延用了ISO的日历体系(ISO-8601 calendar system)，但是我们也可以将它用于非ISO的日历体系。并且与它的前辈们不同，java.time包中的类是不可变且线程安全的。另外，非常方便的是所有的核心类基本都包含日期计算，并且不用担心线程安全问题。


新的时间及日期API主要包含在下面几个package:

* java.time: 这个是新Javag Date Time API的基础包。所有核心类都放在这个package下，比如LocalDate, LocalTime, LocalDateTime, Instant, Period, Duration 等等。所有这些类都是不可变并且是线程安全的。大部分情况下，使用这些类就足够完成常见的时间日期需求。
* java.time.chrono: 这个package定义了非ISO日历体系的通用API，比如日本，泰国等。我们可以扩展其中的类来创建我们自己的日历体系。
* java.time.format: 这个package包含用于格式化(formatting)和解析(parsing)日期(date time)对象的类。大部分情况下，我们不需要直接使用它们，因为`java.time` package中的核心类已经提供了formatting和parsing的方法了。
* java.time.temporal: 正如名字显示的，这个package包含了一些临时对象，可以让我们用来确定特别的日期或者时间。比如，我们可以用它来查找某个月的第一天或者最后一天。所有的方法基本都是以“withXXX”格式命名。
* java.time.zone: 这个package包含了一些类用于支持不同的时区。


下面是里面的一些关键的类：

* Instant: 它代表的是时间戳。在Java 7 date time API，时间戳是以a number of millseconds since Jan. 1st. 1970，在Java 8 则表现为a number of seconds and a number of nanoseconds since Jan. 1st 1970.
* LocalDate: A date without a time-zone in the ISO-8601 calendar system, such as 2007-12-03.
* LocalTime: 它代表的是不含日期的时间。
* LocalDateTime: 它包含了日期及时间，不过还是没有偏移信息或者说时区。
* ZonedDateTime: 这是一个包含时区的完整的日期时间，偏移量是以UTC/格林威治时间为基准的。
* Period & Duration: 时间间隔，类似于MySQL中的Interval。其中Period是date-based amount of time in the ISO-8601 calendar system, such as '2 years, 3 months and 4 days'，而Duration则是A time-based amount of time, such as '34.5 seconds'。This class models a quantity or amount of time in terms of seconds and nanoseconds. It can be accessed using other duration-based units, such as minutes and hours.

新的库还增加了ZoneOffset及Zoned，可以为时区提供更好的支持。有了新的DateTimeFormatter之后日期的解析及格式化也变得焕然一新了。下面让我们看一些具体例子：

#### 1. java.time.LocalDate

LocalDate is an immutable class that represents Date with default format of yyyy-MM-dd. We can use now() method to get the current date. We can also provide input arguments for year, month and date to create LocalDate instance. This class provides overloaded method for now() where we can pass ZoneId for getting date in specific time zone.

	public class LocalDateExample {
	 
	    public static void main(String[] args) {
	         
	        //Current Date
	        LocalDate today = LocalDate.now();
	        System.out.println("Current Date="+today);
	         
	        //Creating LocalDate by providing input arguments
	        LocalDate firstDay_2015 = LocalDate.of(2015, Month.JANUARY, 1);
	        System.out.println("Specific Date="+firstDay_2015);
	         
	        //Current date in "Asia/Kolkata", you can get it from ZoneId javadoc
	        LocalDate todayKolkata = LocalDate.now(ZoneId.of("Asia/Kolkata"));
	        System.out.println("Current Date in IST="+todayKolkata);
	 
	        //Getting date from the base date i.e 01/01/1970
	        LocalDate dateFromBase = LocalDate.ofEpochDay(365);
	        System.out.println("365th day from base date= "+dateFromBase);
	         
	        LocalDate hundredDay2014 = LocalDate.ofYearDay(2014, 100);
	        System.out.println("100th day of 2014="+hundredDay2014);
	    }
	}

#### 2. java.time.LocalTime

LocalTime is an immutable class whose instance represents a time in the human readable format. It’s default format is hh:mm:ss.zzz. Just like LocalDate, this class provides time zone support and creating instance by passing hour, minute and second as input arguments. Let’s look at it’s usage with a simple program.

	public class LocalTimeExample {
	 
	    public static void main(String[] args) {
	         
	        //Current Time
	        LocalTime time = LocalTime.now();
	        System.out.println("Current Time="+time);
	         
	        //Creating LocalTime by providing input arguments
	        LocalTime specificTime = LocalTime.of(12,20,25,40);
	        System.out.println("Specific Time of Day="+specificTime);
	         
	        //Current date in "Asia/Kolkata", you can get it from ZoneId javadoc
	        LocalTime timeKolkata = LocalTime.now(ZoneId.of("Asia/Kolkata"));
	        System.out.println("Current Time in IST="+timeKolkata);
	 
	        //Getting date from the base date i.e 01/01/1970
	        LocalTime specificSecondTime = LocalTime.ofSecondOfDay(10000);
	        System.out.println("10000th second time= "+specificSecondTime);
	    }
	}

#### 3. java.time.LocalDateTime

LocalDateTime is an immutable date-time object that represents a date-time, with default format as yyyy-MM-dd-HH-mm-ss.zzz. It provides a factory method that takes LocalDate and LocalTime input arguments to create LocalDateTime instance. 


	public class LocalDateTimeExample {
	 
	    public static void main(String[] args) {
	        //Current Date
	        LocalDateTime today = LocalDateTime.now();
	        System.out.println("Current DateTime="+today);
	         
	        //Current Date using LocalDate and LocalTime
	        today = LocalDateTime.of(LocalDate.now(), LocalTime.now());
	        System.out.println("Current DateTime="+today);
	         
	        //Creating LocalDateTime by providing input arguments
	        LocalDateTime specificDate = LocalDateTime.of(2014, Month.JANUARY, 1, 10, 10, 30);
	        System.out.println("Specific Date="+specificDate);
	         
	         
	        //Current date in "Asia/Kolkata", you can get it from ZoneId javadoc
	        LocalDateTime todayKolkata = LocalDateTime.now(ZoneId.of("Asia/Kolkata"));
	        System.out.println("Current Date in IST="+todayKolkata);
	 
	        //Getting date from the base date i.e 01/01/1970
	        LocalDateTime dateFromBase = LocalDateTime.ofEpochSecond(10000, 0, ZoneOffset.UTC);
	        System.out.println("10000th second time from 01/01/1970= "+dateFromBase);
	    }
	}

#### 4. java.time.Instant

Instant class is used to work with machine readable time format, it stores date time in unix timestamp.

	public class InstantExample {
	 
	    public static void main(String[] args) {
	        //Current timestamp
	        Instant timestamp = Instant.now();
	        System.out.println("Current Timestamp = " + timestamp);
	         
	        //Instant from timestamp
	        Instant specificTime = Instant.ofEpochMilli(timestamp.toEpochMilli());
	        System.out.println("Specific Time = "+specificTime);
	         
	        //Duration example
	        Duration thirtyDay = Duration.ofDays(30);
	        System.out.println(thirtyDay);
	    }
	 
	}


#### 5. Date API Utilities

基本上所有的Date Time核心类都提供了工具方法用于plus/minus days, weeks, months等，不需要额外使用DateUtils.addXXX()方法了. 还有一些Utils方法使用TemporalAdjuster来调整日期。还有计算两个日期之间的时间间隔(period)。

	public class DateAPIUtilities {
	 
	    public static void main(String[] args) {
	         
	        LocalDate today = LocalDate.now();
	         
	        //Get the Year, check if it's leap year
	        System.out.println("Year "+today.getYear()+" is Leap Year? "+today.isLeapYear());
	         
	        //Compare two LocalDate for before and after
	        System.out.println("Today is before 01/01/2015? "+today.isBefore(LocalDate.of(2015,1,1)));
	         
	        //Create LocalDateTime from LocalDate
	        System.out.println("Current Time="+today.atTime(LocalTime.now()));
	         
	        //plus and minus operations
	        System.out.println("10 days after today will be "+today.plusDays(10));
	        System.out.println("3 weeks after today will be "+today.plusWeeks(3));
	        System.out.println("20 months after today will be "+today.plusMonths(20));
	 
	        System.out.println("10 days before today will be "+today.minusDays(10));
	        System.out.println("3 weeks before today will be "+today.minusWeeks(3));
	        System.out.println("20 months before today will be "+today.minusMonths(20));
	         
	        //Temporal adjusters for adjusting the dates
	        System.out.println("First date of this month= "+today.with(TemporalAdjusters.firstDayOfMonth()));
	        LocalDate lastDayOfYear = today.with(TemporalAdjusters.lastDayOfYear());
	        System.out.println("Last date of this year= "+lastDayOfYear);
	         
	        Period period = today.until(lastDayOfYear);
	        System.out.println("Period Format= "+period);
	        System.out.println("Months remaining in the year= "+period.getMonths());        
	    }
	}

输出：

	Year 2015 is Leap Year? false
	Today is before 01/01/2015? false
	Current Time=2015-11-17T14:32:34.585
	10 days after today will be 2015-11-27
	3 weeks after today will be 2015-12-08
	20 months after today will be 2017-07-17
	10 days before today will be 2015-11-07
	3 weeks before today will be 2015-10-27
	20 months before today will be 2014-03-17
	First date of this month= 2015-11-01
	Last date of this year= 2015-12-31
	Period Format= P1M14D
	Months remaining in the year= 1


#### 6. Parsing and Formatting

解析和格式化日期是最常见的操作了。现在Date对象直接就包含format方法了，不需要使用SimpleDateFormat了。而且`DateTimeFormatter`内建了许多日期格式，基本不需要自定义。

	public class DateParseFormatExample {
	 
	    public static void main(String[] args) {
	         
	       // Format examples
	        LocalDate date = LocalDate.now();
	        // default format
	        System.out.println("Default format of LocalDate=" + date);
	        // specific format
	        System.out.println(date.format(DateTimeFormatter.ofPattern("d::MMM::uuuu")));
	        System.out.println(date.format(DateTimeFormatter.BASIC_ISO_DATE));

	        LocalDateTime dateTime = LocalDateTime.now();
	        // default format
	        System.out.println("Default format of LocalDateTime=" + dateTime);
	        // specific format
	        System.out.println(dateTime.format(DateTimeFormatter.ofPattern("d::MMM::uuuu HH::mm::ss")));
	        System.out.println(dateTime.format(DateTimeFormatter.BASIC_ISO_DATE));

	        Instant timestamp = Instant.now();
	        // default format
	        System.out.println("Default format of Instant=" + timestamp);

	        // Parse examples
	        LocalDateTime dt =
	                LocalDateTime.parse("16::Nov::2015 17::39::48", DateTimeFormatter.ofPattern("d::MMM::uuuu HH::mm::ss"));
	        System.out.println("Default format after parsing = " + dt);
	    }
	 
	}

输出：

	Default format of LocalDate=2015-11-17
	17::Nov::2015
	20151117
	Default format of LocalDateTime=2015-11-17T14:38:51.293
	17::Nov::2015 14::38::51
	20151117
	Default format of Instant=2015-11-17T06:38:51.293Z
	Default format after parsing = 2015-11-16T17:39:48


#### 7. Legacy Date Time Support

为了能够兼容老的Date Time对象，新API提供了几个工具方法，让我们可以在新老日期对象之间进行转换。


	public class DateAPILegacySupport {
	 
	    public static void main(String[] args) {
	         
	        // Date to Instant
	        Instant timestamp = new Date().toInstant();
	        // Now we can convert Instant to LocalDateTime or other similar classes
	        LocalDateTime date = LocalDateTime.ofInstant(timestamp, ZoneId.of(ZoneId.SHORT_IDS.get("PST")));
	        System.out.println("Date = " + date);

	        // Calendar to Instant
	        Instant time = Calendar.getInstance().toInstant();
	        System.out.println(time);
	        // TimeZone to ZoneId
	        ZoneId defaultZone = TimeZone.getDefault().toZoneId();
	        System.out.println(defaultZone);

	        // ZonedDateTime from specific Calendar
	        ZonedDateTime gregorianCalendarDateTime = new GregorianCalendar().toZonedDateTime();
	        System.out.println(gregorianCalendarDateTime);

	        // Date API to Legacy classes
	        Date dt = Date.from(Instant.now());
	        System.out.println(dt);

	        TimeZone tz = TimeZone.getTimeZone(defaultZone);
	        System.out.println(tz);

	        GregorianCalendar gc = GregorianCalendar.from(gregorianCalendarDateTime);
	        System.out.println(gc);
	    }
	}

输出：

	Date = 2015-11-16T22:43:24.318
	2015-11-17T06:43:24.478Z
	Asia/Shanghai
	2015-11-17T14:43:24.502+08:00[Asia/Shanghai]
	Tue Nov 17 14:43:24 CST 2015
	sun.util.calendar.ZoneInfo[id="Asia/Shanghai",offset=28800000,dstSavings=0,useDaylight=false,transitions=19,lastRule=null]
	java.util.GregorianCalendar[time=1447742604502,areFieldsSet=true,areAllFieldsSet=true,lenient=true,zone=sun.util.calendar.ZoneInfo[id="Asia/Shanghai",offset=28800000,dstSavings=0,useDaylight=false,transitions=19,lastRule=null],firstDayOfWeek=2,minimalDaysInFirstWeek=4,ERA=1,YEAR=2015,MONTH=10,WEEK_OF_YEAR=47,WEEK_OF_MONTH=3,DAY_OF_MONTH=17,DAY_OF_YEAR=321,DAY_OF_WEEK=3,DAY_OF_WEEK_IN_MONTH=3,AM_PM=1,HOUR=2,HOUR_OF_DAY=14,MINUTE=43,SECOND=24,MILLISECOND=502,ZONE_OFFSET=28800000,DST_OFFSET=0]


参考文章
------

1. [Java 8 Date Time API Example Tutorial – LocalDate, Instant, LocalDateTime, Parse and Format](http://www.journaldev.com/2800/java-8-date-time-api-example-tutorial-localdate-instant-localdatetime-parse-and-format)
2. [Java Date Time Tutorial](http://tutorials.jenkov.com/java-date-time/index.html) 非常详细，强烈推荐。



