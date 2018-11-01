设计模式分享
----------

* 开发模式
    * (OOP) 面向对象编程
    * (DDD) 领域驱动开发
    * (TDD) 测试驱动开发
* [设计模式的8个基本原则](http://blog.phpzendo.com/?p=470)
    * 单一职责原则（Single Responsibility Principle）: 一个类只负责一个功能领域中的相应职责（只做一类事情）
    * 里氏代换原则（Liskov Substitution Principle）:  任何基类可以出现的地方，子类一定可以出现
    * 依赖倒转原则（Dependence Inversion Principle）: 核心面向接口编程，依赖于抽象而不依赖于具体实现
    * 接口隔离原则（Interface Segregation Principle : 使用多个隔离的接口，比使用单个庞大的接口要好
    * 迪米特法则，又称最少知道原则（Demeter Principle）: 把复杂留给自己，把简单留给别人
    * 开闭原则（Open Close Principle）: 对扩展开放，对修改关闭。
    * 合成复用原则（Composite Reuse Principle）: 原则是尽量使用合成/聚合的方式，而不是使用继承。
    * 避免过度设计 
* [常见的设计模式](https://blog.csdn.net/Al_assad/article/details/79279719)
    1. 创建型模式 (Creational Patterns)
        1. Abstract Factory: Creates an instance of several families of classes. Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
        2. Builder: Separates object construction from its representation. Separate the construction of a complex object from its representation so that the same construction processes can create different representations.
        3. Factory Method: Creates an instance of several derived classes. Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory Method lets a class defer instantiation to subclasses.
        4. Prototype: A fully initialized instance to be copied or cloned. Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
        5. Singleton: A class of which only a single instance can exist. Ensure a class only has one instance, and provide a global point of access to it.
    2. 结构型模式 (Structural Patterns)
        1. Adapter: Match interfaces of different classes.Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn’t otherwise because of incompatible interfaces.
        2. Bridge: Separates an object’s interface from its implementation. Decouple an abstraction from its implementation so that the two can vary independently.
        3. Composite: A tree structure of simple and composite objects. Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
        4. Decorator: Add responsibilities to objects dynamically.  Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.
        5. Facade: A single class that represents an entire subsystem. Provide a unified interface to a set of interfaces in a system. Facade defines a higher-level interface that makes the subsystem easier to use.
        6. Flyweight: A fine-grained instance used for efficient sharing. Use sharing to support large numbers of fine-grained objects efficiently. A flyweight is a shared object that can be used in multiple contexts simultaneously. The flyweight acts as an independent object in each context — it’s indistinguishable from an instance of the object that’s not shared.
        7. Proxy: An object representing another object. Provide a surrogate or placeholder for another object to control access to it. 
    3. 行为型模式 (Behavioral Patterns)
        1. Chain of Resp. : A way of passing a request between a chain of objects. Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.
        2. Command: Encapsulate a command request as an object. Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
        3. Interpreter: A way to include language elements in a program. Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language.
        4. Iterator: Sequentially access the elements of a collection. Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
        5. Mediator: Defines simplified communication between classes. Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently.
        6. Memento: Capture and restore an object's internal state. Without violating encapsulation, capture and externalize an object’s internal state so that the object can be restored to this state later.
        7. Observer: A way of notifying change to a number of classes. Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
        8. State: Alter an object's behavior when its state changes. Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.
        9. Strategy: Encapsulates an algorithm inside a class. Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
        10. Template: Defer the exact steps of an algorithm to a subclass. Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm’s structure.
        11. Visitor: Defines a new operation to a class without change. Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.
* [UML](https://creately.com/blog/diagrams/uml-diagram-types-examples/)
    * 类图 (Class Diagram)
    * 对象图 (Object Diagram)
    * 包结构图 (Package Diagram)
    * 组件图 (Component diagram)
    * 部署图 (Deployment Diagram)
    * 用例图 (Use Case Diagram)
    * 活动图 (Activity Diagram)
    * 时序图 (Sequence Diagram)
    * 状态图 (State Machine Diagram)
* 程序员的美德
    * 懒惰
    * 洁癖
    * 刨根问底
* 推荐阅读
    * 《设计模式 可复用面向对象软件的基础》 
    * 《Head First设计模式》
    * 《重构:改善既有代码的设计》
    * 《代码整洁之道》
    * 《代码大全》
