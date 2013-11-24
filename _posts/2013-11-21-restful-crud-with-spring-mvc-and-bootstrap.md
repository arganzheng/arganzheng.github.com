---
layout: post
title: RESTful CRUD with spring-mvc-and-bootstrap
---


bootstrap非常简单好用，以一个简单的实例说明一下。假设我们有一个事件表。

	public class Event{
	    int id;
	    String topic;
	    String name;
	}

我们要搞一个配置页面，让后台业务人员进行配置。

### URL规划

参考这篇文章：[URL Conventions](http://microformats.org/wiki/rest/urls)

GET /event
    return a list of all events

POST /event
    submit an event for create or update(if id>0)  

DELETE /event/{name}
    delete an event

说明：由于采用了Modal和ajax技术，所以减少了很多页面。


### HTML页面 event_list.vm

	#set( $layout = "layout/default.vm" )

	<script type="text/javascript">
	function addEvent(topic){
		addOrEditEvent({'topic': topic}, function(event){
			alert("保存成功！");
			setTimeout(window.location.reload(), 2000);
		});
	}

	function editEvent(event){
		addOrEditEvent(event, function(event){
			alert("保存成功！");
			setTimeout(window.location.reload(), 2000);
		});
	}

	var basePath = location.protocol + "//" + location.host

	function deleteEvent(name){
		if(!confirm("确定要删除吗？")){
			return;
		}
		
		$.ajax(basePath + "/admin/event/" + name, {
			type: "DELETE",
			dataType: "json",
			contentType: "application/json",
			success: function(data, textStatus){
			    if(data==true){
					alert("删除成功！")
					setTimeout(window.location.href="/event", 2000);
				}else{
					alert("删除失败！"+ data.errorMessage);
				}
			},
			error: function(xhr, ts, e){
				alert("系统错误！\n\n" + e);
			}
		});
	}

	function addOrEditEvent(event, successCallback){
		var dlg = $(
			'<div class="modal fade" style="width: 480px;">' +
				'<div class="modal-dialog">' +
					'<div class="modal-content">' +
						'<div class="modal-header">' + 
				        	'<button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button>' +
				        	'<h4>Event</h4>' +
				        '</div>' +  // .modal-header
				        '<div class="modal-body">' + 
				        	'<input type="hidden" name="id" />' +
						    '<table>' +
								'<tr>' +
								'	<td width="60">topic</td>' +
								'	<td><input type="text" name="topic" class="form-control" /></td>' +
								'</tr>' +
								'<tr>' +
								'	<td>name</td>' +
								'	<td><input type="text" name="name" class="form-control"/></td>' +
								'</tr>' +
							'</table>' +
						'</div>' + // .modal-body
					    '<div class="modal-footer">' +
					    	'<button name="save" type="button" class="btn btn-primary">保存</button>' +
					    	'<button name="cancel" type="button" class="btn btn-default" data-dismiss="modal">取消</button>' +
						'</div>' +  // .modal-footer
					'</div>' + // .modal-content
				'</div>' + // .modal-dialog 
			'</div>' // .modal
		);

		var fs = ['id', 'topic', 'name'];
		// 填充旧值
		$.each(fs, function(i, f){
			$("[name=" + f + "]", dlg)[0].value = event[f] || "";
		});
		
		$("[name=save]", dlg).click(function(){
			event = {};
			$.each(fs, function(i, f){
				event[f] = $("[name=" + f + "]", dlg)[0].value;
			});
			$.ajax(basePath + "/event", {
				type: "POST",
				contentType: "application/json",
				data: JSON.stringify(event),
				dataType: "json",
				success: function(d, s){
					if(d){
						dlg.modal("hide");
						(successCallback || new Function())(event);
					}else{
						alert("保存失败！");
					}
				},
				error: function(xhr, ts, e){
					alert("系统错误！\n\n" + e);
				}
			});
		});
		
		$("[name=cancel]", dlg).click(function(){
			dlg.modal("hide");
		});

		dlg.modal("show");
	}
	</script>

	<div class="row" style="margin:20px">
	<p><h3><a onclick="addEvent();">添加新的topic和事件 </a><h3></p>
	<hr />
	#foreach ($entry in $eventMap.entrySet()) 
	    <h3>$entry.getKey() (<a onclick="addEvent('${entry.getKey()}');">&nbsp;添加事件 </a>)</h3> 
		<table class="table table-bordered table-hover table-condensed">
			<thead >
	    		<tr>
	    			<th>topic</th>
	    			<th>name</th>
	    			<th>操作</th>
	    		</tr>
	        </thead>
			<tbody>
			#foreach ($event in $entry.getValue()) 
				<tr>
	        		<td>$event.topic</td>
	        		<td>$event.name</td>
	        		<td>
						<a onclick="editEvent({id: '${event.id}', topic: '${event.topic}', name: '${event.name}'});">编辑 </a> 
						<a onclick="deleteEvent('${event.name}');">删除</a>
					</td>
	        	</tr>
			</tbody>
			#end
		</table>
	#end
	</div>

使用了bootstrap的modal，在一个页面搞定CRUD，cool!

说明：这里的modal是用js拼装而成的，但是其实，bootstrap提供了相应的方法，可以直接把一个预定义的modal拿来使用。具体参见: [modals](http://getbootstrap.com/javascript/#modals)


### 后端Controller


	@Controller
	@RequestMapping("/event")
	public class EventController {

	    private static final Logger logger = Logger.getLogger(EventController.class);

	    @Autowired
	    private EventService        eventService;

	    @RequestMapping(method = RequestMethod.GET)
	    public String listEvent(Model model) {
	        Map<String, List<Event>> events = eventService.listAllGroupByTopic();
	        model.addAttribute("eventMap", events);
	        return "event_list";
	    }

	    @RequestMapping(value = "/{name}", method = RequestMethod.DELETE)
	    @ResponseBody
	    public boolean deleteEvent(@PathVariable("name")
	    String name) {
	        return eventService.removeEvent(StringUtils.trimToEmpty(name));
	    }

	    @RequestMapping(method = RequestMethod.POST)
	    @ResponseBody
	    public boolean saveEvent(@RequestBody
	    Event event) {
	        try {
	            eventService.saveEvent(event);
	        } catch (Exception e) {
	            logger.error("SaveEvent failed!", e);
	            return false;
	        }
	        return true;
	    }
	}

其中eventService.saveEvent(event)方法判断event.id是不是>0，如果大于0表示更新，否则是新增。

So easy, is it?!

-- EOF --

