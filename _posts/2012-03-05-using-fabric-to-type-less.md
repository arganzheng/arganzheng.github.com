---
layout: post
title: Using Fabric to Type Less
---
> Fabric is a tool written in Python that lets you defind tasks and execute them by running fab taskname. Fabfiles are pure Python, so you can build larger tasks out of smaller ones very easily and do just about anything you want. It’s similar to ant, but without the excessive over-engineering.

    from fabric.api import *
    import os
    import fabric.contrib.project as project

    PROD = 'sjl.webfactional.com'
    DEST_PATH = '/home/sjl/webapps/slc/'
    ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
    DEPLOY_PATH = os.path.join(ROOT_PATH, 'deploy')

    def clean():
        local('rm -rf ./deploy')

    def regen():
        clean()
        local('hyde -g -s .')

    def serve():
        local('hyde -w -s .')

    def reserve():
        regen()
        serve()

    def smush():
        local('smusher ./media/images')

    @hosts(PROD)
    def publish():
        regen()
        project.rsync_project(
                remote_dir=DEST_PATH,
                local_dir=DEPLOY_PATH.rstrip('/') + '/',
                delete=True
        )

> The task I use most often while developing is fab reserve, which regenerates the site and then starts serving it so I can view the result in a browser.

> I use fab smush whenever I add new images. This runs smusher on all of the images to optimize them without reducing quality.

> When I’m ready to publish changes to the live site I run fab publish, which will regenerate my local version and copy it up to the WebFaction server.


### References:
* [Moving from Django to Hyde](http://stevelosh.com/blog/2010/01/moving-from-django-to-hyde/#using-fabric-to-type-less)
