+++
title =  "Deploying AI Models"
date = 2019-07-17T22:40:19-05:00
tags = []
featured_image = ""
description = ""
+++

Here's a discussion that we had about how to deploy models to be used in an AI application.

The thought process follows a pattern of providing a web interface of some sort with the requests being queued up and handled asynchrounously by a server running an AI model.

<!--more-->

## Agenda

* Welcome & Introduction
* Project Updates
* Deploying AI Models
* Plan for next week

## Deployment Use Case

Let's start by thinking about this use case from the [Bug Analysis Project](https://github.com/HSV-AI/bug-analysis)

After we train some sort of model, how do we make it available for the masses?

Think about something like this:

![Image](https://miro.medium.com/max/875/1*BmQ3UFuQXptZc2yhqAybLA.png)

How would we build this?

What about updating the model after you deploy it?


## References

[Train and serve a TensorFlow model with TensorFlow Serving](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)

[fastai serving](https://github.com/developmentseed/fastai-serving)

[There are two very different ways to deploy ML models, here’s both](https://towardsdatascience.com/there-are-two-very-different-ways-to-deploy-ml-models-heres-both-ce2e97c7b9b1)
