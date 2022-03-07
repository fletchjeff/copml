---
cover: >-
  https://images.unsplash.com/photo-1620712943543-bcc4688e7485?crop=entropy&cs=srgb&fm=jpg&ixid=MnwxOTcwMjR8MHwxfHNlYXJjaHw2fHxtYWNoaW5lJTIwbGVhcm5pbmd8ZW58MHx8fHwxNjQ2NjUxODM4&ixlib=rb-1.2.1&q=85
coverY: 479.2131979695431
---

# Introduction

_This post is adapted from a document written by myself and_ [_Ade Adewumni_](https://www.linkedin.com/in/adeadewunmi/) _and was_ [_originally publish_](https://www.cloudera.com/content/dam/www/marketing/resources/ebooks/blueprint-for-continuous-operations-for-production-machine-learning.pdf.landing.html) _on the_ [_Cloudera_](https://cloudera.com) _website._

More organisations are adopting machine learning (ML) and the market has matured enough to be able to offer more options and capabilities. However, in many cases the business value derived from their investment in ML has been inconsistent. This isn’t because enterprise customers have been careless in their approach to machine learning. Many were careful to conduct ‘science experiments’ to identify which of their business challenges machine learning could be successfully applied to. The problem is that success in developing these proofs of concept (PoCs) does not necessarily translate into the successful ‘operationalization’ or deployment of machine learning. Further, this type of experimentation is not cheap. Outside of the so-called Big Tech firms\[^1] few organisations can sustain extensive experimentation without clear evidence that it will contribute to the bottom line. If none or very few of the successful PoCs make it to deployment, it’s difficult for organizations to justify ongoing investment. This is true even if it’s been demonstrated, through PoCs, that machine learning can be successfully applied to the underlying business challenge.

Enterprise customers are realising that in order to see a return on their investment in machine learning, these successful ML experiments need a formalised path to deployment and a structured approach to post-deployment maintenance. They are also aware of the importance of doing this in a way that minimizes costs and friction which make it harder to sustain the value of these ML applications in the medium to long term. This is where the methodology, Continuous Operations for Production Machine Learning (COPML) can help. COPML is a comprehensive approach to model development and operation that facilitates the creation of machine learning applications which deliver continuous business value for as long as possible and can be smoothly retired once this ceases to be the case.

The scope of COPML isn’t limited to model artifacts or the tools and systems that support their ongoing maintenance and ensure that they are smoothly served to downstream applications. It also supports the fostering of collaboration between the various teams involved in the machine learning workflow. This document provides an exploration of the COPML methodology, using two example use cases to further illustrate what its application could look like in practice.

\[^1]: Big Tech is used to collectively describe the most popular and best-performing American technology companies e.g. Facebook, Amazon, Microsoft, Apple, Netflix and Alphabet etc.
