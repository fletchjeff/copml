# 3: Cross Functional Teams

![](<assets/Cross Function Teams.png>)

The reality is that this kind of workflow requires input from different teams within the business. In other words, it introduces a cross-functional team requirement:

* The **data engineering** team needs to make sure that the data is clean, up to date and available.
* The **data science** team then needs to perform data exploration and model building
* The **machine learning operations** team needs to manage the model post-deployment and make sure that it is always available, operating accurately and continually accessible to the relevant business users.

It is reasonable to expect that different parties within the cross-functional teams might have differing and strongly held opinions about how things should be done. Unless there is careful coordination, it is possible to inadvertently create the conditions for ‘The Great Circle of Blame’.

![](<assets/Great Circle of Blame.png>)

This circle of blame describes a set of behaviours and attitudes that can arise within cross-functional teams wherein blame for things going wrong is attributed to colleagues based in other teams. For example, data scientists might blame their data engineer colleagues for not providing the data in their preferred format or that is not available. Data engineers might blame the data scientists for not adhering to corporate security standards during model development. ML Ops practitioners might express frustration about data scientists not writing tests for their models and data scientists might be exasperated by ML Ops refusal to accept Notebooks as working models. The Great Circle of Blame might look different in different organisations but if left unaddressed it can be a contributory factor for machine learning models failing to make it to production. In addressing this issue, it is important to develop agreed ways of working _and_ to select a platform (and tooling) that is capable of supporting all members of the cross-functional teams to work in the agreed ways.
