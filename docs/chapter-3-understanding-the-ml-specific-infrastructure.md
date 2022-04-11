# 3: Machine Learning Infrastructure

This section highlights which aspects of your infrastructure, platform and tooling are required in order to support the cross-functional teams involved in delivering the machine learning workflow described above. The details provided here are specific to the Cloudera stack but the principles are not. These ideas can be applied to other platforms or a bespoke build comprising various components from the Open Source Software (OSS) world. It’s worth noting that this latter case presents its own set of issues.

## Data Access

![](<assets/Data Access.png>)

The starting point for all machine learning projects, after the business requirement, is data. The first thing to consider is what data is already being stored by the organisation and whether it is applicable to the project. If the available data fits the requirements, then the next consideration is how accessible it is to the people and processes that need it. However, if there are gaps in the data, then it might be necessary to acquire additional data. This can come from internal data sources which aren’t collected or stored in an accessible manner. This data could also come from an external third party provider. However the data is acquired, it will all need to be accessible to the people and processes that need it within the overall machine learning workflow.

Another point of consideration for the organisation is whether there is value in storing the products of intermediate data processing, such as features (representations of data that can be ingested by models), within a feature store.

### Feature Store

[https://towardsdatascience.com/do-you-really-need-a-feature-store-e59e3cc666d3](https://towardsdatascience.com/do-you-really-need-a-feature-store-e59e3cc666d3#:\~:text=The%20canonical%20use%20of%20a%20feature%20store\&text=are%20used%20in%20multiple%20models,always%20need%20a%20feature%20store.)

At present, there is no consensus on the term feature store. In its simplest form it can be considered a store for curated features so they can be reused. In its more complex manifestation, a feature store might go beyond facilitating access to features and also seek to deliver high availability and low latency in production as well as more [complex requirements](https://eugeneyan.com/writing/feature-stores/).

In machine learning, model training requires the inputs for the model training process to be represented as numeric values. As an example, categorical variables like red, green, and blue must be transformed via a process like [one hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/). Relatedly, certain machine learning algorithms work better when numeric inputs are scaled or normalised. Scaling is the process of taking a column of numbers that ranges from, for example, 1 to 1000 and creating a new column that ranges from 0 to 1 (often referred to as normalising) or -1 to 1 (often referred to as standardising). This represents the final stage of the data preparation before it can be used in the machine learning model training process. This feature data is not necessarily useful to other users of the data platform outside of machine learning model training though.

In some machine learning implementations feature creation is incorporated into the model training process using a ‘pipeline’ i.e. an automated sequence of steps. This pipeline will create the features needed for the model training process. These features are ephemeral in nature – they are discarded once the model training process is done. However, some workflows allow these intermediate features to be more permanent by storing them in a feature store. There can be benefits to doing this. For example it can help save on the compute resources required or speed up the process of hyperparameter optimization by removing the need to recalculate the extracted features for each model training run. Feature stores can also help with reproducibility. We’ll cover this in a bit more detail later in this white paper.

From a data engineering perspective, Data Access is primarily focused on making data accessible by the data science and ML Ops teams. The data needs to be clean and as up to date as is necessary to meet the business requirement. Within the Cloudera stack this would involve making the data available via CDSW/CML. This could be in the form of files in the HDFS or the cloud object store, a Hive database in a DataHub, a virtual Data Warehouse in a Cloudera Data Warehouse (CDW) experience or even in HBase within the Cloudera Operational Database (COD) experience. In a DataHub the options include Druid or Kudu as the underlying storage.

The decision about what to use depends on the particular machine learning project and the data used for it. It can be tabular data, or document-like key-value data, or binary data like images and audio files. Keeping data within the stack minimises the complexity of security and governance of moving data between disparate systems.

In CDP the follow storage systems are available and are best suited to certain data structures:

| **Data storage system**              | **Data structures**                    |
| ------------------------------------ | -------------------------------------- |
| COD (HBase)                          | key-value data (i.e. document storage) |
| CDW (Hive / Impala / Hive LLAP)      | tabular data                           |
| Data Hub (HDFS / Cloud Object Store) | unstructured data (images / audio)     |

Putting extracted features into a feature store raises the question of responsibility. There are no hard and fast rules about this. Our advice is that the relevant teams - data engineering, data science or model operations- decide how responsibility is allocated based on relevant skills and knowledge. Within the Cloudera stack, the requirements for the feature store can also be met using either COD (with Apache Phoenix) or CDW for tabular data or COD for unstructured, key-value data. Having said that, the one place that makes the most sense is HBase as it supports a wide range of dataset types with low latency lookup and also has better snapshot capabilities for reproducibility.

Regardless of which CDP experience is used to make the data available to the data science team, CDE is configured and optimized to do this at scale and in an automated way that should eliminate friction between all involved parties.

## Data Exploration

![](<assets/Data Exploration.png>)

Exploratory data analysis (EDA) is a fairly well understood function within the data science community. CDSW/CML supports EDA via its native workbench as well as third party notebook solutions such as Jupyter Lab and R Studio.

![](assets/10.png)

It is at this stage that the data science team will look at the ‘shape’ of the data and come up with a plan for building a suitable model. EDA is often combined with the Model Building stage, however this doesn’t mean a single team necessarily manages the entire process. It may be that there are good organizational reasons to split this activity, for example to accommodate areas of specialization. Some personnel may be more inclined towards statistical data analysis and so focus on EDA while others might specialise in model optimization and concentrate on that part of the model development process.

What is most important is that at this stage the data science team should have easy access to all the relevant data and have the flexibility to augment that data (if necessary) in order to best serve the business requirement.

### Source Control

One of the main parts of the artifacts created during the various stages of deploying a machine learning project is the actual code written by the data scientists and machine learning engineers. Creating code usually starts once Data Exploration starts, but there may be some code that is used to facilitate or create data that is needed for data access. These code artifacts need to be persisted and stored and have the ability to keep track of a history of changes. This is required for the auditability and regulatory requirements discussed later, but also for the developers to be able to collaborate effectively and roll back if any changes are made that cause issues with the project. This is where source control comes into play. Like any other form of software development, not all changes to code are significant, but anything that is considered significant or just as a way to store the current state of the code and other software artifacts, source control tools are invaluable.

Source control systems are a solved problem, with many feature rich, tried and tested tools available. CML/CDSW seamlessly integrates with any git compliant source control system: Github, Gitlab, Bitbucket etc. Some source control systems have workflow automation tools built in which can integrate with the CML/CDSW APIs, such as the Jobs API to further enhance workflow automation.

The main thing to keep in mind is that source control is used to store the code. It can be used to store other artifacts, like saved model files or data sets, but it's not designed for that. Source control systems should be used as part of the overall platform to manage version control for software code.

## Model Building

![](<assets/Model Building.png>)

While model building is often done in conjunction with the previous step, bigger teams might support some degree of specialisations of roles. Some team members will focus on implementing the best approach to addressing the business use case while others might concentrate on model selection and optimization. For the sake of clarity, let's confirm what is meant by **model** here. The code used to create the model will likely include a current set of optimizations and hyper parameters specific to the model. This code and any iterations used to create a new version of the model that contains changes to the way the model be trained should be checked into the source control system.

### What is a model?

The term model has many uses within different contexts and its meaning can change even within the data science context. For the purpose of this document a model is:

_“A combination of a data transformer, an algorithm and configuration details that can be used to make a new prediction based on new data.”_

Regardless of whether this is implemented as a single step, a couple of steps that involve getting the transformed data to and from a feature store, or an activity within a pipeline, the definition still holds. The final form of the model is an artifact.

### Model Formats

The final trained model artifact is a file that can be loaded at a later date and be used to make new predictions. This file can exist in a variety of formats including:

* **Pickle**: a standard python serialisation library used to save models from scikit-learn
* **Spark MLWritable**: the standard model storage format included with Spark, but limited to use only within Spark
* **PMML**: (Predictive Model Markup Language) a standardised language used to represent predictive analytic models in a portable text format
* **ONNX**: (Open Neural Network Exchange) provides a portable model format for deep learning, using Google Protocol Buffers for the schema definition

### Model Optimisation

The initial model building process incorporates some work towards improving model performance through hyperparameter optimization and some exploration of different approaches to feature creation. The [Experiments feature](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-running-an-experiment.html) within CML/CDSW supports this model optimization work. Alternatively, this can be done using some of the available model optimization OSS tools such as TPOT, auto-sklearn, MLFlow etc. Apache Spark does not have any built-in tools for model optimization, so the CML/CDSW Experiments feature would be a good option in this case.

AutoML and other automated hyperparameter optimization tools are useful but should be considered in light of available compute resources and machine learning skills. A data scientist with enough experience and domain expertise will probably know what to do to get to an optimized model more quickly and cheaply than an AutoML tool would. For many use cases an AutoML tool is no replacement for an experienced data scientist, who will be able to develop a more effective and interpretable model. Consequently, it should not necessarily be included as part of the process for all new models if it doesn’t need to be.

The infrastructure needed for doing model training is another point of consideration. The next section explores this further but it’s worth noting even during the early stages of model building, model optimization will require several iterations of model training. The type of model, the size of the data and available compute resources will dictate what kind of infrastructure is most suitable for running the model training process. For example, training a big neural network on a very large dataset will require multiple GPU nodes to complete in a reasonable amount of time, but this is expensive infrastructure. In this situation, training needs would be better fulfilled using CML in the public cloud to take advantage of its ability to autoscale the number of nodes required to meet the compute demand. If this were a smaller model training process or public cloud was not available, then using an existing on-premise CDP Base cluster with Spark on Yarn would make more sense.

Once the initial model build is complete and there is a working process for training the model, this needs to be automated so that the model can be periodically re-trained without direct input from the data science team.

## Model Training

![](<assets/Model Training.png>)

As part of the ongoing production machine learning workflow, the machine learning models will need to be periodically retrained. This may be because new data becomes available, the data landscape changes or a newer version of the algorithm is released that optimizes some part of the models’ performance. Having the right infrastructure becomes more important here as this often needs to happen without direct involvement from the data science teams. The machine learning operations team needs a way to run a model training process directly on the available data. This may include a pipeline that does feature creation as part of the model training, or separate processes that put the features into a feature store first. Where the model training jobs sit will depend on the use case.

These model training jobs need to be automated in a way that they can be triggered through the monitoring system (discussed later in the document). The requirement is that model training jobs are schedulable and triggerable through an API call from a process running as part of the ongoing model monitoring system. This can be done entirely within CML/CDSW using the [Jobs](https://docs.cloudera.com/machine-learning/1.1/jobs-pipelines/topics/ml-creating-a-job.html) feature, or if this is a larger Spark based job it can be done in CDE. The public cloud implementation of CML uses Kubernetes to scale up and down the resources that are available for model training. This allows it to support model training jobs that need a lot of capacity for a short period of time.

A hybrid cloud approach would probably provide the optimal price to performance ratio for organisations that need to manage model training for a variety of models with a range of infrastructure requirements. Having said that, there are still complexities around data locality and its impact on efficient data access that need to be understood and managed. The economics of hybrid cloud machine learning is discussed in further detail [here](https://drive.google.com/file/d/1etdhutUFEjNOsrJPqbLIavRJD3R9nhjr/view?usp=sharing).

## Deploy and Serve

![](<assets/Deploy Serve.png>)

The trained model now needs to go into **production** i.e. the model’s output needs to be ‘served’ to or integrated into downstream applications. Depending on the availability requirements discussed later in the document, some organisations run completely separate environments to differentiate between development and production systems. Once a model has been developed and tested on the development system, it is moved on to the production system for deployment. This allows for different architectures and configurations between the two platform types to meet differing compute and availability requirements. Within CML/CDSW, this development-to-production capability can be facilitated in a single workspace by using the Teams [feature](https://docs.cloudera.com/machine-learning/cloud/user-accounts/topics/ml-creating-a-team.html). Alternatively CML/CDSW can be deployed with completely separate workspaces in the same or different environments (for CML) or different hardware (for CDSW) and the projects can be moved using the underlying source control system and redeployed. For more automation, the projects can use the [CML v2 API](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html) or [AMP specification](https://docs.cloudera.com/machine-learning/cloud/applied-ml-prototypes/topics/ml-amp-project-spec.html) which will allow for automating model and job deployments as well.

The two main modes of operation for production machine learning models are **batch** and **near realtime**. There are other modes of operation for machine learning models, but they are far less common.

### Batch Models

A batch process runs periodically and will make several inferences at a time. It might be a process that runs batch inference on tabular data and adds or updates a column with predictions made by the model. These kinds of models are supported in CML/CDSW using the Jobs feature to periodically trigger a script that can fetch new data from and update one of the datastores within CDP (e.g. COD or CDW) with a new prediction. As illustrated in the churn projects, the model metrics are updated during the batch process, but if model metric tracking is not required, it’s possible to just update the table on each new batch run.

### Realtime Models

Realtime models’ output are usually made available via APIs that can be called by downstream relevant applications in order to make new predictions on an ad hoc basis. This is the default mode for the [Models feature](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html) within CML/CDSW. The model artifact (i.e. the model file) is loaded as a Python function which is then wrapped in a process that creates an API endpoint which can accept JSON as input data, run the model code to create a prediction and then return the prediction as JSON via that same API endpoint. This process is usually referred to as model serving.

Model deployment and serving infrastructure have specific availability and reproducibility requirements. These are detailed in the section on continuous operations for production machine learning.

### Model Registry

A Model Registry is a mechanism that tracks and manages models, model artifacts and related to metadata for any deployed model type. In CML/CDSW, there are 2 options for a model registry. There is the [model governance](https://docs.cloudera.com/machine-learning/cloud/model-governance/topics/ml-enabling-model-governance.html) feature, which registers the various stages of model deployment with CDP’s Atlas:

![](assets/14.png)

Alternatively it's possible to use MLFlow within a CML/CDSW project, but this will be local to the project and not registered globally like it would be with Atlas.

![](assets/15.png)

## Monitoring Systems

![](assets/Monitor.png)

The final stage in this machine learning workflow supports the monitoring of the model. This aim is to ensure that the model output is of a consistent performance level and reliably available to the downstream applications that need to consume it, as per the business requirement. The monitoring-related tasks are automated and delivered with the aid of optimized tooling and infrastructure. For example, within CML/CDSW it's possible to check and monitor the real-time model to ensure it's up and working.

For the metric tracking requirements there is an [SDK](https://docs.cloudera.com/cdsw/1.9.1/models/topics/cdsw-tracking-model-metrics.html) that allows the model operations team to record any number of relevant metrics each time the model is called. These metrics can be updated to include real-world data for supervised learning models and the periodic addition of aggregate metrics to track any required statistical performance metrics over time.

As will be discussed later, it’s important to know how the model is performing against both the statistical and business requirements. When monitoring a model, many implementations will focus on the statistical metrics and how these vary over time. However, this is insufficient. Many projects fail because the business requirements they were designed to deliver aren’t being met . Take for example the churn example accompanying this report, the fact that the model is 95% accurate at predicting churn is not useful if this doesn’t translate to measurable reduction in actual churn rate. It’s achieving the latter –the business requirement – that actually delivers value to the business.

All of the above requirements can be implemented using components from CML/CDSW and other experiences within the CDP stack. It is possible to build an implementation using open source components provided the data security and governance requirements can be fulfilled too. While these two criteria don’t have a direct impact on model performance, they are important concerns for an IT department; it will also result in fewer integration issues down the line. An end-to-end ML solution that lacks robust data security and governance is unlikely to be accepted by an enterprise IT department.
