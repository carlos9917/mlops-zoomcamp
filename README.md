# MLOps Zoomcamp

<p align="center">
  <a href="https://datatalks.club" target="_blank">
    <img src="images/banner.png">
  </a>
</p>


Our MLOps Zoomcamp course

- Sign up here: https://airtable.com/shrCb8y6eTbPKwSTL (it's not automated, you will not receive an email immediately after filling in the form)
- Register in [DataTalks.Club's Slack](https://datatalks.club/slack.html)
- Join the [`#course-mlops-zoomcamp`](https://app.slack.com/client/T01ATQK62F8/C02R98X7DS9) channel
- [Tweet about the course!](https://ctt.ac/fH67W)
- Subscribe to the [public Google calendar](https://calendar.google.com/calendar/?cid=M3Jzbmg0ZDA2aHVsY2M1ZjcyNDJtODNyMTRAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ) (subscription works from desktop only)
- [Youtube Playlist with all the videos](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK)
- [Technical FAQ](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit) 


## Overview

### Objective

Teach practical aspects of productionizing ML services — from collecting requirements to model deployment and monitoring.

### Target audience

Data scientists and ML engineers. Also software engineers and data engineers interested in learning about putting ML in production.

### Pre-requisites

* Python
* Docker
* Being comfortable with command line 
* Prior exposure to machine learning (at work or from other courses, e.g. from [ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp))
* Prior programming experience (at least 1+ year)

### Timeline

Course start: 16 of May


## Syllabus

This is a draft and will change. 


### [Module 1: Introduction](01-intro)

* What is MLOps
* MLOps maturity model
* Running example: NY Taxi trips dataset
* Why do we need MLOps
* Course overview
* Environment preparation
* Homework

[More details](01-intro)

### Module 2: Experiment tracking

* Experiment tracking intro
* Getting started with MLflow
* Experiment tracking with MLflow
* Saving and loading models with MLflow
* Model registry
* MLflow in practice
* Homework


### Module 3: Orchestration and ML Pipelines

* ML Pipelines: introduction
* Prefect
* Turning a notebook into a pipeline
* Kubeflow Pipelines
* Homework 


### Module 4: Model Deployment 

* Batch vs online
* For online: web services vs streaming
* Serving models in Batch mode
* Web services
* Streaming (Kinesis/SQS + AWS Lambda)
* Homework


### Module 5: Model Monitoring

* ML monitoring vs software monitoring 
* Data quality monitoring
* Data drift / concept drift 
* Batch vs real-time monitoring 
* Tools: Evidently, Prometheus and Grafana
* Homework 


### Module 6: Best Practices

* Devops
* Virtual environments and Docker
* Python: logging, linting
* Testing: unit, integration, regression 
* CI/CD (github actions)
* Infrastructure as code (terraform, cloudformation)
* Cookiecutter
* Makefiles
* Homework


### Module 7: Processes

* CRISP-DM, CRISP-ML
* ML Canvas
* Data Landscape canvas
* [MLOps Stack Canvas](https://miro.com/miroverse/mlops-stack-canvas/)
* Documentation practices in ML projects (Model Cards Toolkit)


### Project

* End-to-end project with all the things above


## Running example

To make it easier to connect different modules together, we’d like to use the same running example throughout the course.

Possible candidates: 

* [https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) - predict the ride duration or if the driver is going to be tipped or not


## Instructors

- Larysa Visengeriyeva
- Cristian Martinez
- Kevin Kho
- Theofilos Papapanagiotou 
- Alexey Grigorev
- Emeli Dral
- Sejal Vaidya


## Other courses from DataTalks.Club:

- [Machine Learning Zoomcamp - free 4-month course about ML Engineering](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)
- [Data Engineering Zoomcamp - free 9-week course about Data Engineering](https://github.com/DataTalksClub/data-engineering-zoomcamp/)


## FAQ

**I want to start preparing for the course. What can I do?**

If you haven't used Flask or Docker

* Check [Module 5](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment) form ML Zoomcamp
* The [section about Docker](https://github.com/DataTalksClub/data-engineering-zoomcamp/tree/main/week_1_basics_n_setup/2_docker_sql) from Data Engineering Zoomcamp could also be useful

If you have no previous experience with ML

* Check [Module 1](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/01-intro) from ML Zoomcamp for an overview
* [Module 3](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/03-classification) will also be helpful if you want to learn Scikit-Learn (we'll use it in this course)
* We'll also use XGBoost. You don't have to know it well, but if you want to learn more about it, refer to [module 6](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/06-trees) of ML Zoomcamp


**I registered but haven't received an invite link. Is it normal?**

Yes, we haven't automated it. You'll get a mail from us eventually, don't worry.

If you want to make sure you don't miss anything:

* Register in [our Slack](https://datatalks.club/slack.html) and join the `#course-mlops-zoomcamp` channel
* Subscribe to [our YouTube channel](https://youtube.com/c/datatalksclub)

**Is it going to be live?**

No and yes. There will be two parts:

* Lectures: Pre-recorded, you can watch them when it's convenient for you. 
* Office hours: Live on Mondays (17:00 CET), but recorded, so you can watch later.


## Partners

Thanks to our friends for spreading the word about the course

<p align="center">
  <a href="https://dphi.tech/">
    <img height="75" src="https://datatalks.club/images/partners/dphi.png">
  </a>
  <a href="https://www.confetti.ai/">
    <img height="75" src="https://datatalks.club/images/partners/confetti.png">
  </a>
</p>

# Data Engineering Zoomcamp

- Register in [DataTalks.Club's Slack](https://datatalks.club/slack.html)
- Join the [`#course-data-engineering`](https://app.slack.com/client/T01ATQK62F8/C01FABYF2RG) channel
- The videos are published to [DataTalks.Club's YouTube channel](https://www.youtube.com/c/DataTalksClub) in [the course playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb) 
- [Frequenty asked technical questions](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit?usp=sharing)


Syllabus

* [Week 1: Introduction & Prerequisites](#week-1-introduction--prerequisites)
* [Week 2: Data ingestion](#week-2-data-ingestion)
* [Week 3: Data Warehouse](#week-3-data-warehouse)
* [Week 4: Analytics Engineering](#week-4-analytics-engineering)
* [Week 5: Batch processing](#week-5-batch-processing)
* [Week 6: Streaming](#week-6-streaming)
* [Week 7, 8 & 9: Project](#week-7-8--9-project)

## Taking the course

### Self-paced mode

All the materials of the course are freely available, so you 
can take the course at your own pace 

* Follow the suggested syllabus (see below) week by week
* You don't need to fill in the registration form. Just start watching the videos and join Slack 
* Check [FAQ](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit?usp=sharing) if you have problems
* If you can't find a solution to your problem in FAQ, ask for help in Slack


### 2022 Cohort

* **Start**: 17 January 2022
* **Registration link**: https://airtable.com/shr6oVXeQvSI5HuWD
* [Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vR9oQiYnAVvzL4dagnhvp0sngqagF0AceD0FGjhS-dnzMTBzNQIal3-hOgkTibVQvfuqbQ69b0fvRnf/pubhtml)
* Subscribe to our [public Google Calendar](https://calendar.google.com/calendar/?cid=ZXIxcjA1M3ZlYjJpcXU0dTFmaG02MzVxMG9AZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ) (it works from Desktop only)


### Asking for help in Slack

The best way to get support is to use [DataTalks.Club's Slack](https://datatalks.club/slack.html). Join the [`#course-data-engineering`](https://app.slack.com/client/T01ATQK62F8/C01FABYF2RG) channel.

To make discussions in Slack more organized:

* Follow [these recommendations](asking-questions.md) when asking for help
* Read the [DataTalks.Club community guidelines](https://datatalks.club/slack/guidelines.html)


## Syllabus


### [Week 1: Introduction & Prerequisites](week_1_basics_n_setup)

* Course overview
* Introduction to GCP
* Docker and docker-compose 
* Running Postgres locally with Docker
* Setting up infrastructure on GCP with Terraform
* Preparing the environment for the course
* Homework

[More details](week_1_basics_n_setup)


### [Week 2: Data ingestion](week_2_data_ingestion)

* Data Lake
* Workflow orchestration
* Setting up Airflow locally
* Ingesting data to GCP with Airflow
* Ingesting data to local Postgres with Airflow
* Moving data from AWS to GCP (Transfer service)
* Homework

[More details](week_2_data_ingestion)


### [Week 3: Data Warehouse](week_3_data_warehouse)


* Data Warehouse
* BigQuery
* Partitoning and clustering
* BigQuery best practices
* Internals of BigQuery
* Integrating BigQuery with Airflow
* BigQuery Machine Learning

[More details](week_3_data_warehouse)


### [Week 4: Analytics engineering](week_4_analytics_engineering/)

* Basics of analytics engineering
* dbt (data build tool)
* BigQuery and dbt
* Postgres and dbt
* dbt models
* Testing and documenting
* Deployment to the cloud and locally
* Visualising the data with google data studio and metabase 


[More details](week_4_analytics_engineering)


### [Week 5: Batch processing](week_5_batch_processing)

* Batch processing 
* What is Spark
* Spark Dataframes
* Spark SQL
* Internals: GroupBy and joins

[More details](week_5_batch_processing)

### [Week 6: Streaming](week_6_stream_processing)

* Introduction to Kafka
* Schemas (avro)
* Kafka Streams
* Kafka Connect and KSQL

[More details](week_6_stream_processing)


### [Week 7, 8 & 9: Project](week_7_project)

Putting everything we learned to practice

* Week 7 and 8: working on your own project
* Week 9: reviewing your peers

[More details](week_7_project)


## Overview

### Architecture diagram
<img src="images/architecture/arch_1.jpg"/>

### Technologies
* *Google Cloud Platform (GCP)*: Cloud-based auto-scaling platform by Google
  * *Google Cloud Storage (GCS)*: Data Lake
  * *BigQuery*: Data Warehouse
* *Terraform*: Infrastructure-as-Code (IaC)
* *Docker*: Containerization
* *SQL*: Data Analysis & Exploration
* *Airflow*: Pipeline Orchestration
* *dbt*: Data Transformation
* *Spark*: Distributed Processing
* *Kafka*: Streaming


### Prerequisites

To get most out of this course, you should feel comfortable with coding and command line,
and know the basics of SQL. Prior experience with Python will be helpful, but you can pick 
Python relatively fast if you have experience with other programming languages.

Prior experience with data engineering is not required.



## Instructors

- Ankush Khanna (https://linkedin.com/in/ankushkhanna2)
- Sejal Vaidya (https://linkedin.com/in/vaidyasejal)
- Victoria Perez Mola (https://www.linkedin.com/in/victoriaperezmola/)
- Alexey Grigorev (https://linkedin.com/in/agrigorev)

## Tools 

For this course you'll need to have the following software installed on your computer:

* Docker and Docker-Compose
* Python 3 (e.g. via [Anaconda](https://www.anaconda.com/products/individual))
* Google Cloud SDK 
* Terraform

See [Week 1](week_1_basics_n_setup) for more details about installing these tools



## FAQ


* **Q**: I registered, but haven't received a confirmation email. Is it normal?
  **A**: Yes, it's normal. It's not automated. But you will receive an email eventually 
* **Q**: At what time of the day will it happen?
  **A**: Office hours will happen on Mondays at 17:00 CET. But everything will be recorded, so you can watch it whenever it's convenient for you
* **Q**: Will there be a certificate?
  **A**: Yes, if you complete the project
* **Q**: I'm 100% not sure I'll be able to attend. Can I still sign up?
  **A**: Yes, please do! You'll receive all the updates and then you can watch the course at your own pace. 
* **Q**: Do you plan to run a ML engineering course as well?
**A**: Glad you asked. [We do](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) :)
* **Q**: I'm stuck! I've got a technical question!
  **A**: Ask on Slack! And check out the [student FAQ](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit?usp=sharing); many common issues have been answered already. If your issue is solved, please add how you solved it to the document. Thanks!



## Our friends 

Big thanks to other communities for helping us spread the word about the course: 

* [DPhi](https://dphi.tech/)
* [MLOps.community](https://mlops.community/)

Check them out - they are cool!
