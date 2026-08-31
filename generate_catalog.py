"""
Generate the synthetic learning catalog at data/catalog.csv.

Run once:  python generate_catalog.py

The catalog holds three kinds of learning resource, distinguished by the
``resource_type`` column. The brief asks for a roadmap of "courses, projects and
assessments", so all three are first-class entities in one table rather than
courses only:

    course      Teaches skills. ``skills`` lists what the learner acquires.
    project     Applies skills already taught. ``skills`` lists what it
                exercises; ``prerequisites`` are the courses that teach them.
    assessment  Validates a skill cluster. Short, and gated behind the courses
                that cover the material.

Why one table instead of three files
------------------------------------
A single table means one embedding index and one prerequisite graph. The path
generator can then interleave all three types by the same topological rules,
which is what lets a milestone be "finish these courses, build this project, pass
this assessment" rather than a hardcoded string.

Note on the identifier column
-----------------------------
The primary key column is still named ``course_id`` even though it now addresses
projects and assessments too. Renaming it would ripple through ``progress.course_id``
and ``feedback.course_id`` in the database plus every module, for no functional
gain. ID prefixes carry the type (DS/WD/UX/BM/CD = course, PR = project,
AS = assessment) and ``resource_type`` is authoritative.
"""

import csv
import os

# Nominal effort per course, by difficulty. Projects and assessments carry their
# own explicit durations because effort varies far more between them.
COURSE_HOURS = {"beginner": 6, "intermediate": 15, "advanced": 30}

COURSES = [
    # ── DATA SCIENCE (beginner) ──────────────────────────────────────────────
    ("DS001", "Introduction to Python for Data Science", "beginner", "Data Science",
     "Learn the fundamentals of Python programming with a focus on data science workflows. "
     "Covers variables, loops, functions, and basic data structures. "
     "By the end you will write scripts to load and inspect datasets.",
     "Python,data types,loops,functions,pandas basics",
     ""),
    ("DS002", "Statistics for Data Science", "beginner", "Data Science",
     "A practical introduction to descriptive and inferential statistics essential for data analysis. "
     "Topics include mean, median, variance, probability distributions, and hypothesis testing. "
     "All examples are implemented in Python using NumPy and SciPy.",
     "statistics,probability,hypothesis testing,NumPy,SciPy",
     ""),
    ("DS003", "Data Cleaning and Preprocessing with Pandas", "beginner", "Data Science",
     "Master the art of wrangling messy real-world datasets using the Pandas library. "
     "Covers missing value handling, outlier detection, encoding, and feature scaling. "
     "You will clean three realistic datasets as hands-on projects.",
     "pandas,data cleaning,feature engineering,missing values",
     "DS001"),
    ("DS004", "Data Visualization with Matplotlib and Seaborn", "beginner", "Data Science",
     "Discover how to communicate insights through compelling charts and plots. "
     "Covers line charts, bar plots, histograms, heatmaps, and interactive visualizations. "
     "Projects include exploratory data analysis on real datasets.",
     "matplotlib,seaborn,data visualization,EDA",
     "DS001"),
    ("DS005", "SQL for Data Analysts", "beginner", "Data Science",
     "Learn SQL from the ground up to query relational databases like a professional. "
     "Covers SELECT, JOIN, GROUP BY, subqueries, window functions, and query optimization. "
     "Practice on a realistic sales database included in the course.",
     "SQL,databases,querying,joins,aggregations",
     ""),
    ("DS006", "Excel for Data Analysis", "beginner", "Data Science",
     "Build data analysis skills using Microsoft Excel without writing a single line of code. "
     "Covers pivot tables, VLOOKUP, conditional formatting, and charting best practices. "
     "Ideal for analysts transitioning from spreadsheets to code-based tools.",
     "Excel,pivot tables,VLOOKUP,charting",
     ""),
    # DATA SCIENCE (intermediate)
    ("DS007", "Machine Learning with Scikit-Learn", "intermediate", "Data Science",
     "Implement and tune supervised and unsupervised machine learning algorithms using scikit-learn. "
     "Covers regression, classification, clustering, model evaluation, and cross-validation. "
     "End-to-end project: predict customer churn.",
     "machine learning,scikit-learn,regression,classification,model evaluation",
     "DS001,DS002,DS003"),
    ("DS008", "Applied Natural Language Processing", "intermediate", "Data Science",
     "Build NLP pipelines for text classification, sentiment analysis, and named entity recognition. "
     "Uses spaCy, NLTK, and HuggingFace transformers. "
     "Project: classify customer support tickets automatically.",
     "NLP,text classification,transformers,spaCy,sentiment analysis",
     "DS007"),
    ("DS009", "Feature Engineering and Selection", "intermediate", "Data Science",
     "Learn systematic techniques for creating and selecting the features that most improve model performance. "
     "Covers polynomial features, target encoding, mutual information, and SHAP-based selection. "
     "Applied to tabular and time-series data.",
     "feature engineering,feature selection,SHAP,target encoding",
     "DS007"),
    ("DS010", "Time Series Analysis and Forecasting", "intermediate", "Data Science",
     "Analyse temporal data and build forecasting models using ARIMA, Prophet, and LSTM networks. "
     "Covers stationarity, seasonality decomposition, and evaluation metrics for time-series. "
     "Project: forecast retail sales for the next 30 days.",
     "time series,ARIMA,Prophet,LSTM,forecasting",
     "DS007"),
    ("DS011", "Data Engineering with Python and Airflow", "intermediate", "Data Science",
     "Design and automate ETL pipelines using Apache Airflow and Python operators. "
     "Covers DAGs, task dependencies, scheduling, and monitoring pipeline health. "
     "Integrates with PostgreSQL and cloud object storage.",
     "ETL,Airflow,data pipelines,DAGs,PostgreSQL",
     "DS001,DS005"),
    ("DS012", "A/B Testing and Experimentation", "intermediate", "Data Science",
     "Run rigorous A/B tests to evaluate product changes with statistical confidence. "
     "Covers experimental design, sample-size calculation, p-values, and Bayesian alternatives. "
     "Case studies from e-commerce and SaaS product teams.",
     "A/B testing,statistics,experimentation,Bayesian,p-value",
     "DS002"),
    ("DS013", "Recommender Systems", "intermediate", "Data Science",
     "Build collaborative filtering and content-based recommendation engines from scratch. "
     "Covers matrix factorization, cosine similarity, and hybrid approaches. "
     "Project: movie recommendation system using the MovieLens dataset.",
     "recommendation systems,collaborative filtering,matrix factorization,cosine similarity",
     "DS007"),
    # DATA SCIENCE (advanced)
    ("DS014", "Deep Learning with PyTorch", "advanced", "Data Science",
     "Design and train deep neural networks for image, text, and tabular data using PyTorch. "
     "Covers CNNs, RNNs, attention mechanisms, and transfer learning. "
     "Capstone: fine-tune a BERT model for domain-specific sentiment analysis.",
     "deep learning,PyTorch,CNN,RNN,transformers,transfer learning",
     "DS007"),
    ("DS015", "MLOps: Deploying Models to Production", "advanced", "Data Science",
     "Learn best practices for packaging, deploying, monitoring, and retraining ML models at scale. "
     "Covers Docker, FastAPI serving, MLflow experiment tracking, and drift detection. "
     "End-to-end project: deploy a fraud-detection model to a cloud endpoint.",
     "MLOps,Docker,FastAPI,MLflow,model deployment,drift detection",
     "DS007,DS011"),
    ("DS016", "Large Language Models: Prompting and Fine-Tuning", "advanced", "Data Science",
     "Understand the architecture of transformer-based LLMs and learn to leverage them effectively. "
     "Covers prompt engineering, retrieval-augmented generation (RAG), and LoRA fine-tuning. "
     "Project: build a domain-specific Q&A chatbot without expensive API calls.",
     "LLMs,transformers,prompt engineering,RAG,LoRA,fine-tuning",
     "DS014"),
    ("DS017", "Causal Inference for Data Scientists", "advanced", "Data Science",
     "Go beyond correlation and learn to estimate causal effects from observational data. "
     "Covers potential outcomes, propensity scores, difference-in-differences, and instrumental variables. "
     "Applied to marketing attribution and policy evaluation.",
     "causal inference,propensity score,difference-in-differences,econometrics",
     "DS012"),

    # ── WEB DEVELOPMENT (beginner) ───────────────────────────────────────────
    ("WD001", "HTML and CSS Fundamentals", "beginner", "Web Development",
     "Build the foundation of every website with HTML5 semantic markup and CSS3 styling. "
     "Covers flexbox, grid, responsive design, and accessibility basics. "
     "By the end you will have a personal portfolio page live on GitHub Pages.",
     "HTML,CSS,responsive design,flexbox,CSS Grid",
     ""),
    ("WD002", "JavaScript for Beginners", "beginner", "Web Development",
     "Learn the programming language of the web from variables and functions to DOM manipulation. "
     "Covers ES6+ features, event handling, fetch API, and asynchronous programming basics. "
     "Project: interactive quiz application.",
     "JavaScript,DOM,ES6,async/await,events",
     "WD001"),
    ("WD003", "Version Control with Git and GitHub", "beginner", "Web Development",
     "Master Git workflows every developer uses daily: commits, branches, merges, and pull requests. "
     "Covers collaborative development, resolving conflicts, and GitHub Actions CI basics. "
     "You will contribute to an open-source repository.",
     "Git,GitHub,version control,branching,CI/CD",
     ""),
    ("WD004", "Introduction to React", "beginner", "Web Development",
     "Build dynamic single-page applications using React, the most popular UI library. "
     "Covers components, props, state, hooks (useState, useEffect), and routing with React Router. "
     "Project: to-do list app with local storage persistence.",
     "React,components,hooks,JSX,React Router",
     "WD002"),
    ("WD005", "Node.js and Express Basics", "beginner", "Web Development",
     "Create your first backend REST API using Node.js and the Express framework. "
     "Covers routing, middleware, request/response cycle, and connecting to a SQLite database. "
     "Project: simple CRUD API for a blog.",
     "Node.js,Express,REST API,middleware,CRUD",
     "WD002"),
    # WEB DEVELOPMENT (intermediate)
    ("WD006", "Full-Stack Web Development with React and Node", "intermediate", "Web Development",
     "Connect a React frontend to a Node/Express backend and ship a full-stack application. "
     "Covers JWT authentication, protected routes, REST API design, and deployment to Heroku. "
     "Project: full-stack social bookmarking app.",
     "React,Node.js,JWT,authentication,full-stack,REST",
     "WD004,WD005"),
    ("WD007", "TypeScript for JavaScript Developers", "intermediate", "Web Development",
     "Add static types to your JavaScript code to catch bugs early and improve editor tooling. "
     "Covers type annotations, interfaces, generics, and integrating TypeScript into React projects. "
     "Refactor an existing JavaScript project to TypeScript.",
     "TypeScript,static typing,generics,interfaces",
     "WD002"),
    ("WD008", "Next.js: Server-Side Rendering and Static Sites", "intermediate", "Web Development",
     "Build production-ready web applications with Next.js including SSR, SSG, and API routes. "
     "Covers file-based routing, Image optimisation, Middleware, and deploying to Vercel. "
     "Project: e-commerce storefront with product pages and checkout.",
     "Next.js,SSR,SSG,Vercel,performance",
     "WD004"),
    ("WD009", "Relational Databases with PostgreSQL", "intermediate", "Web Development",
     "Design normalised database schemas and write efficient queries for web applications. "
     "Covers indexing, transactions, stored procedures, and integrating with Node.js via pg. "
     "Project: design the database for a multi-tenant SaaS app.",
     "PostgreSQL,SQL,database design,transactions,indexing",
     "WD005"),
    ("WD010", "RESTful API Design and Best Practices", "intermediate", "Web Development",
     "Design clean, versioned, and secure REST APIs that developers love to use. "
     "Covers resource naming, status codes, pagination, rate limiting, and OpenAPI documentation. "
     "Project: redesign and document an existing API.",
     "REST,API design,OpenAPI,Swagger,versioning",
     "WD005"),
    ("WD011", "Web Performance Optimisation", "intermediate", "Web Development",
     "Speed up websites using Core Web Vitals, lazy loading, code splitting, and caching strategies. "
     "Covers Lighthouse audits, critical rendering path, CDN configuration, and image optimisation. "
     "Project: optimise a slow e-commerce page to score 90+ on Lighthouse.",
     "performance,Core Web Vitals,lazy loading,CDN,Lighthouse",
     "WD006"),
    # WEB DEVELOPMENT (advanced)
    ("WD012", "Micro-frontends Architecture", "advanced", "Web Development",
     "Break large monolithic frontends into independently deployable micro-frontends. "
     "Covers Module Federation, single-spa, shared design systems, and cross-team workflows. "
     "Case study: migrating a monolith to micro-frontends without downtime.",
     "micro-frontends,Module Federation,architecture,monorepo",
     "WD006,WD007"),
    ("WD013", "GraphQL API Development", "advanced", "Web Development",
     "Design and build GraphQL APIs with Apollo Server, queries, mutations, subscriptions, and DataLoader. "
     "Covers schema-first design, resolver patterns, caching, and securing GraphQL endpoints. "
     "Project: real-time collaborative notes API.",
     "GraphQL,Apollo,resolvers,subscriptions,real-time",
     "WD009,WD010"),
    ("WD014", "DevOps for Web Developers", "advanced", "Web Development",
     "Automate build, test, and deployment pipelines for web applications. "
     "Covers Docker, GitHub Actions, Nginx configuration, SSL, and blue-green deployments. "
     "Project: fully automated CI/CD pipeline for a Next.js app.",
     "DevOps,Docker,CI/CD,Nginx,GitHub Actions",
     "WD003,WD006"),

    # ── UX/UI DESIGN (beginner) ──────────────────────────────────────────────
    ("UX001", "UX Design Fundamentals", "beginner", "UX/UI Design",
     "Understand the principles of user-centred design and how to apply them to digital products. "
     "Covers design thinking, user research methods, empathy mapping, and personas. "
     "Complete a design brief for a mobile app concept.",
     "UX design,design thinking,empathy mapping,user research,personas",
     ""),
    ("UX002", "Introduction to Figma", "beginner", "UX/UI Design",
     "Learn Figma, the industry-standard design and prototyping tool, from scratch. "
     "Covers frames, components, auto-layout, styles, and building interactive prototypes. "
     "Project: redesign a mobile app screen in Figma.",
     "Figma,wireframing,prototyping,auto-layout,components",
     ""),
    ("UX003", "Visual Design Principles", "beginner", "UX/UI Design",
     "Master the visual building blocks of great interface design: typography, color, spacing, and hierarchy. "
     "Covers gestalt principles, grid systems, icon design, and accessible color palettes. "
     "Redesign three poorly designed UI screens applying each principle.",
     "visual design,typography,color theory,hierarchy,grids",
     ""),
    ("UX004", "User Research Methods", "beginner", "UX/UI Design",
     "Learn how to gather actionable insights from real users through interviews, surveys, and usability testing. "
     "Covers recruiting participants, writing discussion guides, affinity mapping, and reporting findings. "
     "Conduct a guerrilla usability test on a live product.",
     "user research,usability testing,interviews,affinity mapping",
     "UX001"),
    ("UX005", "Information Architecture and Navigation Design", "beginner", "UX/UI Design",
     "Organise content so that users always know where they are and where to go next. "
     "Covers card sorting, tree testing, site maps, and navigation patterns for web and mobile. "
     "Project: redesign the IA of a government website.",
     "information architecture,card sorting,navigation,site maps",
     "UX001"),
    # UX/UI DESIGN (intermediate)
    ("UX006", "Interaction Design and Motion", "intermediate", "UX/UI Design",
     "Design micro-interactions and animations that guide users and communicate feedback. "
     "Covers timing, easing, state transitions, and building animated prototypes in Figma and Principle. "
     "Project: design an onboarding flow with meaningful motion.",
     "interaction design,micro-interactions,animation,motion design,Figma",
     "UX002,UX003"),
    ("UX007", "Design Systems: Building and Maintaining", "intermediate", "UX/UI Design",
     "Create a scalable design system that keeps design and engineering in sync across large products. "
     "Covers component libraries, tokens, documentation, versioning, and Storybook integration. "
     "Project: build a complete design system for a SaaS dashboard.",
     "design systems,components,design tokens,Storybook,documentation",
     "UX002,UX003"),
    ("UX008", "UX Writing and Content Strategy", "intermediate", "UX/UI Design",
     "Write clear, helpful interface copy that guides users and reflects brand voice. "
     "Covers microcopy, error messages, onboarding flows, and content auditing. "
     "Rewrite the copy for a real product's critical user flows.",
     "UX writing,microcopy,content strategy,brand voice,error messages",
     "UX001"),
    ("UX009", "Accessibility in UX Design", "intermediate", "UX/UI Design",
     "Design digital products that are usable by people with a wide range of disabilities. "
     "Covers WCAG 2.1 guidelines, ARIA roles, screen reader testing, and accessible color contrast. "
     "Audit and remediate accessibility issues in an existing product.",
     "accessibility,WCAG,ARIA,inclusive design,screen readers",
     "UX002,UX003"),
    ("UX010", "Mobile UX Design Patterns", "intermediate", "UX/UI Design",
     "Apply proven mobile design patterns to build apps users find intuitive on iOS and Android. "
     "Covers gesture navigation, touch targets, bottom sheets, and platform-specific conventions. "
     "Design a cross-platform mobile app in Figma.",
     "mobile design,iOS,Android,gestures,design patterns",
     "UX002,UX005"),
    # UX/UI DESIGN (advanced)
    ("UX011", "Service Design and Customer Journey Mapping", "advanced", "UX/UI Design",
     "Design end-to-end customer experiences that span digital touchpoints, physical channels, and people. "
     "Covers journey mapping, service blueprinting, stakeholder alignment, and co-design workshops. "
     "Redesign the customer experience for a retail banking onboarding journey.",
     "service design,journey mapping,service blueprinting,co-design",
     "UX004,UX005"),
    ("UX012", "Advanced Prototyping with Framer", "advanced", "UX/UI Design",
     "Build high-fidelity, production-like prototypes in Framer using React-based components. "
     "Covers overrides, variants, CMS integration, and sharing prototypes with developers. "
     "Project: fully interactive data dashboard prototype.",
     "Framer,prototyping,React components,advanced interactions",
     "UX006,UX007"),
    ("UX013", "Quantitative UX Research and Analytics", "advanced", "UX/UI Design",
     "Combine UX intuition with data: track behavioural metrics, run surveys at scale, and interpret analytics. "
     "Covers funnel analysis, cohort analysis, Google Analytics 4, and Hotjar session replays. "
     "Identify and fix a significant drop-off in a real product funnel.",
     "quantitative research,analytics,funnel analysis,Google Analytics,Hotjar",
     "UX004,UX009"),

    # ── BUSINESS / MARKETING (beginner) ─────────────────────────────────────
    ("BM001", "Digital Marketing Fundamentals", "beginner", "Business/Marketing",
     "Get a complete overview of the digital marketing landscape: SEO, SEM, social, email, and content. "
     "Covers the marketing funnel, target audience definition, KPIs, and campaign planning. "
     "Build a digital marketing plan for a fictional product launch.",
     "digital marketing,SEO,SEM,content marketing,email marketing",
     ""),
    ("BM002", "Business Analytics with Excel and Google Sheets", "beginner", "Business/Marketing",
     "Analyse business data and present insights to stakeholders without writing code. "
     "Covers dashboards, pivot tables, VLOOKUP, trend analysis, and scenario modelling. "
     "Project: revenue dashboard for a fictional e-commerce business.",
     "Excel,Google Sheets,business analytics,dashboards,pivot tables",
     ""),
    ("BM003", "Introduction to Entrepreneurship", "beginner", "Business/Marketing",
     "Explore the mindset and practical steps needed to turn an idea into a viable business. "
     "Covers opportunity identification, lean startup methodology, business model canvas, and pitching. "
     "Develop and pitch your own startup concept.",
     "entrepreneurship,lean startup,business model canvas,pitching,ideation",
     ""),
    ("BM004", "Social Media Marketing Strategy", "beginner", "Business/Marketing",
     "Create and execute social media strategies that grow brand awareness and drive conversions. "
     "Covers platform selection, content calendars, community management, and paid social basics. "
     "Project: 30-day social media plan for a small business.",
     "social media,content strategy,community management,paid social,Instagram",
     "BM001"),
    ("BM005", "Project Management Fundamentals", "beginner", "Business/Marketing",
     "Learn how to plan, execute, and deliver projects on time and within budget. "
     "Covers waterfall vs agile methodologies, Gantt charts, risk registers, and stakeholder communication. "
     "Simulate managing a software product launch.",
     "project management,agile,scrum,Gantt,risk management",
     ""),
    # BUSINESS / MARKETING (intermediate)
    ("BM006", "SEO: Technical and Content Strategy", "intermediate", "Business/Marketing",
     "Improve organic search rankings through technical SEO and high-quality content creation. "
     "Covers crawl optimisation, schema markup, keyword research, link building, and Core Web Vitals. "
     "Audit and optimise a live website for SEO.",
     "SEO,technical SEO,keyword research,link building,schema markup",
     "BM001"),
    ("BM007", "Google Analytics 4 and Data-Driven Marketing", "intermediate", "Business/Marketing",
     "Measure what matters: set up GA4, build custom reports, and use data to improve campaigns. "
     "Covers event tracking, conversion funnels, attribution models, and BigQuery export. "
     "Project: diagnose and fix a conversion rate problem using analytics.",
     "Google Analytics,GA4,conversion optimisation,attribution,data-driven marketing",
     "BM001,BM002"),
    ("BM008", "Growth Hacking and Conversion Optimisation", "intermediate", "Business/Marketing",
     "Apply rapid experimentation to accelerate user acquisition, activation, and retention. "
     "Covers AARRR framework, landing page optimisation, viral loops, and email drip campaigns. "
     "Run a full growth experiment cycle on a real product.",
     "growth hacking,CRO,AARRR,landing pages,email automation",
     "BM007"),
    ("BM009", "Financial Modelling for Business Decisions", "intermediate", "Business/Marketing",
     "Build financial models in Excel to evaluate business decisions, forecast revenue, and value companies. "
     "Covers three-statement models, DCF valuation, scenario analysis, and sensitivity tables. "
     "Model the financial impact of launching a new product line.",
     "financial modelling,Excel,DCF,valuation,forecasting",
     "BM002"),
    ("BM010", "Product Management Essentials", "intermediate", "Business/Marketing",
     "Learn how to define, prioritise, and ship products that customers love and businesses need. "
     "Covers PRDs, user stories, OKRs, roadmapping, and working with engineering and design. "
     "Create a full product brief for a mobile app feature.",
     "product management,OKRs,roadmapping,user stories,PRD",
     "BM005"),
    # BUSINESS / MARKETING (advanced)
    ("BM011", "Brand Strategy and Positioning", "advanced", "Business/Marketing",
     "Define and communicate a distinctive brand position that resonates deeply with target customers. "
     "Covers brand archetypes, positioning maps, messaging frameworks, and brand equity measurement. "
     "Develop a complete brand strategy for a tech startup.",
     "brand strategy,positioning,messaging,brand equity,marketing",
     "BM001,BM004"),
    ("BM012", "Advanced Paid Media: Google and Meta Ads", "advanced", "Business/Marketing",
     "Manage large-scale paid advertising campaigns across Google Search, Display, YouTube, and Meta. "
     "Covers audience targeting, bidding strategies, creative testing, ROAS optimisation, and attribution. "
     "Manage a $10k/month mock budget across platforms.",
     "Google Ads,Meta Ads,paid media,ROAS,attribution,audience targeting",
     "BM007"),
    ("BM013", "Strategic Business Analysis", "advanced", "Business/Marketing",
     "Use advanced frameworks to analyse industries, evaluate strategy, and support executive decisions. "
     "Covers Porter's Five Forces, SWOT, value chain analysis, competitive benchmarking, and case interview prep. "
     "Complete a full competitive analysis for a Fortune 500 company.",
     "strategy,Porter's Five Forces,SWOT,competitive analysis,business analysis",
     "BM009,BM010"),

    # ── CLOUD / DEVOPS (beginner) ─────────────────────────────────────────────
    ("CD001", "Introduction to Cloud Computing", "beginner", "Cloud/DevOps",
     "Understand the core concepts, service models, and major providers of cloud computing. "
     "Covers IaaS, PaaS, SaaS, AWS vs Azure vs GCP, pricing models, and cloud security basics. "
     "Set up a free-tier cloud account and deploy a static website.",
     "cloud computing,AWS,Azure,GCP,IaaS,PaaS,SaaS",
     ""),
    ("CD002", "Linux Command Line for Developers", "beginner", "Cloud/DevOps",
     "Navigate and control Linux/Unix systems from the command line with confidence. "
     "Covers file system, permissions, bash scripting, process management, and SSH. "
     "Automate a real server setup task with a bash script.",
     "Linux,bash,command line,SSH,scripting",
     ""),
    ("CD003", "Docker for Beginners", "beginner", "Cloud/DevOps",
     "Package applications into containers so they run consistently anywhere. "
     "Covers Docker images, containers, Dockerfiles, volumes, networking, and Docker Compose. "
     "Containerise a Python Flask application.",
     "Docker,containers,Dockerfile,Docker Compose,containerisation",
     "CD002"),
    ("CD004", "Networking Fundamentals for Cloud Engineers", "beginner", "Cloud/DevOps",
     "Learn the networking concepts every cloud practitioner needs: TCP/IP, DNS, HTTP/S, load balancers, and firewalls. "
     "Covers subnets, VPCs, routing tables, and network security groups in AWS. "
     "Design a simple three-tier network architecture.",
     "networking,TCP/IP,DNS,VPC,subnets,load balancer",
     "CD001"),
    ("CD005", "AWS Cloud Practitioner Essentials", "beginner", "Cloud/DevOps",
     "Prepare for the AWS Cloud Practitioner certification with hands-on labs covering 20+ AWS services. "
     "Covers EC2, S3, RDS, Lambda, IAM, CloudWatch, and cost management. "
     "Pass mock exams and earn a verified badge.",
     "AWS,EC2,S3,RDS,Lambda,IAM,CloudWatch",
     "CD001"),
    # CLOUD / DEVOPS (intermediate)
    ("CD006", "Kubernetes: Container Orchestration", "intermediate", "Cloud/DevOps",
     "Deploy, scale, and manage containerised applications using Kubernetes. "
     "Covers Pods, Deployments, Services, ConfigMaps, Ingress, HPA, and Helm charts. "
     "Deploy a microservices application to a managed Kubernetes cluster.",
     "Kubernetes,containers,Helm,microservices,orchestration",
     "CD003"),
    ("CD007", "Terraform: Infrastructure as Code", "intermediate", "Cloud/DevOps",
     "Provision and manage cloud infrastructure reproducibly using Terraform. "
     "Covers HCL syntax, providers, modules, state management, and multi-environment deployments. "
     "Automate the setup of a production-grade AWS environment.",
     "Terraform,IaC,AWS,infrastructure,HCL,modules",
     "CD005"),
    ("CD008", "CI/CD Pipelines with GitHub Actions", "intermediate", "Cloud/DevOps",
     "Build automated pipelines that test, build, and deploy software on every code change. "
     "Covers workflow YAML syntax, matrix builds, secrets management, and deployment to AWS and Azure. "
     "Set up a pipeline that deploys a Dockerised app end-to-end.",
     "CI/CD,GitHub Actions,pipelines,automation,deployment",
     "CD003,WD003"),
    ("CD009", "AWS Solutions Architect Associate", "intermediate", "Cloud/DevOps",
     "Design highly available, fault-tolerant, and cost-efficient architectures on AWS. "
     "Covers VPC design, auto-scaling, RDS Multi-AZ, ELB, CloudFront, and well-architected framework. "
     "Pass the SAA-C03 certification exam.",
     "AWS,architecture,VPC,auto-scaling,CloudFront,S3,high availability",
     "CD005,CD004"),
    ("CD010", "Observability: Logging, Metrics, and Tracing", "intermediate", "Cloud/DevOps",
     "Instrument and monitor distributed systems so you can find and fix problems fast. "
     "Covers Prometheus, Grafana, OpenTelemetry, distributed tracing, and alerting strategies. "
     "Instrument a microservices app with full observability.",
     "observability,Prometheus,Grafana,OpenTelemetry,monitoring,logging",
     "CD006"),
    ("CD011", "Site Reliability Engineering Fundamentals", "intermediate", "Cloud/DevOps",
     "Apply SRE practices to keep production systems reliable, scalable, and easy to operate. "
     "Covers SLOs, SLAs, error budgets, incident management, and chaos engineering. "
     "Define SLOs and run a game-day chaos experiment.",
     "SRE,SLO,SLA,incident management,chaos engineering,reliability",
     "CD010"),
    # CLOUD / DEVOPS (advanced)
    ("CD012", "Advanced Kubernetes: Security and Scaling", "advanced", "Cloud/DevOps",
     "Secure and scale Kubernetes clusters for production workloads handling millions of requests. "
     "Covers RBAC, network policies, Pod Security Standards, cluster autoscaler, and Karpenter. "
     "Harden and scale a production cluster under load.",
     "Kubernetes,security,RBAC,autoscaling,Karpenter,network policies",
     "CD006,CD010"),
    ("CD013", "Multi-Cloud Strategy and Architecture", "advanced", "Cloud/DevOps",
     "Design systems that span AWS, Azure, and GCP to maximise resilience and avoid vendor lock-in. "
     "Covers cloud-agnostic tooling, data replication, identity federation, and cost governance. "
     "Architect a multi-cloud deployment for a global fintech application.",
     "multi-cloud,AWS,Azure,GCP,architecture,resilience",
     "CD009"),
    ("CD014", "Platform Engineering: Building Internal Developer Platforms", "advanced", "Cloud/DevOps",
     "Build self-service platforms that accelerate developer velocity while enforcing guardrails. "
     "Covers Backstage, golden paths, Crossplane, GitOps with ArgoCD, and platform metrics. "
     "Build and demo an internal developer portal.",
     "platform engineering,Backstage,GitOps,ArgoCD,developer experience,Crossplane",
     "CD007,CD012"),

    # ── ADDITIONAL CROSS-CATEGORY COURSES ────────────────────────────────────
    # Data Science extras
    ("DS018", "Business Intelligence with Power BI", "intermediate", "Data Science",
     "Build interactive business dashboards and reports using Microsoft Power BI. "
     "Covers DAX calculations, data modelling, Power Query, and publishing to the Power BI Service. "
     "Project: executive sales dashboard connected to a live data source.",
     "Power BI,DAX,data visualisation,business intelligence,dashboards",
     "DS003,DS005"),
    ("DS019", "Cloud Data Warehousing with BigQuery", "intermediate", "Data Science",
     "Run fast analytics on massive datasets using Google BigQuery. "
     "Covers loading data, writing efficient SQL, partitioning, clustering, and cost control. "
     "Analyse 1 billion rows of NYC taxi trip data.",
     "BigQuery,SQL,cloud,data warehousing,analytics",
     "DS005"),
    ("DS020", "Data Governance and Privacy Engineering", "advanced", "Data Science",
     "Design data systems that are compliant with GDPR, CCPA, and internal governance policies. "
     "Covers data catalogues, lineage tracking, PII anonymisation, consent management, and auditing. "
     "Implement a data governance framework for a healthcare dataset.",
     "data governance,GDPR,privacy,PII,compliance,data catalogue",
     "DS011"),
    # Web Development extras
    ("WD015", "Progressive Web Apps (PWA)", "intermediate", "Web Development",
     "Turn websites into app-like experiences that work offline and can be installed on any device. "
     "Covers service workers, Web App Manifest, push notifications, and caching strategies. "
     "Convert an existing React app into a fully functional PWA.",
     "PWA,service workers,offline,push notifications,React",
     "WD004"),
    ("WD016", "Web Security Fundamentals", "intermediate", "Web Development",
     "Protect web applications against the OWASP Top 10 vulnerabilities and common attacks. "
     "Covers XSS, CSRF, SQL injection, authentication hardening, HTTPS, and Content Security Policy. "
     "Pentest and fix a deliberately vulnerable web application.",
     "web security,OWASP,XSS,CSRF,SQL injection,authentication",
     "WD006"),
    ("WD017", "Serverless Architecture with AWS Lambda", "advanced", "Web Development",
     "Build scalable backend functions without managing servers using AWS Lambda and the Serverless Framework. "
     "Covers triggers, API Gateway, DynamoDB, cold starts, and cost optimisation for serverless. "
     "Migrate a monolithic REST API to a serverless architecture.",
     "serverless,AWS Lambda,API Gateway,DynamoDB,Serverless Framework",
     "WD010,CD005"),
    # UX extras
    ("UX014", "Design for Emerging Interfaces: VR and AR", "advanced", "UX/UI Design",
     "Design immersive experiences for virtual and augmented reality platforms. "
     "Covers spatial design principles, 3D interaction patterns, comfort heuristics, and prototyping in Unity. "
     "Design and prototype a museum AR experience.",
     "VR,AR,spatial design,immersive,Unity,3D interaction",
     "UX006,UX007"),
    ("UX015", "Design Leadership and Team Management", "advanced", "UX/UI Design",
     "Lead design teams effectively and advocate for design at the executive level. "
     "Covers hiring, critique facilitation, OKR alignment, design ops, and communicating ROI of design. "
     "Create a design leadership playbook for a growing startup.",
     "design leadership,design ops,team management,OKRs,communication",
     "UX011,UX007"),
    # Business extras
    ("BM014", "Data-Driven HR and People Analytics", "intermediate", "Business/Marketing",
     "Apply data analysis to improve hiring, retention, and employee engagement decisions. "
     "Covers attrition modelling, compensation benchmarking, engagement surveys, and HRIS data. "
     "Build a people analytics dashboard for a 500-person company.",
     "people analytics,HR,attrition,engagement,dashboards",
     "BM002,DS005"),
    ("BM015", "E-commerce: Build and Scale an Online Store", "intermediate", "Business/Marketing",
     "Launch and grow a profitable online store from zero to first 1000 customers. "
     "Covers Shopify setup, product photography, email marketing, Facebook ads, and scaling operations. "
     "Launch a fully functional Shopify store.",
     "e-commerce,Shopify,email marketing,Facebook Ads,operations",
     "BM001,BM004"),
    # Cloud extras
    ("CD015", "Google Cloud Professional Cloud Architect", "advanced", "Cloud/DevOps",
     "Design enterprise-grade cloud solutions on Google Cloud Platform for the Professional Cloud Architect exam. "
     "Covers GKE, BigQuery, Anthos, Spanner, VPC design, identity management, and cost optimisation. "
     "Pass the GCP PCA certification exam.",
     "GCP,Google Cloud,Kubernetes,BigQuery,architecture,certification",
     "CD013"),
    ("CD016", "FinOps: Cloud Cost Management", "intermediate", "Cloud/DevOps",
     "Optimise cloud spending without sacrificing performance or reliability. "
     "Covers Reserved Instances, Spot/Preemptible VMs, rightsizing, tagging strategy, and cost allocation. "
     "Reduce a mock cloud bill by 40% using FinOps practices.",
     "FinOps,cost management,AWS,cloud,Reserved Instances,Spot instances",
     "CD005"),
    ("CD017", "Security Engineering in the Cloud", "advanced", "Cloud/DevOps",
     "Implement defence-in-depth security across cloud environments to protect sensitive workloads. "
     "Covers IAM least-privilege, secrets management, WAF, DDoS protection, threat modelling, and compliance. "
     "Perform a security posture assessment and remediation on an AWS account.",
     "cloud security,IAM,WAF,DDoS,secrets management,compliance,threat modelling",
     "CD009,CD012"),
]


# ── PROJECTS ──────────────────────────────────────────────────────────────────
# Applied work that exercises skills taught elsewhere. ``skills`` names what the
# project puts into practice, and ``prerequisites`` are the courses that teach
# those skills, so a project can never be scheduled before its groundwork.
# Tuple: (id, title, difficulty, category, description, skills, prerequisites, hours)
PROJECTS = [
    # ── Data Science ─────────────────────────────────────────────────────────
    ("PR001", "Project: Sales Analytics Dashboard", "beginner", "Data Science",
     "Build an end-to-end sales dashboard from a raw transactional export. "
     "You will model the data in SQL, aggregate it into monthly cohorts, and present "
     "the findings as a single-page dashboard a sales manager could act on.",
     "SQL,Excel,pivot tables,dashboards,data visualization,aggregations",
     "DS005,DS006", 10),
    ("PR002", "Project: Exploratory Analysis of Public Health Data", "beginner", "Data Science",
     "Take a messy public health dataset and turn it into a defensible written analysis. "
     "You will handle missing values and outliers, then support three findings with charts. "
     "Deliverable is a notebook plus a one-page summary for a non-technical reader.",
     "pandas,data cleaning,EDA,matplotlib,seaborn,missing values",
     "DS003,DS004", 12),
    ("PR003", "Project: Customer Churn Prediction Model", "intermediate", "Data Science",
     "Build and evaluate a churn classifier on realistic subscription data. "
     "You will engineer features, compare three model families, and justify your choice "
     "using precision-recall trade-offs rather than accuracy alone.",
     "machine learning,scikit-learn,classification,feature engineering,model evaluation",
     "DS007,DS009", 20),
    ("PR004", "Project: Review Sentiment Analysis Pipeline", "intermediate", "Data Science",
     "Construct a pipeline that classifies product review sentiment and surfaces themes. "
     "You will compare a classical baseline against a transformer and report where each fails. "
     "Deliverable is a reusable pipeline plus an error analysis.",
     "NLP,text classification,transformers,spaCy,sentiment analysis",
     "DS008", 18),
    ("PR005", "Project: Retail Demand Forecasting System", "intermediate", "Data Science",
     "Forecast 30-day demand across several product lines and quantify your uncertainty. "
     "You will decompose seasonality, benchmark ARIMA against Prophet, and back-test honestly. "
     "Deliverable includes a forecast with prediction intervals.",
     "time series,ARIMA,Prophet,forecasting,statistics",
     "DS010", 20),
    ("PR006", "Project: Deploy a Monitored ML Service", "advanced", "Data Science",
     "Take a trained model to a running endpoint with monitoring and a rollback path. "
     "You will containerise the service, track experiments, and add drift detection with alerts. "
     "Deliverable is a deployed service and a runbook.",
     "MLOps,Docker,FastAPI,MLflow,model deployment,drift detection",
     "DS015", 25),

    # ── Web Development ──────────────────────────────────────────────────────
    ("PR007", "Project: Responsive Personal Portfolio", "beginner", "Web Development",
     "Ship a responsive portfolio site that scores well on Lighthouse. "
     "You will build a mobile-first layout with flexbox and CSS Grid, then deploy it publicly. "
     "Deliverable is a live URL and the source repository.",
     "HTML,CSS,responsive design,flexbox,CSS Grid",
     "WD001", 8),
    ("PR008", "Project: Interactive Task Manager App", "beginner", "Web Development",
     "Build a task manager with filtering, persistence and keyboard accessibility. "
     "You will manage component state, handle events, and persist to local storage. "
     "Deliverable is a working app with no console errors.",
     "JavaScript,React,components,hooks,DOM,events",
     "WD002,WD004", 14),
    ("PR009", "Project: Full-Stack App with Authentication", "intermediate", "Web Development",
     "Build a full-stack bookmarking application with real authentication. "
     "You will implement JWT auth, protected routes, and a REST API against a database. "
     "Deliverable is a deployed app with signup, login and per-user data.",
     "React,Node.js,Express,JWT,authentication,full-stack,REST",
     "WD006", 25),
    ("PR010", "Project: E-commerce Storefront with Next.js", "intermediate", "Web Development",
     "Build a storefront with server-rendered product pages and a working checkout flow. "
     "You will model the catalog in PostgreSQL and use static generation for product pages. "
     "Deliverable is a deployed storefront with a seeded catalog.",
     "Next.js,SSR,SSG,PostgreSQL,performance,database design",
     "WD008,WD009", 30),
    ("PR011", "Project: Production GraphQL API", "advanced", "Web Development",
     "Design and ship a GraphQL API with subscriptions, caching and query-depth limits. "
     "You will solve the N+1 problem with DataLoader and secure the endpoint against abuse. "
     "Deliverable is a documented API with load-test results.",
     "GraphQL,Apollo,resolvers,subscriptions,real-time,API design",
     "WD013", 28),

    # ── UX/UI Design ─────────────────────────────────────────────────────────
    ("PR012", "Project: Usability Audit and Redesign", "beginner", "UX/UI Design",
     "Run a usability audit on a real public website and propose an evidence-based redesign. "
     "You will test with five participants, synthesise findings, and prioritise fixes by impact. "
     "Deliverable is an audit report plus annotated redesign screens.",
     "UX design,user research,usability testing,personas,affinity mapping",
     "UX001,UX004", 12),
    ("PR013", "Project: Mobile App Prototype in Figma", "beginner", "UX/UI Design",
     "Design and prototype a three-flow mobile app to a consistent visual standard. "
     "You will build a component library with auto-layout and apply type and colour systems. "
     "Deliverable is an interactive prototype ready for user testing.",
     "Figma,wireframing,prototyping,auto-layout,visual design,typography",
     "UX002,UX003", 14),
    ("PR014", "Project: Design System for a SaaS Dashboard", "intermediate", "UX/UI Design",
     "Build a documented design system that design and engineering can both work from. "
     "You will define tokens, build a component library, and document usage and versioning rules. "
     "Deliverable is a component library plus written guidelines.",
     "design systems,components,design tokens,documentation,Storybook",
     "UX007", 24),
    ("PR015", "Project: Accessibility Remediation Case Study", "intermediate", "UX/UI Design",
     "Audit an existing interface against WCAG and remediate the failures you find. "
     "You will test with a screen reader, fix contrast and focus order, and document each change. "
     "Deliverable is a before-and-after report with conformance notes.",
     "accessibility,WCAG,ARIA,inclusive design,screen readers",
     "UX009", 16),
    ("PR016", "Project: End-to-End Service Blueprint", "advanced", "UX/UI Design",
     "Map a multi-channel service and design an intervention at its weakest point. "
     "You will produce a journey map, a blueprint including backstage actors, and a pilot plan. "
     "Deliverable is a blueprint plus a measurable pilot proposal.",
     "service design,journey mapping,service blueprinting,co-design",
     "UX011", 22),

    # ── Business/Marketing ───────────────────────────────────────────────────
    ("PR017", "Project: Content and SEO Plan for a Niche Site", "beginner", "Business/Marketing",
     "Produce a three-month content and SEO plan for a niche site and publish the first pieces. "
     "You will do keyword research, map search intent to content, and set measurable targets. "
     "Deliverable is a content calendar plus two published articles.",
     "SEO,content marketing,keyword research,digital marketing",
     "BM001", 12),
    ("PR018", "Project: Marketing Analytics Dashboard", "intermediate", "Business/Marketing",
     "Build a dashboard that attributes revenue to channels and explains what to do next. "
     "You will configure event tracking, model the funnel, and report acquisition cost by channel. "
     "Deliverable is a live dashboard plus a written read-out.",
     "Google Analytics,GA4,attribution,dashboards,business analytics,conversion optimisation",
     "BM002,BM007", 16),
    ("PR019", "Project: SaaS Financial Model", "intermediate", "Business/Marketing",
     "Build a driver-based three-statement model for a SaaS business with a scenario toggle. "
     "You will model cohort retention, CAC payback and runway, then stress-test the assumptions. "
     "Deliverable is a documented model plus a one-page summary.",
     "financial modelling,Excel,forecasting,valuation,DCF",
     "BM009", 20),
    ("PR020", "Project: Run a Conversion Experiment", "intermediate", "Business/Marketing",
     "Design, run and analyse a real conversion experiment end to end. "
     "You will form a hypothesis, size the test, run it, and report a decision with confidence. "
     "Deliverable is an experiment write-up including what you would do next.",
     "CRO,growth hacking,A/B testing,landing pages,experimentation",
     "BM008", 18),
    ("PR021", "Project: Go-to-Market Strategy", "advanced", "Business/Marketing",
     "Build a defensible go-to-market plan for a new product entering a contested market. "
     "You will segment the market, define positioning, and model channel economics. "
     "Deliverable is a strategy document plus a launch plan.",
     "brand strategy,positioning,strategy,competitive analysis,messaging",
     "BM011,BM013", 24),

    # ── Cloud/DevOps ─────────────────────────────────────────────────────────
    ("PR022", "Project: Containerise and Deploy a Web App", "beginner", "Cloud/DevOps",
     "Take an existing application from local-only to running in the cloud. "
     "You will write a multi-stage Dockerfile, push to a registry, and deploy behind a load balancer. "
     "Deliverable is a reachable URL plus the deployment steps.",
     "Docker,containers,Dockerfile,AWS,EC2,S3,cloud computing",
     "CD003,CD005", 12),
    ("PR023", "Project: Infrastructure as Code with Terraform", "intermediate", "Cloud/DevOps",
     "Reproduce a complete environment from code, with no console clicking. "
     "You will write reusable modules, manage remote state, and prove the environment rebuilds cleanly. "
     "Deliverable is a Terraform repository plus a teardown and rebuild log.",
     "Terraform,IaC,AWS,infrastructure,HCL,modules",
     "CD007", 18),
    ("PR024", "Project: CI/CD Pipeline with Observability", "intermediate", "Cloud/DevOps",
     "Build a pipeline that tests, builds and deploys on every merge, and tells you when it breaks. "
     "You will add gated stages, instrument the service, and wire up dashboards and alerts. "
     "Deliverable is a working pipeline plus an alert that fires on a seeded failure.",
     "CI/CD,GitHub Actions,automation,deployment,observability,Prometheus,Grafana",
     "CD008,CD010", 22),
    ("PR025", "Project: Multi-Region Highly Available Architecture", "advanced", "Cloud/DevOps",
     "Design and document an architecture that survives the loss of an entire region. "
     "You will choose replication strategies, define RTO and RPO targets, and cost the design. "
     "Deliverable is an architecture decision record plus a failover runbook.",
     "AWS,architecture,high availability,VPC,auto-scaling,multi-cloud,resilience",
     "CD009,CD013", 30),
]

# ── ASSESSMENTS ───────────────────────────────────────────────────────────────
# Short checks that validate a cluster of skills. Gated behind the courses that
# cover the material, so an assessment always lands after the teaching.
# Tuple: (id, title, difficulty, category, description, skills, prerequisites, hours)
ASSESSMENTS = [
    # ── Data Science ─────────────────────────────────────────────────────────
    ("AS001", "Assessment: Python and Data Handling", "beginner", "Data Science",
     "Confirm you can load, reshape and clean a dataset unaided. "
     "Mixed multiple-choice and short coding tasks against an unseen dataset.",
     "Python,pandas,data cleaning,data types,functions",
     "DS001,DS003", 1),
    ("AS002", "Assessment: SQL Proficiency", "beginner", "Data Science",
     "Write queries against an unfamiliar schema under time pressure. "
     "Covers joins, grouping, subqueries and window functions.",
     "SQL,joins,aggregations,querying,databases",
     "DS005", 1),
    ("AS003", "Assessment: Statistics and Experimentation", "intermediate", "Data Science",
     "Interpret experiment results and identify invalid conclusions. "
     "Covers hypothesis testing, statistical power, p-values and common traps.",
     "statistics,hypothesis testing,A/B testing,experimentation,p-value",
     "DS002,DS012", 2),
    ("AS004", "Assessment: Machine Learning Competency", "intermediate", "Data Science",
     "Diagnose and repair a deliberately flawed modelling pipeline. "
     "Covers leakage, imbalanced classes, validation strategy and metric selection.",
     "machine learning,scikit-learn,model evaluation,feature engineering,classification",
     "DS007,DS009", 2),
    ("AS005", "Assessment: MLOps Production Readiness", "advanced", "Data Science",
     "Review a model deployment against a production readiness checklist. "
     "Covers reproducibility, monitoring, rollback and drift response.",
     "MLOps,model deployment,drift detection,Docker,MLflow",
     "DS015", 2),

    # ── Web Development ──────────────────────────────────────────────────────
    ("AS006", "Assessment: Frontend Fundamentals", "beginner", "Web Development",
     "Reproduce a given layout and fix scripted bugs in provided code. "
     "Covers semantic HTML, responsive CSS and core JavaScript behaviour.",
     "HTML,CSS,JavaScript,responsive design,DOM",
     "WD001,WD002", 1),
    ("AS007", "Assessment: React Component Design", "beginner", "Web Development",
     "Refactor a poorly structured component tree and justify each change. "
     "Covers state placement, hook rules, keys and re-render behaviour.",
     "React,components,hooks,JSX",
     "WD004", 1),
    ("AS008", "Assessment: Backend and API Design", "intermediate", "Web Development",
     "Critique an existing API and propose a versioned redesign. "
     "Covers resource modelling, status codes, pagination and idempotency.",
     "REST,API design,Node.js,Express,OpenAPI,versioning",
     "WD005,WD010", 2),
    ("AS009", "Assessment: Web Security Audit", "intermediate", "Web Development",
     "Find and rank the vulnerabilities in a deliberately insecure application. "
     "Covers XSS, CSRF, injection and broken authentication.",
     "web security,OWASP,XSS,CSRF,SQL injection,authentication",
     "WD016", 2),

    # ── UX/UI Design ─────────────────────────────────────────────────────────
    ("AS010", "Assessment: UX Research Methods", "beginner", "UX/UI Design",
     "Choose and defend the right research method for several scenarios. "
     "Covers method selection, sample size, bias and synthesis.",
     "user research,usability testing,interviews,UX design,personas",
     "UX001,UX004", 1),
    ("AS011", "Assessment: Visual and Interaction Critique", "beginner", "UX/UI Design",
     "Critique a set of screens against stated design principles. "
     "Covers hierarchy, typography, colour contrast and interaction feedback.",
     "visual design,typography,color theory,hierarchy,Figma,interaction design",
     "UX002,UX003", 1),
    ("AS012", "Assessment: Design Systems", "intermediate", "UX/UI Design",
     "Resolve component API and token structure decisions for a growing system. "
     "Covers naming, variants, versioning and adoption strategy.",
     "design systems,components,design tokens,documentation",
     "UX007", 2),
    ("AS013", "Assessment: Accessibility Conformance", "intermediate", "UX/UI Design",
     "Evaluate screens against WCAG success criteria and cite the specific failures. "
     "Covers contrast, keyboard operability, ARIA misuse and focus management.",
     "accessibility,WCAG,ARIA,inclusive design,screen readers",
     "UX009", 2),

    # ── Business/Marketing ───────────────────────────────────────────────────
    ("AS014", "Assessment: Digital Marketing Channels", "beginner", "Business/Marketing",
     "Allocate a fixed budget across channels and defend the split. "
     "Covers channel economics, organic versus paid trade-offs and funnel stages.",
     "digital marketing,SEO,SEM,content marketing,email marketing",
     "BM001", 1),
    ("AS015", "Assessment: Analytics and Attribution", "intermediate", "Business/Marketing",
     "Diagnose a broken analytics setup and correct the attribution model. "
     "Covers event design, attribution windows and misleading metrics.",
     "Google Analytics,GA4,attribution,conversion optimisation,business analytics",
     "BM007", 2),
    ("AS016", "Assessment: Financial Modelling", "intermediate", "Business/Marketing",
     "Audit a financial model for broken logic and unsupportable assumptions. "
     "Covers driver structure, circular references and scenario design.",
     "financial modelling,Excel,forecasting,valuation,DCF",
     "BM009", 2),
    ("AS017", "Assessment: Product Strategy Case", "advanced", "Business/Marketing",
     "Work a product strategy case and commit to a prioritisation decision. "
     "Covers segmentation, positioning, trade-offs and success metrics.",
     "product management,strategy,OKRs,competitive analysis,roadmapping",
     "BM010,BM013", 2),

    # ── Cloud/DevOps ─────────────────────────────────────────────────────────
    ("AS018", "Assessment: Linux and Containers", "beginner", "Cloud/DevOps",
     "Diagnose a broken container and a misbehaving Linux service. "
     "Covers shell fluency, permissions, image layers and networking basics.",
     "Linux,bash,command line,Docker,containers,Dockerfile",
     "CD002,CD003", 1),
    ("AS019", "Assessment: Kubernetes Operations", "intermediate", "Cloud/DevOps",
     "Restore a failing workload in a cluster you have not seen before. "
     "Covers scheduling failures, probes, resource limits and service routing.",
     "Kubernetes,containers,orchestration,Helm,microservices",
     "CD006", 2),
    ("AS020", "Assessment: Cloud Architecture Design", "advanced", "Cloud/DevOps",
     "Review a proposed architecture and identify its failure modes and cost risks. "
     "Covers availability, scaling, network design and least-privilege access.",
     "AWS,architecture,VPC,auto-scaling,high availability,IAM",
     "CD009", 2),
]

FIELDNAMES = [
    "course_id", "title", "description", "skills", "prerequisites",
    "difficulty_level", "category", "resource_type", "duration_hours",
]


def _course_rows():
    for cid, title, diff, cat, desc, skills, prereqs in COURSES:
        yield {
            "course_id": cid,
            "title": title,
            "description": desc,
            "skills": skills,
            "prerequisites": prereqs,
            "difficulty_level": diff,
            "category": cat,
            "resource_type": "course",
            "duration_hours": COURSE_HOURS.get(diff, 10),
        }


def _typed_rows(records, resource_type):
    for cid, title, diff, cat, desc, skills, prereqs, hours in records:
        yield {
            "course_id": cid,
            "title": title,
            "description": desc,
            "skills": skills,
            "prerequisites": prereqs,
            "difficulty_level": diff,
            "category": cat,
            "resource_type": resource_type,
            "duration_hours": hours,
        }


def build_rows():
    """Return every catalog row: courses first, then projects, then assessments."""
    return [
        *_course_rows(),
        *_typed_rows(PROJECTS, "project"),
        *_typed_rows(ASSESSMENTS, "assessment"),
    ]


def validate(rows):
    """
    Fail loudly on structural problems before writing the file.

    A dangling prerequisite degrades the path generator silently rather than
    raising, so catching it here is far cheaper than debugging a truncated path
    later.
    """
    ids = [r["course_id"] for r in rows]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate course_id values: {sorted(duplicates)}")

    id_set = set(ids)
    dangling = {}
    for row in rows:
        missing = [
            p.strip()
            for p in row["prerequisites"].split(",")
            if p.strip() and p.strip() not in id_set
        ]
        if missing:
            dangling[row["course_id"]] = missing
    if dangling:
        raise ValueError(f"Prerequisites referencing unknown IDs: {dangling}")

    for row in rows:
        if not row["skills"].strip():
            raise ValueError(f"{row['course_id']} has no skills listed")
        if row["difficulty_level"] not in ("beginner", "intermediate", "advanced"):
            raise ValueError(f"{row['course_id']} has bad difficulty: {row['difficulty_level']}")
        if int(row["duration_hours"]) <= 0:
            raise ValueError(f"{row['course_id']} has a non-positive duration")

    # A project or assessment with no prerequisites could be scheduled before
    # anything that teaches its skills, which defeats the point of the type.
    for row in rows:
        if row["resource_type"] in ("project", "assessment") and not row["prerequisites"].strip():
            raise ValueError(
                f"{row['course_id']} is a {row['resource_type']} with no prerequisites"
            )


def main():
    os.makedirs("data", exist_ok=True)
    rows = build_rows()
    validate(rows)

    out_path = os.path.join("data", "catalog.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["resource_type"]] = counts.get(row["resource_type"], 0) + 1
    summary = ", ".join(f"{n} {t}s" for t, n in sorted(counts.items()))
    print(f"Written {len(rows)} resources to {out_path} ({summary})")


if __name__ == "__main__":
    main()
