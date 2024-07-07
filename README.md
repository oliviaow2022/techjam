# techjam
This repository was built for the Tiktok Techjam 2024 competition under Track 7: On Unleashing Potential in Machine Learning Infrastructure.

## Problem Statement

In the rapidly evolving landscape of technology, machine learning (ML) stands at the forefront of innovation. As we step into 2024, the potential for ML to revolutionize industries is immense. This hackathon aims to explore and create new opportunities that leverage the power of ML infrastructure to solve complex problems, enhance efficiency, and drive growth.

## Our Inspiration


Active learning is based on the notion that not every data point is equally valuable when training a model. In large datasets with many unlabeled data points, conventionally labelling them all can be expensive and extremely time-consuming, especially since labelling requires knowledgeable annotators specific to each use case.

For instance, in image classification, assuming a cost of USD $0.10 per image, labelling 100,000 images could amount to around USD $10,000. Depending on the requirements and use cases of the company, these costs may fluctuate.

Addressing this pain point by focusing on the most informative samples can save both time and money.

Our inspiration to develop Labella was driven by the desire to enhance the efficiency of data labelling processes. We aimed to:
1. Reduce Costs:
    - Minimise labelling expenses by selecting only the most valuable data points for annotation.
    - Avoid unnecessary labelling of redundant or less informative samples.
2. Save Time:
    - Speed up the labelling process by concentrating on the most uncertain or diverse data points.
    - Enable quicker model improvements by iteratively training on the most informative samples.
3. Optimise Resource Allocation:
    - Efficiently utilise expert annotators by directing them to label the most critical data points.
    - Reduce the workload on annotators by decreasing the volume of data needing annotation.

Labella aims to provide a solution that leverages active learning to streamline the data labelling process, making it more cost-effective, time-efficient, and ultimately leading to better-performing models.

## What it does

This platform enables users to create various machine learning projects, including image classification (single and multi-label) and sentiment analysis. In each project, users can upload their custom dataset or choose from existing ones. In image classification, after training their model, the active learning model scores and identifies the most informative images that need to be labelled by the user. Users then label these images and re-train the model with the newly labelled images,enhancing the accuracy and other aspects of the model.

## How we built it

Our team focused on two main aspects: developing a scalable, user-friendly platform and creating an active learning cycle for image classification (single and multi-label) and sentiment analysis. We integrated these into a comprehensive pipeline that involves labelling, training, and re-labelling until the user is satisfied with their model’s performance. 

## Challenges we ran into

| Challenge | Solution |
|-----------|----------|
| How to use an active learning cycle for different kinds of projects. | Sentiment analysis and image classification are different and therefore have different kinds of active learning cycles. |
| Choosing the most suitable active learning algorithms that balance accuracy and time. | Conducted extensive research and experimented with various algorithms (such as DeBERTa-v3 model, Resnet) to identify those that best suit the 3 different kinds of machine learning projects. |
| Ensuring the platform can handle large-scale datasets. | Leveraged cloud computing resources (AWS S3 buckets for datasets). |
| Difficulty in creating an intuitive user flow when integrating the GUI with the active learning process. | Experimented with different user flows to refine user flows. |

## Accomplishments that we're proud of

We have developed an intuitive and user-friendly interface for Labella, making the labelling process seamless and efficient for users.We also ensured that Labella can handle large datasets efficiently, maintaining performance and responsiveness. Labella can also support various use cases, such as sentiment analytics and image classification tasks.

## What we learned

Our key takeaways when developing Labella::
1. A user-friendly interface and seamless experience are critical for adoption and user satisfaction.
2. Customisation options are key to meeting the different requirements of different users.
3. Developing a platform that supports various data types and machine learning tasks required flexible and scalable solutions.
4. Each use case, such as image classification and sentiment analysis, presented unique challenges and required tailored approaches.
5. As datasets grow, ensuring the platform remains fast and responsive is a significant challenge.

## What's next for Labella

With further development, we envision several enhancements and expansions for Labella.
1. Pretrained model selection
    - Provide users with a wider range of pretrained models to choose from for various tasks
    - Allow more customisation of pretrained models to fit specific user needs and datasets	
2. Broader range of use cases
    - Expand active learning capabilities to object detection tasks and semantic and instance segmentation tasks
3. Enhanced user experience
    - Provide detailed analytics and insights on the labelling progress
    - Facilitate collaborative labelling efforts by supporting multiple annotators working on the same project with seamless integration and version control.
    - Allow users to view the status and estimated time to completion of training jobs.

## Built with
- Front-end:  NextJS
- Back-end: Python Flask
- ML: PyTorch, ModAL, Scikit-learn
- APIs: Swagger API
- Cloud: AWS S3 buckets
