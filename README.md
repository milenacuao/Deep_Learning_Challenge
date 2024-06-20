# Deep_Learning_Challenge

https://colab.research.google.com/drive/1ePZIX_VvdgyhgTbkKbz7eBedi0775tX9?usp=sharing
https://colab.research.google.com/drive/1w56HjqmMcGRfRrQvsYcXNxr_Fx5eOcKj?usp=sharing


The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

REPORT
Overview.

The primary objective of this analysis is to develop a binary classifier using a deep learning neural network to predict the success rate of applicants seeking funding from Alphabet Soup, a nonprofit organization. The classifier aims to assist Alphabet Soup in identifying potential successful applicants based on their provided features, as Application type, Affiliated sector of industry, Government organization classification, Funding use case, Income classification, Funding amount requested, Effectiveness of fund usage.

About the  Data Preprocessing, to prepare the dataset for model training, several preprocessing steps were undertaken:

-Dropping Unnecessary Columns like Columns deemed irrelevant to the predictive analysis were removed.
-Encoding Categorical Variables, Categorical data were transformed into numerical values using appropriate encoding techniques.
-Data Splitting: The dataset was divided into training and testing subsets to evaluate model performance.

To enhance model performance, several optimization strategies were employed: adjusting input data by modifying the dataset, adding neurons and hidden layers to experiment with different network architectures, testing different activation functions to evaluate their impact on model accuracy, and tuning the number of epochs to balance training time and accuracy.

RESULTS

The performance metrics of the deep learning model during the evaluation phase are:

<img width="723" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/411f7cac-d8ba-458a-a1f1-cf9f982c30b6">

The loss value indicates the degree to which the predictions of the model deviate from the actual results. In this context, the loss function used is likely the binary cross-entropy, which is appropriate for binary classification tasks. A loss of 0.5518 suggests that there is still a considerable error margin between the predicted and actual labels. Lower loss values are generally preferred as they indicate better model performance.
Accuracy measures the proportion of correctly predicted instances out of the total instances. An accuracy of 73.26% means that the model correctly predicts the success of funding applicants approximately 73% of the time. While this is a reasonable accuracy, it falls short of the desired threshold of 75%, indicating room for improvement.
The model underwent 268 training epochs, with each epoch taking around 532 milliseconds, translating to efficient training steps of about 2 milliseconds each.

The model achieves a moderate level of accuracy at 73.26%, which indicates a reasonable predictive capability but also underscores the necessity for further improvements. By exploring additional optimization techniques and possibly alternative models, it may be possible to achieve the target accuracy of 75% or higher. The current performance provides a solid foundation for further development and refinement.

Data Prepocessing
What variable(s) are the target(s) for your model? 
- The target variable(s) for the model is IS_SUCCESSFUL, as it represents the binary classification outcome variable.
What variable(s) are the features for your model?
- The feature variables for the model are all the other columns in the DataFrame,  except IS_SUCCESSFUL.
What variable(s) should be removed from the input data because they are neither targets nor features?
 -EIN or Employee Identification Number does not contain relevant information for our predictive mode
  <img width="923" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/404a9d0b-3eff-47af-836f-fdb7027a0461">

Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
The code sets up a basic deep neural network with two hidden layers and one output layer. The first hidden layer has 8 neurons with ReLU activation, the second hidden layer has 5 neurons with ReLU activation, and the output layer has 1 neuron with a sigmoid activation function, making it suitable for binary classification tasks. The nn.summary() call provides a detailed overview of the model's structure.

- <img width="921" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/6e8aaf7b-fbdc-49ba-baa6-0eaeed33e84f">

Were you able to achieve the target model performance?

<img width="789" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/6b00493e-0b67-4b2c-9395-d847b25825c4">

The model achieves a moderate level of accuracy at 73.26%, which indicates a reasonable predictive capability but also underscores the necessity for further improvements. By exploring additional optimization techniques and possibly alternative models, it may be possible to achieve the target accuracy of 75% or higher. The current performance provides a solid foundation for further development and refinement.

What steps did you take in your attempts to increase model performance?

<img width="907" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/49848148-2b57-4f35-9d6e-731429fe4dfa">

<img width="845" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/7660d275-031c-4ed8-8e75-0fa3b3efc46c">
<img width="938" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/b253c1b3-e30a-4a5b-931f-3fb81753d499">
<img width="689" alt="image" src="https://github.com/milenacuao/Deep_Learning_Challenge/assets/151895571/3bb44f55-4473-44a3-93db-a72f0de3e706">




