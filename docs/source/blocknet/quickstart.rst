Quickstart
==========
BlockNet quick start guide

How to install
--------------
.. code::

 pip install https://github.com/Text-Analytics/BlockNet/master.zip

How to get an address from a text message
----------------------------------------------------

An example of using the BlockNet library to extract addresses from the "Comment text" column in the sample_data.csv file.  
 - **Step 1**. Create loss function or simulator of the physical process.First, the pandas library is imported and data from the sample_data.csv file is loaded into a DataFrame object. 

 - **Step 2**. Create loss function or simulator of the physical process.Next, an instance of the AddressExtractor class from the factfinder library is created and applied to the "Comment Text" column using the progress_apply() method. This method applies the function to each column element and returns a new DataFrame object with new "Street" columns. 

 - **Step 3**. Create loss function or simulator of the physical process.At the end, the new DataFrame object is displayed using display().

.. code:: python

 import warnings
 warnings.simplefilter('ignore')
 import pandas as pd
 from factfinder import AddressExtractor

 df = pd.read_csv('sample_data.csv', index_col=0)
 model = AddressExtractor()
 df[['Street', 'Score', 'Текст комментария_normalized']] = df['Текст комментария'].progress_apply(lambda t: model.run(t))
 df = df.dropna(subset='Street')

 display(df)

How to classify text messages 
----------------------------------------------------

This code loads a csv file with the name and applies the TextClassifier model from the factfinder library to the "Comment text" column. The model is designed to classify texts into a given number of categories (in this case, 1). The model results in two new columns for each text in the "Comment Text" column: "cats" and "probs". The "cats" column contains the predicted category for each text, and the "probs" column contains the probability that the text belongs to this category.
 - **Step 1**. The torch module is imported, which is the main module for working with neural networks in PyTorch. The pandas module is imported under the alias pd for working with data in table format. The TextClassifier class is imported from the factfinder library, which is used for text data classification. A device_type object is created to specify the use of a graphics processing unit (GPU) for calculations if available. The path to the file ‘FactFinder/data/raw/Admiralteyskiy.csv’ is specified.
 - **Step 2**. An instance of the TextClassifier class is created with the specified parameters: repository_id - model identifier, number_of_categories - number of classification categories, device_type - type of device used. The csv file is read using the read_csv() function from pandas and saved in a DataFrame object named df_predict. The column ‘Comment text’ is renamed to ‘Text’ using the rename() function. Rows containing missing values in the ‘Text’ column are removed using the dropna() function. The first 100 rows are selected using the head() function. The model.run(x) function is applied to the ‘Text’ column using the progress_map() method, which applies the function to each element of the column, and the results are saved in the ‘cats’ and ‘probs’ columns using the DataFrame method.
 - **Step 3**. Output result.

.. code:: python

 import torch
 import pandas as pd
 from factfinder import TextClassifier
 device_type = torch.device('cuda:0')
 path_to_file = 'FactFinder/data/raw/Адмиралтейский.csv'

 model = TextClassifier(
    repository_id="Sandrro/cc-model",
    number_of_categories=1,
    device_type=device_type,
 )
 df_predict = pd.read_csv(path_to_file, sep=';')
 df_predict.rename(columns={'Текст комментария':'Текст'}, inplace=True)
 df_predict = df_predict.dropna(subset='Текст')
 df_predict = df_predict.head(100)
 df_predict[['cats','probs']] = pd.DataFrame(df_predict['Текст'].progress_map(lambda x: model.run(x)).to_list())
 df_predict