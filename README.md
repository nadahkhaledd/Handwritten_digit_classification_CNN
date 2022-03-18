# Handwritten_digit_classification_CNN
### Machine Learning model to detect hand written digits using Python.

**Input**
- image of a handwritten digit

**Output**
- label of the digit

**Libraries used**
- Tensor Flow - Keras: to load the MNIST Handwritten digits dataset

***

### How it works
> - Split data into training and testing
> - Normalize data.

> - Start a CNN model to train data. [4 Models]
>> -  Model_1 -> 3 convolutional layers(32,64,128) , 1 fully connected layer (64), one output layer. 
>> - Model_2 -> 2 convolutional layers(64,64) , 2 fully connected layer (64, 32), one output layer. 
>> - Model_3 -> 2 convolutional layers(32,64) , 1 fully connected layer (64), one output layer. 
>> - Model_4 -> 1 convolutional layer(32) , 2 fully connected layers (64, 32), one output layer. 

> - Run chosen model and calculate accuracy.
> - input an image to detect digit.


## Example
![output](https://user-images.githubusercontent.com/63652516/159053107-9fa7f72f-7544-4511-ae62-73eb6c8614fd.PNG)




