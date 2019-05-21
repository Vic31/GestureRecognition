# Gesture Recognition
In the real world, it is natural to use hand gesture to communicate with other, which can be used with machines too. Widely used area is video games where players like to control their game using hand gestures like hand swing etc. Hence in these fields, gesture recognition becomes a very important project/research area. In our project we are focusing on the static hand gesture recognition using two different algorithms and analyzing the results to determine which is the better one. Basically, this involves classifying a certain image stored in local storage(hence , static) to a particular gesture. We use American Sign Language(ASL) where there are 26 hand gestures mapped to 26 letters of English alphabet set.

2 Approaches: Back-propogation Neural Network (BPNN) and Support Vector Machine (SVM)
BPNN:
1. Propagation
(a) Propagate forward through the network to generate the output value(s)
(b) Calculation of the error term
(c) Propagation of the output activation back through the network using the training pattern target to generate the deltas.
2. Update weights using below steps
(a) The weight’s output delta and input activation are multiplied to find the gradient of the weight.
(b) A ratio (percentage) of the weight’s gradient is subtracted from the weight.
SVM:
• Training a model for a static-gesture recognizer, which is a multi-class classifier that pre- dicts the static sign language gestures.
• Locating the hand in the raw image and feeding this section of the image to the static gesture recognizer.