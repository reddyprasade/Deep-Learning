# Recurrent neural network (RNN) 
* Recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. 
* This allows it to exhibit temporal dynamic behavior. 
* Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs. 
* This makes them applicable to tasks such as 
      * Unsegmented, 
      * Connected handwriting recognition or 
      * speech recognition.
      * Text Clasfication 
* The term “recurrent neural network” is used indiscriminately to refer to two broad classes of networks with a similar general structure, where one is finite impulse and the other is infinite impulse. 
* Both classes of networks exhibit temporal dynamic behavior.
* A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that can not be unrolled.
* Both finite impulse and infinite impulse recurrent networks can have additional stored states, and the storage can be under direct control by the neural network.
* The storage can also be replaced by another network or graph, if that incorporates time delays or has feedback loops. 
* Such controlled states are referred to as gated state or gated memory, and are part of `long short-term memory networks (LSTMs)` and `gated recurrent units`.This is also called Feedback Neural Network.

