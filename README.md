# contextMagnitude

This repo contains simulations published in the following paper:

_Sheahan\*, Luyckx\*, Nelli, Teupe, & Summerfield_ (2021) Neural state space alignment for magnitude generalisation in humans and recurrent networks. _Neuron (in press)_

---

This project trains a simple RNN model to perform a basic relative magnitude reasoning task. The network is trained to classify a given input number (one-hot) as either 'more' or 'less' than the previous number it sees of the same type. The ranges of numbers presented are blocked in time (optional) and their ranges indicated by an explicit context cue (also optional), to see whether the network learns to use the range (context) of the neighbouring numbers to guide its choice.
After training, we compare the representations formed in the hidden unit activations to those observed in neural EEG recordings of human participants performing the same task (human experiments by Fabrice Luyckx). For both humans and RNNs, the neural representations of stimuli are arranged into parallel, magnitude-aligned and context-separated number lines that are both mean-centred (subtractively normalised) and of the same length (divisively normalised), despite the numbers in each context spanning different ranges.

We further manipulate the presence of temporal blocking, context cueing and the quality of the network's short-term memory to isolate the factors leading to context use (behaviourally), as well as the context-separation and normalisation observed in the resultant representations.
