%{
 This script takes saved the RDMs of artificial neural network activations
 from several different RNN models, and evaluates how strongly each activation
 RDM correlates with model RDMs.
 Model RDMs include:
 - a magnitude context RDM that varies between code magnitude in absolute
 vs relative terms within a normalised range.
 - a pure between-context context RDM

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 19/02/2020
 Issues: N/A
 Notes: - model RDM scripts and design by Stephanie Nelli & Fabrice Luyckx
%}

% ----------------------------------------------------------------------- %
%% Load in all our different activation RDMs 





% ----------------------------------------------------------------------- %
%% Fit model RDMs to each activation RDM in turn and save summary metrics for the fit
% e.g. normalisation value, Beta value for correlation with between-context RDM





% ----------------------------------------------------------------------- %
%% Plot how our summary metrics changed as a function of different RNN model parameters
% e.g. width of RNN recurrent layer, width of hidden layer, BPTT length...




% ----------------------------------------------------------------------- %



