% make erc matlab wrapper
% Ming-Yu Liu 04/12/2010
clear all;
clc;
%restoredefaultpath;

% mex -v -c pub_ers/ERS/MERCCInput.cpp
% mex -v -c pub_ers/ERS/MERCOutput.cpp
% mex -v -c pub_ers/ERS/MERCDisjointSet.cpp
% mex -v -c pub_ers/ERS/MERCFunctions.cpp
% mex -v -c pub_ers/ERS/MERCLazyGreedy.cpp
% mex pub_ers/ERS/mex_ers.cpp MERCCInput.o* MERCOutput.o* MERCDisjointSet.o* MERCFunctions.o* MERCLazyGreedy.o* -outdir pub_ers
% 
% delete *.o*

mex -c HSERS/MERCCInput.cpp
mex -c HSERS/MERCOutput.cpp
mex -c HSERS/MERCDisjointSet.cpp
mex -c HSERS/MERCFunctions.cpp
mex -c HSERS/MERCLazyGreedy.cpp
mex HSERS/mex_MSERS.cpp MERCCInput.o* MERCOutput.o* MERCDisjointSet.o* MERCFunctions.o* MERCLazyGreedy.o*

delete *.o*