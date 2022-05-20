# LDPreference
Local Differential Private Mechanisms for Pereference/Vote Data Analyses

Setup experiments in settings_random.py and run main.py

Implemented (almost) all local differential private mechanisms for preference data aggregation


1. LAPLACE: add Laplace random noises
2. ADDITIVE: use the Additive mechanism
3. SAMPLEX0LAPLACE: sampling one candidate then use the Laplace mechanism 
4. SAMPLEX0BRR: sampling one candidate then use binary randomized response
5. SAMPLEX0SUBSET: sampling one candidate then use Subset mechanism
6. SAMPLEX1PIECEWISE: use the Piecewise mechanism
7. SAMPLEXLAPLACE: optimized weighted-sampling one candidate then use the Laplace mechanism
8. SAMPLEXSUBSET: optimized weighted-sampling one candidate then use the Subset mechanism
