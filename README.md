# dv4sa
Sentiment Analysis Based On Combining Term Weighting Scheme And Word Vectors

Usage:<br/>
Install [Anaconda](http://continuum.io/downloads) <br/>
$ wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86_64.sh <br/>
$ bash Anaconda-2.1.0-Linux-x86_64.sh

\begin{tabular}{|l|l|c|c|c||c|c|}{result1}{Results of Sws-b-w2v methods against baselines} 
\hline 
 & Method & PL-sen-05 & PL-subj-5k & MPQA & PL-2K & IMDB\tabularnewline
\hline 
\multirow{3}{*}{} & LCR-uni-w2v & $\textit{78.4}$ & 91.3 & 85.6 & 88.2 & 88.29\tabularnewline
\cline{2-7} 
 & LCR-bi-w2v &  $\underline{\boldsymbol{79.5}}$ & 92.0 & $\boldsymbol{85.7}$ & $\boldsymbol{89.8}$ & 91.22\tabularnewline
\cline{2-7} 
 & OR-uni-w2v & 78.3 & 91.2 & 85.3 & 88.2 & 88.60\tabularnewline
\cline{2-7} 
our & OR-bi-w2v & $\boldsymbol{79.4}$ & 91.9 & 85.5 & $\underline{\boldsymbol{89.8}}$ & $\underline{\boldsymbol{91.56}}$ \tabularnewline
\cline{2-7} 
\multirow{2}{*}{} & WFO-uni-w2v(0.1) & 77.2 & 90.3 & 84.3 & 87.9 & $\textit{89.48}$ \tabularnewline
\cline{2-7} 
 & WFO-bi-w2v & 78.4 & 90.8 & 84.6 & 89.2 & 91.39\tabularnewline
\cline{2-7} 
results & WFO-uni-w2v(0.05) & 77.9 & 91.1 & 85.0 & 88.1 & 89.18\tabularnewline
\cline{2-7} 
\multirow{3}{*}{} & WFO-bi-w2v & 79.3 & 91.5 & 85.4 & 89.5 & $\boldsymbol{91.54}$ \tabularnewline
\cline{2-7} 
 & WFO-uni-w2v(0.01) & 78.1 & 91.1 & 85.3 & $\textit{88.4}$ & 88.66\tabularnewline
\cline{2-7} 
 & WFO-uni-w2v & 79.2 & 91.7 & 85.7 & $\boldsymbol{89.7}$ & $\boldsymbol{91.52}$\tabularnewline
\hline
\hline 
  & Bag-of-words & 77.1 & 91.0 & $\boldsymbol{86.3}$ & 88.1 & 88.6\tabularnewline
\cline{2-7} 
other & cre-tfidf-uni & 77.5 & 91.8 & - & 88.7 & 88.8\tabularnewline
\cline{2-7} 
 & cre-tfidf-bi  & 78.6 & $\boldsymbol{92.8}$ & - & 89.7 & 91.3\tabularnewline
\cline{2-7} 
results & nbsvm-uni & 78.1 & $\boldsymbol{92.4}$ & 85.3 & 87.8 & 88.29\tabularnewline
\cline{2-7} 
 & nbsvm-bi &  $\boldsymbol{79.4}$ & $\underline{\boldsymbol{93.2}}$ &  $\underline{\boldsymbol{86.3}}$ & 89.5 & 91.22\tabularnewline
\hline 
\end{tabular}