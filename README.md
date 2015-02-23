# dv4sa
Sentiment Analysis Based On Combining Term Weighting Scheme And Word Vectors

version 1.0.0:
+ average feature vectors
+ tf-idf feature vectors
+ credibility adjustment tf-idf feature vectors
+ bag of centroids feature vectors

verison 1.1.0:<br/>
for PL-sent dataset
<table>
  <tr>
    <th>Weight Scheme</th><th>Accuracy</th>
  </tr>
  <tr>
    <td>bag of words sklearn</td><td>77.1%</td>
  </tr>
  <tr>
    <td>tf-idf-uni</td><td>77.2%</td>
  </tr>
  <tr>
    <td>cre-tfidf-uni</td><td>77.5%</td>
  </tr>
  <tr>
    <td>word vecs average</td><td>77.3%</td>
  </tr>
  <tr>
    <td>word vecs tf-idf</td><td>77.2%</td>
  </tr>
  <tr>
    <td>word vecs cre tf-idf</td><td>76.9%</td>
  </tr>
  <tr>
    <td>word vecs bag of centroids</td><td>75.3% (c=0.1)</td>
  </tr>
  <tr>
    <td>word vecs cre tfidf bag of centroids</td><td>76.2% (c=0.1)</td>
  </tr>
</table>

for PL-subj dataset
<table>
  <tr>
    <th>Weight Scheme</th><th>Accuracy</th>
  </tr>
  <tr>
    <td>bag of words sklearn</td><td>90.8%</td>
  </tr>
  <tr>
    <td>tf-idf-uni</td><td>90.8%</td>
  </tr>
  <tr>
    <td>cre-tfidf-uni</td><td>91.2%</td>
  </tr>
  <tr>
    <td>word vecs average</td><td>90.9%</td>
  </tr>
  <tr>
    <td>word vecs tf-idf</td><td>90.5%</td>
  </tr>
  <tr>
    <td>word vecs cre tf-idf</td><td>90.5%</td>
  </tr>
  <tr>
    <td>word vecs bag of centroids</td><td>90.7% (c=0.1)</td>
  </tr>
  <tr>
    <td>word vecs cre tfidf bag of centroids</td><td>91.0% (c=0.1)</td>
  </tr>
</table>

Usage:<br/>
Install [Anaconda](http://continuum.io/downloads) <br/>
$ wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86_64.sh <br/>
$ bash Anaconda-2.1.0-Linux-x86_64.sh

./run.sh