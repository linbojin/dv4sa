if [ "$#" -ne 1 ]; then
  echo "Usage: ./run.sh sentiment-analysis-*.py"
  exit 1
fi

fileName=$1
echo "Running python ${fileName}"
time python $fileName | tee "result-of-${fileName}"
echo "Done, result saved in file result-of-${fileName}"
