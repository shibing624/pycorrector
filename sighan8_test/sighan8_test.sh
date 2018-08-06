#!/bin/bash
#
set -eu



start_time=`date +%s`


# # correct
python sighan8_test.py



end_time=`date +%s`

# # evaluate results
java -jar sighan8csc_release1.0/Tool/sighan15csc.jar \
     -i sighan8_result/corrected_result.txt \
     -t sighan8csc_release1.0/Test/SIGHAN15_CSC_TestTruth.txt \
     -o sighan8_result/sighan15_evaluation_test.txt

echo "---------------------  Runtime was `expr $end_time - $start_time` s.  ----------------\n\n
 $(cat sighan8_result/sighan15_evaluation_test.txt)" \
        > sighan8_result/sighan15_evaluation_test.txt 


# # output result summary
head -n 20 sighan8_result/sighan15_evaluation_test.txt 

# # create a tmp file for read and compare result with ground truth
python analyze_result.py 

# # remove tmp file
rm ../pycorrector/data/*.pkl
