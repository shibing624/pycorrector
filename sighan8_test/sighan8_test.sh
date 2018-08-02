#!/bin/bash
#

start_time=`date +%s`



python sighan8_test.py



end_time=`date +%s`

echo "---------------------  Runtime was `expr $end_time - $start_time` s.  ----------------\n\n
 $(cat sighan8_result/sighan8_evaluation_test.txt)" \
        > sighan8_result/sighan8_evaluation_test.txt 


java -jar sighan8csc_release1.0/Tool/sighan15csc.jar \
     -i sighan8_result/corrected_result.txt \
     -t sighan8csc_release1.0/Test/SIGHAN15_CSC_TestTruth.txt \
     -o sighan8_result/sighan8_evaluation_test.txt

head -n 20 sighan8_result/sighan8_evaluation_test.txt 

python analyze_result.py 
