# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import argparse
import codecs

import jieba

def parse():
    parser = argparse.ArgumentParser(description = 'this is for tokenize file with one sentence in each line')
    parser.add_argument('-i','--input_file', required = True,
                        help = 'file to be tokenized')
    parser.add_argument('-o','--output_file', required = True,
                        help = 'file to store tokenized setence')
    parser.add_argument('-p','--parallel', default = 4, type = int,
                        help = 'tokenize file in parallel or not')
    parser.add_argument('-d','--dict',
                        help = 'load extra dictionary')
    parser.add_argument('-c','--token_char', default = False,
                        help = 'character-level tokenize sentence or not')
    return parser.parse_args()

def tokenize(args):
    if args.parallel > 1:
        jieba.enable_parallel(args.parallel)

    if args.dict:
        jieba.load_userdict(args.dict)

    file_in  = codecs.open(args.input_file, 'rb', encoding = 'utf-8').read()
    words    = " ".join(jieba.cut(file_in)).replace("\n ", "\n")
    file_out = codecs.open(args.output_file, 'w', encoding = 'utf-8')
    file_out.write(words)
    file_out.close()

def char_tokenize(args):
    file_in = codecs.open(args.input_file, 'rb', encoding = 'utf-8').read()
    words   = file_in.replace("", " ").replace("\n ", "\n")[1:]
    file_out = codecs.open(args.output_file, 'w', encoding = 'utf-8')
    file_out.write(words)
    file_out.close()

def main():
    if len(sys.argv) != 1 and sys.argv[1][0] != "-":
        sys.stdout.write(" ".join(jieba.cut(sys.argv[1])) + "\n")
    else:
        args = parse()

        if args.token_char:
            char_tokenize(args)
        else:  
            tokenize(args)

if __name__ == "__main__":
    main()
