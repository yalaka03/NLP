from ngram import NGramModel
from tokenizer_util import tokenize
import sys
if len(sys.argv) > 1:

    arguments = sys.argv[1:]
    for arg in arguments:
        print(arg)
    good_turing = False
    interpolation = False
    n = 3
    if "g" in arguments:
        good_turing = True
    if "i" in arguments:
        interpolation = True
    file_path = arguments[-1]
    model = NGramModel(file_path, n, interpolation, good_turing)
    model.setup()
    model.train_model()
    sentence = input("input sentence:")
    sentence = tokenize(sentence)[0]
    print(sentence)
    print(model.probability(sentence))

else:
    print("No arguments provided.")
