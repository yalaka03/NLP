To run perplexity, use the function perplexity and send in a tokenized sentence as argument.

# Generation
1. OOD for n grams gives many zeroes, as to be expected since words like pakisthan or India are not in the literature provided. But other commonly used vocabulary does give some nice values, but seldom make sense.

2. Good turing gives many answers as many of them have same probability since unseen probability is higher, more senseless generations are given.

3. Linear Interpolation does give some better generations, but will terminate to zero because of no smoothing involved.

4. The perplexity scores of train are always less, and alomost same for all sentences, since this is a statistical model and all of them are seen atleast once.

5. The perplexity scores of test is always higher than that of train, showing for unseen the prob is lower so higher perplexity.

6. On both the corpus, good turing has more perplexity on that of the test set. This is nbecause , Good-Turing smoothing tends to assign more probability mass to unseen elements, resulting in higher perplexity for unseen elements compared to linear interpolation. 