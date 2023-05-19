with open('test.tsv', 'r') as reader:
    f = reader.readlines()
with open('test2.tsv', 'w') as writer:
    for x in f:
        tag, sentence = x.strip().split("\t")
        writer.write(sentence + "\t" + tag+ '\n')