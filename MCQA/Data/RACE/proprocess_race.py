# cite from https://github.com/nlpdata/strategy/blob/master/code/preprocess.py
import json
import os



def preprocess():
    for f1 in ["dev", "train", "test"]:
        output = []
        for f2 in ["high", "middle"]:
            fl = os.listdir( f1 + "/" + f2)
            for fn in fl:
                with open(f1 + "/" + f2 + "/" + fn, "r") as f:
                    data = json.load(f)
                    d = [[data["article"]], [], f1 + "/" + f2 + "/" + fn]
                    for i in range(len(data["questions"])):
                        q = {}
                        q["question"] = data["questions"][i]
                        q["choice"] = []
                        for j in range(len(data["options"][i])):
                            q["choice"] += [data["options"][i][j]]
                        q["answer"] = q["choice"][ord(data["answers"][i]) - ord("A")]
                        d[1] += [q]
                    output += [d]
        print(f1, len(output))

        # output = hltag(output)
        #
        # print(f1, len(output))

        with open(f1 + ".json", "w") as f:
            json.dump(output, f, indent=2)


if __name__ == '__main__':
    preprocess()