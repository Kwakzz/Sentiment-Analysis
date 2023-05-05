# import datasets library
from datasets import load_dataset

# assign csv file to variable
data_files = "training.1600000.processed.noemoticon.csv"

# return csv file as DatasetDict
twitter_dataset = load_dataset("csv", data_files=data_files)

# remove unwanted columns from dataset
twitter_dataset = twitter_dataset.map(remove_columns=["1467810369", "Mon Apr 06 22:19:45 PDT 2009", "NO_QUERY", "_TheSpecialOne_"])


# rename target column "label"
twitter_dataset = twitter_dataset.rename_column("0", "label")

# rename text column, "Sequence"
twitter_dataset = twitter_dataset.rename_column("@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D", "Sequence")

# sequences with a positive sentiment have been assigned a value of 4
# this function changes the 4s to 1s.
def change_positive_to_1(example):
    if example["label"] == 4:
        example["label"] = 1
    return example
        
# remove all sequences that start with the @. 
# these sequences start with user handles
# we do not want our model to learn this 
twitter_dataset = twitter_dataset.filter(lambda x: x["Sequence"][0] != "@")

# apply the change_positive_to_1 function to all sequences in the dataset
twitter_dataset = twitter_dataset.map(change_positive_to_1)

print(twitter_dataset)


# change the dataset's format to pandas
twitter_dataset.set_format("pandas")

# print the number of sequences in the cleaned dataset
#print(len(twitter_dataset["train"]))

# print the first ten entries
print(twitter_dataset["train"][:10])

# the commented out code below splits the datatset into train and test split
# 20% of the dataset makes up the test split and 80% makes up the train split

#twitter_dataset = twitter_dataset["train"].train_test_split(train_size=0.8, seed=42)

# the commented out code below saves the splits as "twitter-train.csv" and "twitter-test.csv"

#for split, dataset in twitter_dataset.items():
#   dataset.to_csv(f"twitter-{split}.csv", index = False)