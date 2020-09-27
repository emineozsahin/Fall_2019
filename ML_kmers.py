# Emine Ozsahin 
# December 2019
# Packages
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
import shap

# I converted the file genes_csv to a data frame to make a list of differentially expressed genes and extract the conditions
# csv file to data frame
df_csv = pd.read_csv("genes.csv", header=0, index_col=0, quotechar='"', sep=",", na_values=["nan", "-", ".", " ", ""], delimiter=',')
# print(df_csv.isnull().sum()) ## There are missing values, so I filled them with NA because later I need this to exclude missing values from my list
modified_df_csv = df_csv.fillna("NA")
# print(modified_df_csv.isnull().sum()) # There are no missing values now
# I made a list of tuples of condition, level of expression and gene names to use as keys for the dictionary which I made it later on this script called regulatory sequences
DEGs = []
keys = modified_df_csv.to_dict('records')
# Above code generates a list contains dictionaries by giving column names (e.g. Serum Starved Uo regulated) as keys and gene names as values.
# I looped to the dictionaries to access the data
for nested_dictionary in keys:
    # I looped to the keys and values of the dictionary elements
    for key, value in nested_dictionary.items():
        split_key = key.split()
        condition = " ".join(split_key[0:2])
        level = " ".join(split_key[2:4])
        # I eliminate the missing gene names from tuples by using if statement otherwise it makes tuples has condition and expression levels and 'nan' statement as a gene name
        if value != "NA":
            gene_names = value
            key_regulatory_sequences = condition, level, gene_names
            DEGs.append(key_regulatory_sequences)
# print(DEGs)
# print(len(set(DEGs))) #232
# print(len(DEGs)) #232
# I made a dictionary from fasta file. I assigned the chromosome names as keys and sequences as value
chromosome_dict = {}
chromosome_name = ''
sequence = []
with open("eh.fa") as eh_fasta:
    for line in eh_fasta:
        if line.startswith(">") and chromosome_name == '':
            chromosome_name = line.split(' ')[0][1:]
        elif line.startswith(">") and chromosome_name != '':
            chromosome_dict[chromosome_name] = ''.join(sequence)
            chromosome_name = line.split(' ')[0][1:]
            sequence = []
        else:
            sequence.append(line.rstrip())

#I wrote a function for reverse complement the sequences.
def reverse_complement(my_dna):
  reversed_dna = "".join(reversed(my_dna))
  replacement1= reversed_dna.replace("A","t")
  replacement2= replacement1.replace("T","a")
  replacement3= replacement2.replace("G","c")
  replacement4= replacement3.replace("C","g")
  return replacement4.upper()

# I made a dictionary of regulatory sequences as value; and (condition, relative level of expressions and gene names) as key
# I looped to the list of DEGs to find their positions in the chromosome
regulatory_sequences = {}
for gene in DEGs:
    # I opened and looped to the gff3 file to obtain the gene start and end positions, gene orientations, which chromosomes are the genes located
    with open("eh.gff3") as eh_gff:
        gene_name = gene[2]
        for line in eh_gff:
            split_line = line.split()
            # By using if statement I searched the gene names (EHI_) on the non-splited lines and gene string on the splitted lines as splited line index two (according to python) represent the transcript type and assign the gene characteristics to the variables..
            if str(gene_name) in line and "gene" in split_line[2]:
                start = int(split_line[3]) - 1
                end = int(split_line[4]) - 1
                chromosome = split_line[0]
                strand = split_line[6]
                dna_seq_of_DEGs = chromosome_dict[chromosome][int(start):int(end)]
                # To find the potential regulatory sequences of DEGs,
                # I extracted 400 base pairs from upstream and 100 bases from downstream region relative to the translation initiation codon (ATG) and differentiate the positive and negative strands with if statement.
                if strand == "+":
                    location_ATG = dna_seq_of_DEGs.find("ATG")
                    downstream_flanking = start  + location_ATG + 100
                    # one gene ("EHI_172850": 323 993 +) located at the beginning of the chromosome and starts from 323rd bases, there are two more genes too, therefore I used "if conditional statement"  because I cannot substract 400 bases from it.
                    # Therefore three genes potential regulatory sequences are 422, 233, 422 bases long.
                    # I may eliminate them as they might not be usefull or I may try to decrease the lenght of upstream pootential regulatory sequences to 300 bases and 233 bases because if this changes would provide same results, same cluster and prediction, computational time would be reduced.
                    # In case those three genes are not false positive or negatives, keeping them are logical. Therefore, if I would performing this analysis for my study I would try every posible way.
                    if start <= 400:
                        upstream_flanking = 0
                    else:
                        upstream_flanking = start + location_ATG - 400
                    regulatory_seq = chromosome_dict[chromosome][upstream_flanking:downstream_flanking]
                else:
                    name = gene
                    location_CAT = dna_seq_of_DEGs.rfind("CAT")
                    upstream_flanking = start + location_CAT - 100  # This is actually downstream flanking
                    downstream_flanking = start + location_CAT + 400  # This is actually upstream flanking
                    regulatory_seq = reverse_complement(chromosome_dict[chromosome][upstream_flanking:downstream_flanking])
                # I extracted the potential regulatory sequences by using chromosome dictionary.
                # I assigned the keys and values to dictionary regulatory sequences
                # regulatory sequences of negative strands has to be reverse complemented to find the similar sequence patterns when the features clustered
                regulatory_sequences[gene] = regulatory_seq

# a made-up function from class notes on 21st November 2019 was modified. This way I had a dictionary of the kmers sequences as keys and their occurances as value..
def make_kmer(seq, kmer_size):
    kmers = {}
    for initial in range(0, len(seq) - kmer_size + 1, 1):
        kmer = seq [initial: initial + kmer_size]
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1
    return kmers

#I made a kmer list in which each kmer occurance is unique by applying set fuction.
# I changed the values of keys (condition, expression level and gene names) of the regulatory sequences dictionary by giving a new values which are dictinaries
# and their keys are kmer sequence and values are counts. I used different kmer sizes to compare the cross validation scores.
unique_kmers = set()
for name, seq in regulatory_sequences.items():
    kmer_dict = make_kmer(seq, 12)
    for kmer, count in kmer_dict.items():
        unique_kmers.add(kmer)
    regulatory_sequences[name] = kmer_dict
unique_kmer_list = list(unique_kmers)

# I made lists of kmer occurances for the genes (I mean a list for each gene) and append that lists to another list, therefore I had list of lists.
SS_X = []  #List of list kmer counts for serum starved
SS_y = []  #list_of_conditions for serum starved
SR_X = []  #List of list kmer counts for serum replenishment
SR_y = []  #list_of_conditions for serum replenishment
for names, kmr_dict in regulatory_sequences.items():
    kmer_count = []
    # Each kmer in the list was searched in the kmer count dictinary (nested dictinary of the regulatory_sequences dictionary) which I looped its keys and values at the above code.
    for kmer in unique_kmer_list:
        # If the kmer does not excist in the kmer count dictionary of certain genes than occurance was assigned as 0 otherwise it is assigned as their orginal occurance count.
        if kmer in kmr_dict.keys():
            kmer_count.append(kmr_dict[kmer])
        else:
            kmer_count.append(0)
    # I separate the conditions to analyse them separately.
    if "Serum Starved" in names:
        SS_X.append(kmer_count)
        SS_y.append(" ".join(list(names[0:2])))
    else:
        SR_X.append(kmer_count)
        SR_y.append(" ".join(list(names[0:2])))

#import numpy as np
#SS_x = np.asarray(SS_X)
#print(SS_x.shape) #197, 92727

# For machine learning part kmer occurances were labeled with the condition and expression levels.
# Classification made based on kmers occurances.


# This function is taken from stackoverflow: https://stackoverflow.com/questions/18703136/proximity-matrix-in-sklearn-ensemble-randomforestclassifier.
def proximityMatrix(model, X, normalize=True):
    terminals = model.apply(X)
    nTrees = terminals.shape[1]
    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)
    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)
    if normalize:
        proxMat = proxMat / nTrees
    return proxMat

# I made a function to make a variance filteration of data, calculate the cross validation scores, plot the clusters, filter the kmers, make a model
def var_filter_model_plot(X, y, kmer_list):
    # In this step I filtered the columns if their variances are 0 which means that kmer sequence on that column is not a selective feature to differentiate the sequences.
    X_varFilt = VarianceThreshold().fit_transform(X)
    # To be able to see which kmer sequence has more impact (positively or negatively) on the gene expression kmers used to label the shap values.
    # kmers are also filtered as they removed if their occurrences removed at the previous filtration.
    X_filter = VarianceThreshold().fit(X)
    filter_kmer = X_filter.transform([kmer_list])[0]
    # In this step a model which is a Extra Trees Classifier with 128 branches was chosen and trained to classify the kmers features depends on their occurances in the sequences.
    model = ExtraTreesClassifier(128)
    model.fit(X_varFilt, y)
    # For projection I used TSNE with the proximity matrix because without distance analyse.
    # Proximity matrix do not change the data, but calculates the distances and selects better distances randomly.
    # Therefore its projection may not show a real result but may project them better as nice to see.
    proxy = 1 - proximityMatrix(model, X_varFilt, normalize=True)
    X_transformed_proxy = TSNE(n_components=2, perplexity=6).fit_transform(proxy)
    # and visualized the data.
    sns.scatterplot(x=X_transformed_proxy[:, 0],
                    y=X_transformed_proxy[:, 1],
                    hue=y)
    plt.show()
    plt.close()
    # Being able to see the clusters of two group nicely seperated by using proxy distances does not mean the classification are made well.
    # To understand if the model which we chosen is good, cross validation scores are calculated.
    # cross validation scores were calculated by splitting the data by 10 folds to understand how well model were fit to our data.
    csv_score = cross_val_score(ExtraTreesClassifier(128),
                                X=X_varFilt,
                                y=y,
                                cv=StratifiedKFold(10), n_jobs=2)
    mean_score = np.mean(csv_score)
    stddev_score = np.std(csv_score, ddof=1)

    print("Variation Filter shape: " + str(X_varFilt.shape))
    print("Cros validation score: " + str(mean_score), "+/-", str(stddev_score))
    print("[0]: X_varFilt,  [1]: filter_kmer, [2]: model")
    return X_varFilt, filter_kmer, model

SS_cluster = var_filter_model_plot(SS_X, SS_y, unique_kmer_list)
SR_cluster = var_filter_model_plot(SR_X, SR_y, unique_kmer_list)

# I made a funtion to explain the model with shap values. T
# the parameter 10 that I chose in the function means top 10 fetures.
# To be able to see the labels which are kmer sequences, a filtered kmers sequence list is also provided to the function
def shap_explainer(X_VarFilt, filter_kmer, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_VarFilt)
    # display the scores
    shap.summary_plot(shap_values[0],
                      X_VarFilt,
                      filter_kmer,
                      10)
    plt.show()
    plt.close()
    return shap_values

shap_explainer(SS_cluster[0], SS_cluster[1], SS_cluster[2])
shap_explainer(SR_cluster[0], SR_cluster[1], SR_cluster[2])

'''
# 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap as sh

X_varFilt = VarianceThreshold().fit_transform(SS_X)

X_transformed_proxy = TSNE(n_components=2, perplexity=6).fit_transform(X_varFilt)

sns.scatterplot(x=X_transformed_proxy[:, 0],
                    y=X_transformed_proxy[:, 1],
                    hue= SS_y)
plt.show()
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X_varFilt, SS_y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
model.fit(X_train, y_train)

csv_score = cross_val_score(model,
                                X=X_test,
                                y=y_test,
                                cv=StratifiedKFold(10), n_jobs=2)
mean_score = np.mean(csv_score)
stddev_score = np.std(csv_score, ddof=1)
print(str(mean_score) + "+\-" + str(stddev_score))

# Random Forest 0.55+\-0.10540925533894598

explainer = sh.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(shap_values)
'''
