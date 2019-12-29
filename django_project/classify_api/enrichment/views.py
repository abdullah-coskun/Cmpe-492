from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
# Import skope-rules
from rest_framework.views import APIView
from skrules import SkopeRules
# Import skope-rules
from skrules import SkopeRules


# Import libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib import cm
import numpy as np
from sklearn.metrics import confusion_matrix
from IPython.display import display
import os
import csv


#Import basic NLTK tooling
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

#Import benepar parser
import benepar
benepar.download('benepar_en2')

#Tqdm, for the progress bar
from tqdm import tqdm

#Spacy
import spacy
nlp = spacy.load("en_core_web_sm")

from spacy.lemmatizer import Lemmatizer
#from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = nlp.Defaults.create_lemmatizer()
#lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

def getDependenciesFeaturesSets(type):
    """
    Retrieves the set of linguistic features appropriate, based on the name of the feature set given as input.
    Nine groups of features, calculated in the notebook 04_ling_stats_calculator, are considered in this function:
    1. single dependencies
    2. combinations of 2 dependencies
    3. combinations of 3 dependencies
    4. single branches
    5. combinations of 2 branches
    6. combinations of 3 branches
    7. sequences of POSdep
    8. combinations of 2 sequences of POSdep
    9. combinations of 3 sequences of POSdep
    The following feature sets are defined:
    all: the top 10 features for each of groups 1-6, no features from group 7 (TOGETHER WITH ext IS USED IN THE PAPER AS FS3)
    sd: the single dependencies that appeared at least once in the top 10 features of groups 1-3, no feat from group 7 (USED IN THE PAPER AS FS1)
    sdsb: same as sd but for groups 1-6
    sdsb8sel02: the features with delta>0.2 in sdsb (TOGETHER WITH ext IS USED IN THE PAPER AS FS2)
    seq: the features in sdsb8sel02 + the top 10 features of groups 7-9
    ev: a merge of all the previous ones
    two: only dobj and nummod
    FinalSel: a final selection of 11 features (extended with 4 additional ones, outside this function is USED IN THE PAPER AS THE FINAL SET)
    N.B. the suffix ext refers to additional features added outside of this function
    @param type: one of the names of the feature sets above defined
    @return: the appropriate lists of features
    """
    if type == 'all' or type == 'allext':
        significant_dependencies = ['dobj', 'nummod', 'acl', 'amod', 'auxpass',
                                    'advmod', 'nsubjpass', 'nsubj', 'nmod', 'advcl']
        significant_2dependencies = [['ROOT', 'nummod'], ['aux', 'nummod'], ['det', 'nummod'],
                                     ['nummod', 'punct'], ['ROOT', 'dobj'], ['aux', 'dobj'],
                                     ['nummod', 'pobj'], ['nsubj', 'dobj'], ['nsubj', 'nummod'], ['dobj', 'pobj']]
        significant_3dependencies = [['ROOT', 'nummod', 'punct'], ['aux', 'ROOT', 'nummod'], ['aux', 'nummod', 'punct'],
                                     ['det', 'ROOT', 'nummod'], ['det', 'nummod', 'punct'], ['det', 'aux', 'nummod'],
                                     ['ROOT', 'det', 'dobj'], ['nsubj', 'det', 'dobj'], ['aux', 'det', 'dobj'],
                                     ['nsubj', 'aux', 'dobj']]

        significant_branches = ['ROOT_dobj_det', 'ROOT_dobj_acl_aux', 'ROOT_dobj_acl_dobj_det',
                                'ROOT_prep_pobj_det', 'ROOT_auxpass', 'ROOT_prep_pobj_compound', 'ROOT_nsubj',
                                'ROOT_ccomp_aux', 'ROOT_nsubj_nummod', 'ROOT_prep_pobj_nummod']

        significant_2branches = [['ROOT_dobj_det', 'ROOT_nsubj_det'],
                                 ['ROOT_aux', 'ROOT_dobj_det'],
                                 ['ROOT_dobj_det', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_aux '],
                                 ['ROOT_punct', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_dobj_acl_aux'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_dobj_det'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_punct'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_nsubj_det'],
                                 ['ROOT_prep_pobj_det', 'ROOT_punct']]

        significant_3branches = [['ROOT_aux', 'ROOT_dobj_det', 'ROOT_nsubj_det'],
                                 ['ROOT_dobj_det', 'ROOT_nsubj_det', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_dobj_det', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_aux', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_punct', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_aux', 'ROOT_nsubj_det'],
                                 ['ROOT_nsubj_det', 'ROOT_punct', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_dobj_acl_aux', 'ROOT_dobj_det'],
                                 ['ROOT_aux', 'ROOT_dobj_acl_aux', 'ROOT_punct'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_dobj_det', 'ROOT_punct']]

        significant_sequences = []

    elif type == 'sd' or type == 'sdext':
        significant_dependencies = ['dobj', 'nummod', 'acl', 'amod', 'auxpass',
                                    'advmod', 'nsubjpass', 'nsubj', 'nmod', 'aux', 'pobj', 'prep', 'det', 'punct']
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = []
        significant_2branches = []
        significant_3branches = []
        significant_sequences = []
    elif type == 'sdsb' or type == 'sdsbext':
        significant_dependencies = ['dobj', 'nummod', 'acl', 'amod', 'auxpass',
                                    'advmod', 'nsubjpass', 'nsubj', 'nmod', 'aux', 'pobj', 'prep', 'det', 'punct']
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = ['ROOT_dobj_det', 'ROOT_dobj_acl_aux', 'ROOT_prep_pobj_det', 'ROOT_acomp_xcomp_aux',
                                'ROOT_nsubjpass_det',
                                'ROOT_dobj_acl_dobj_det', 'ROOT_prep_pobj_compound', 'ROOT_acomp_xcomp_dobj_det',
                                'ROOT_nsubj_det', 'ROOT_dobj_acl_aux']
        significant_2branches = []
        significant_3branches = []
        significant_sequences = []
    elif type == 'sdsb8sel02' or type == 'sdsb8sel02ext':
        significant_dependencies = ['dobj', 'acl', 'prep', 'det', 'pobj', 'aux', 'nsubj', 'punct']
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = ['ROOT_aux', 'ROOT_dobj_det', 'ROOT_punct', 'ROOT_nsubj_det']
        significant_2branches = []
        significant_3branches = []
        significant_sequences = []
    elif type == 'seq' or type == 'seqext':
        significant_dependencies = ['dobj', 'acl', 'prep', 'det', 'pobj', 'aux', 'nsubj', 'punct']
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = ['ROOT_aux', 'ROOT_dobj_det', 'ROOT_punct', 'ROOT_nsubj_det']
        significant_2branches = []
        significant_3branches = []
        significant_sequences = ['NNdobj', 'TOaux', 'NNPnsubj', 'RBadvmod', 'VBxcomp', 'VBauxpass', 'CDnummod',
                                 'VBROOT', 'NNdobj_INprep', 'VBROOT_DTdet', 'DTdet_NNdobj', 'MDaux_VBROOT',
                                 'JJacomp_TOaux', 'NNPnsubj_MDaux',
                                 'MDaux_VBROOT_DTdet', 'VBROOT_DTdet_NNdobj', 'JJacomp_TOaux_VBxcomp',
                                 'VBROOT_JJacomp_TOaux', 'MDaux_VBROOT_DTdet_NNdobj', 'MDaux_VBROOT_JJacomp_TOaux',
                                 'VBROOT_JJacomp_TOaux_VBxcomp',
                                 'MDaux', 'DTdet', 'NNnsubj', 'INprep']
    elif type == 'ev' or type == 'evext':
        significant_dependencies = ['dobj', 'nummod', 'acl', 'amod', 'auxpass',
                                    'advmod', 'nsubjpass', 'nsubj', 'nmod', 'advcl', 'prep', 'det', 'pobj', 'aux',
                                    'punct']
        significant_2dependencies = [['ROOT', 'nummod'], ['aux', 'nummod'], ['det', 'nummod'],
                                     ['nummod', 'punct'], ['ROOT', 'dobj'], ['aux', 'dobj'],
                                     ['nummod', 'pobj'], ['nsubj', 'dobj'], ['nsubj', 'nummod'], ['dobj', 'pobj']]
        significant_3dependencies = [['ROOT', 'nummod', 'punct'], ['aux', 'ROOT', 'nummod'], ['aux', 'nummod', 'punct'],
                                     ['det', 'ROOT', 'nummod'], ['det', 'nummod', 'punct'], ['det', 'aux', 'nummod'],
                                     ['ROOT', 'det', 'dobj'], ['nsubj', 'det', 'dobj'], ['aux', 'det', 'dobj'],
                                     ['nsubj', 'aux', 'dobj']]

        significant_branches = ['ROOT_dobj_det', 'ROOT_acomp_xcomp_aux', 'ROOT_nsubjpass_det',
                                'ROOT_acomp_xcomp_dobj_det', 'ROOT_nsubj_det',
                                'ROOT_dobj_acl_aux',
                                'ROOT_dobj_acl_dobj_det',
                                'ROOT_prep_pobj_det',
                                'ROOT_auxpass',
                                'ROOT_prep_pobj_compound',
                                'ROOT_nsubj',
                                'ROOT_ccomp_aux',
                                'ROOT_nsubj_nummod',
                                'ROOT_prep_pobj_nummod', 'ROOT_aux', 'ROOT_punct']

        significant_2branches = [['ROOT_dobj_det', 'ROOT_nsubj_det'],
                                 ['ROOT_aux', 'ROOT_dobj_det'],
                                 ['ROOT_dobj_det', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_aux '],
                                 ['ROOT_punct', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_dobj_acl_aux'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_dobj_det'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_punct'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_nsubj_det'],
                                 ['ROOT_prep_pobj_det', 'ROOT_punct']]

        significant_3branches = [['ROOT_aux', 'ROOT_dobj_det', 'ROOT_nsubj_det'],
                                 ['ROOT_dobj_det', 'ROOT_nsubj_det', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_dobj_det', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_aux', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_punct', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_aux', 'ROOT_nsubj_det'],
                                 ['ROOT_nsubj_det', 'ROOT_punct', 'ROOT_punct'],
                                 ['ROOT_aux', 'ROOT_dobj_acl_aux', 'ROOT_dobj_det'],
                                 ['ROOT_aux', 'ROOT_dobj_acl_aux', 'ROOT_punct'],
                                 ['ROOT_dobj_acl_aux', 'ROOT_dobj_det', 'ROOT_punct']]

        significant_sequences = ['NNdobj', 'TOaux', 'NNPnsubj', 'RBadvmod', 'VBxcomp', 'VBauxpass', 'CDnummod',
                                 'VBROOT', 'NNdobj_INprep', 'VBROOT_DTdet', 'DTdet_NNdobj', 'MDaux_VBROOT',
                                 'JJacomp_TOaux', 'NNPnsubj_MDaux',
                                 'MDaux_VBROOT_DTdet', 'VBROOT_DTdet_NNdobj', 'JJacomp_TOaux_VBxcomp',
                                 'VBROOT_JJacomp_TOaux', 'MDaux_VBROOT_DTdet_NNdobj', 'MDaux_VBROOT_JJacomp_TOaux',
                                 'VBROOT_JJacomp_TOaux_VBxcomp',
                                 'MDaux', 'DTdet', 'NNnsubj', 'INprep']
    elif type == 'two':
        significant_dependencies = ['dobj', 'nummod']
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = []
        significant_2branches = []
        significant_3branches = []
        significant_sequences = []

    elif 'FinalSel' in type:
        significant_dependencies = ['nsubj', 'dobj', 'nummod', 'amod', 'acl', 'nmod', 'auxpass', 'nsubjpass', 'prep',
                                    'pobj', 'advmod']
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = []
        significant_2branches = []
        significant_3branches = []
        significant_sequences = []

    else:
        significant_dependencies = []
        significant_2dependencies = []
        significant_3dependencies = []
        significant_branches = []
        significant_2branches = []
        significant_3branches = []
        significant_sequences = []

    return significant_dependencies, significant_2dependencies, significant_3dependencies, significant_branches, significant_2branches, significant_3branches, significant_sequences


def get_all_paths(node, h, max_h):
    """
    Calculates all the dependencies paths (branches) in a requirement dependency tree up to an height of max_h
    @param node: the root of the tree
    @param h: the initial height (typically 0)
    @return: a list of strings representing paths
    """
    if node.n_lefts + node.n_rights == 0 or h == max_h:
        return [node.dep_]
    return [
        node.dep_ + '_' + str(path) for child in node.children for path in get_all_paths(child, h + 1, max_h)
    ]


def createEnrichedDataset(data, new_file_name, dep_feat_type):
    """
    Creates a <new_file_name>.csv file with dataset data enriched with the features in dep_feat_type
    @param data: the original dataset
    @param new_file_name: the name of the new dataset
    @param dep_feat_type: the type of feature sets, see function getDependenciesFeaturesSets for a description
    """

    columns_to_keep = ['ProjectID', 'RequirementText', 'Class', 'IsFunctional', 'IsQuality']
    for c in data.columns:
        if not c in columns_to_keep:
            data = data.drop(c, axis=1)

    # the presence of ext in dep_feat_type indicates that we want to extend the features obtained from function getDependenciesFeaturesSets
    # with additional features from literature
    if "ext" in dep_feat_type:
        data['Length'] = 0
        idx = 0
        for x in data['RequirementText']:
            data.at[idx, 'Length'] = len(x)
            idx = idx + 1
        data['AdvMod'] = 0
        data['AMod'] = 0
        data['AComp'] = 0
        data['DTreeHeight'] = 0

    if "FinalSel" in dep_feat_type:
        data['AComp'] = 0

    # get the features to use
    significant_dependencies, significant_2dependencies, significant_3dependencies, significant_branches, significant_2branches, significant_3branches, significant_sequences = getDependenciesFeaturesSets(
        dep_feat_type)

    # init columns of the dataframe for the appropriate features
    for d in significant_dependencies:
        data[d] = 0
    for c in significant_2dependencies:
        data[c[0] + '+' + c[1]] = 0
    for t in significant_3dependencies:
        data[t[0] + '+' + t[1] + '+' + t[2]] = 0
    for d in significant_branches:
        data[d] = 0
    for c in significant_2branches:
        data[c[0] + '+' + c[1]] = 0
    for t in significant_3branches:
        data[t[0] + '+' + t[1] + '+' + t[2]] = 0
    for s in significant_sequences:
        data[s] = 0

    # loop for all rows in the original dataset
    idx = 0
    for req in tqdm(data['RequirementText'], desc='spaCy analysis', position=0):
        token = tokenizer.tokenize(req)
        doc = nlp(req)
        printed = False
        maxHeight = 1
        req_dep = []
        req_tagged_seq = ''
        for t in doc:
            req_dep.append(t.dep_)
            req_tagged_seq = req_tagged_seq + t.tag_ + t.dep_ + "_"

        dep_br_lists = [get_all_paths(sent.root, 0, 15) for sent in doc.sents]
        dep_br = []
        for l in dep_br_lists:
            if l != ['ROOT']:
                dep_br = dep_br + l
        dep_br.sort()

        if "ext" in dep_feat_type:
            for sent in doc.sents:
                for token in sent:
                    height = 1
                    for t in token.ancestors:
                        height = height + 1
                    if height > maxHeight:
                        maxHeight = height

                    # TODO: Limit to Root verb?
                    if token.dep_ == 'advmod' and token.head.pos_ == 'VERB' and token.pos_ == 'ADV':
                        # print('Pattern 1: VB', token.head, '->', token.dep_, '-> RB', token.text)
                        data.at[idx, 'AdvMod'] = data.at[idx, 'AdvMod'] + 1

                    if token.dep_ == 'amod' and token.head.pos_ == 'NOUN' and token.pos_ == 'ADJ':
                        # Could be made stronger by making the head traversal recursive
                        if token.head.dep_ == 'nsubj':
                            continue
                        # print('Pattern 2: NN', token.head, '->', token.dep_, '-> ADJ', token.text)
                        data.at[idx, 'AMod'] = data.at[idx, 'AMod'] + 1

                    if token.dep_ == 'acomp' and token.head.pos_ == 'VERB' and token.pos_ == 'ADJ':
                        if token.text == 'able':
                            continue
                        # print('Pattern 3: VB', token.head, '->', token.dep_, '-> ADJ', token.text)
                        data.at[idx, 'AComp'] = data.at[idx, 'AComp'] + 1

            # Max height of the dependency tree of a sentence of a given requirement
            data.at[idx, 'DTreeHeight'] = maxHeight

        if "FinalSel" in dep_feat_type:
            for sent in doc.sents:
                for token in sent:
                    height = 1
                    for t in token.ancestors:
                        height = height + 1
                    if height > maxHeight:
                        maxHeight = height

                    if token.dep_ == 'acomp' and token.head.pos_ == 'VERB' and token.pos_ == 'ADJ':
                        if token.text == 'able':
                            continue
                        # print('Pattern 3: VB', token.head, '->', token.dep_, '-> ADJ', token.text)
                        data.at[idx, 'AComp'] = data.at[idx, 'AComp'] + 1

        for d in significant_dependencies:
            if d in req_dep:
                data.at[idx, d] = data.at[idx, d] + 1
        for c in significant_2dependencies:
            if c[0] in req_dep and c[1] in req_dep:
                data.at[idx, c[0] + '+' + c[1]] = data.at[idx, c[0] + '+' + c[1]] + 1
        for t in significant_3dependencies:
            if t[0] in req_dep and t[1] in req_dep and t[2] in req_dep:
                data.at[idx, t[0] + '+' + t[1] + '+' + t[2]] = data.at[idx, t[0] + '+' + t[1] + '+' + t[2]] + 1

        for d in significant_branches:
            if d in dep_br:
                data.at[idx, d] = data.at[idx, d] + 1
        for c in significant_2branches:
            if c[0] in dep_br and c[1] in dep_br:
                data.at[idx, c[0] + '+' + c[1]] = data.at[idx, c[0] + '+' + c[1]] + 1
        for t in significant_3branches:
            if t[0] in dep_br and t[1] in dep_br and t[2] in dep_br:
                data.at[idx, t[0] + '+' + t[1] + '+' + t[2]] = data.at[idx, t[0] + '+' + t[1] + '+' + t[2]] + 1

        for s in significant_sequences:
            if s in req_tagged_seq:
                data.at[idx, s] = data.at[idx, s] + 1

        idx = idx + 1

    if "ext" in dep_feat_type:
        # parser = benepar.Parser("benepar_en2")
        data['Modal'] = 0
        data['Adjective'] = 0
        data['Noun'] = 0
        data['Adverb'] = 0
        data['Cardinal'] = 0
        data['CompSupAdj'] = 0
        data['CompSupAdv'] = 0
        data['Words'] = 0
        data['TreeHeight'] = 0
        data['SubTrees'] = 0
        idx = 0
        for req in tqdm(data['RequirementText'], desc='Parse trees', position=0):
            tokens = tokenizer.tokenize(req)
            data.at[idx, 'Words'] = len(tokens)
            # using nltk here but analogous to universal tags
            tags = nltk.pos_tag(tokens)
            fd = nltk.FreqDist(tag for (word, tag) in tags)
            for key, value in fd.items():
                # print (key + " " + str(value))
                if key == "MD":
                    data.at[idx, 'Modal'] = value
                if key.startswith("JJ"):
                    data.at[idx, 'Adjective'] = value
                if key.startswith("NN"):
                    data.at[idx, 'Noun'] = value
                if key == "RB":
                    data.at[idx, 'Adverb'] = value
                if key == "CD":
                    data.at[idx, 'Cardinal'] = value
                if key == "JJR" or key == "JJS":
                    data.at[idx, 'CompSupAdj'] = data.at[idx, 'CompSupAdj'] + value
                if key == "RBR" or key == "RBS":
                    data.at[idx, 'CompSupAdv'] = data.at[idx, 'CompSupAdv'] + value
            # tree = parser.parse(req)
            # print (tree.height(), end =" ")
            # data.at[idx, 'TreeHeight'] = tree.height()
            # data.at[idx, 'SubTrees'] = len(tree)
            idx = idx + 1

    if "FinalSel" in dep_feat_type:
        # parser = benepar.Parser("benepar_en2")
        data['Modal'] = 0
        data['Adverb'] = 0
        data['Cardinal'] = 0
        idx = 0
        for req in tqdm(data['RequirementText'], desc='Parse trees', position=0):
            tokens = tokenizer.tokenize(req)
            tags = nltk.pos_tag(tokens)
            fd = nltk.FreqDist(tag for (word, tag) in tags)
            for key, value in fd.items():
                # print (key + " " + str(value))
                if key == "MD":
                    data.at[idx, 'Modal'] = value
                if key == "RB":
                    data.at[idx, 'Adverb'] = value
                if key == "CD":
                    data.at[idx, 'Cardinal'] = value
            idx = idx + 1

            # enrichment with features for root verbs (one feature per verb)
    if "verb" in dep_feat_type:
        # first version tried
        verbs_features = ['be', 'use', 'interface', 'comply', 'run',
                          'allow', 'display', 'send', 'track', 'include', 'notify', 'add', 'assign', 'request',
                          'record', 'indicate']
        # second version
        verbs_features = ['be', 'use', 'ensure', 'interface', 'handle', 'take', 'comply', 'run']
        # third version
        verbs_features = ['be', 'use', 'ensure', 'interface', 'handle', 'take', 'comply', 'run',
                          'allow', 'display', 'send', 'track', 'include', 'notify', 'shall', 'add', 'assign',
                          'generate', 'request',
                          'create', 'define', 'record', 'indicate', 'save'
                          ]
        for verb in verbs_features:
            data[verb] = 0

        idx = 0
        for req in tqdm(data['RequirementText'], desc='Analyzing verbs', position=0):
            newr = req.replace("'", "").replace('be able to', '').replace('be capable of', '').replace(
                'provide the ability to', '').replace('be possible to', '')
            doc = nlp(newr)
            for t in doc:
                if t.dep_ == 'ROOT':
                    req_root = lemmatizer(t.orth_, t.pos_)[0]
                    for verb in verbs_features:
                        if req_root == verb:
                            data.at[idx, verb] += 1
                            break
            idx = idx + 1

            # boolean features for root verbs (each feature takes val 1 if the req contains at least one verb in the corresponding list)
    # USED IN THE PAPER AS LAST FEATURE SET
    if "vlist" in dep_feat_type:
        Fverbs = ['allow', 'display', 'send', 'track', 'include', 'notify', 'shall', 'add', 'assign', 'generate',
                  'request', 'create', 'define', 'record', 'indicate', 'save', 'operatte']
        Qverbs = ['be', 'use', 'ensure', 'interface', 'handle', 'take', 'comply', 'run']

        data['hasFverb'] = 0
        data['hasQverb'] = 0

        idx = 0
        for req in tqdm(data['RequirementText'], desc='Analyzing verbs', position=0):
            newr = req.replace("'", "").replace('be able to', '').replace('be capable of', '').replace(
                'provide the ability to', '').replace('be possible to', '')
            doc = nlp(newr)
            for t in doc:
                if t.dep_ == 'ROOT':
                    req_root = lemmatizer(t.orth_, t.pos_)[0]
                    if req_root in Fverbs:
                        data.at[idx, 'hasFverb'] = 1
                    if req_root in Qverbs:
                        data.at[idx, 'hasQverb'] = 1
                    # data.at[idx, 'newVerb'] = 1
            idx = idx + 1

            # print(data[:30])

    # finally save the enriched datasetfile
    data.to_csv(new_file_name, encoding='utf-8')


class EnrichAPIView(APIView):

    def post(self, request, *args, **kwargs):

        folder_source_datasets = '/Users/westerops/Desktop/cmpe/cmpe492/RE-2019-Materials/Manually tagged datasets/'  # can be an url
        folder_dest_datasets = './ling/'

        # creates a folder that will contain the enriched datasets
        try:
            if not os.path.isdir(folder_dest_datasets):
                os.mkdir(folder_dest_datasets)
        except OSError:
            print("Creation of the directory %s failed" % folder_dest_datasets)
            exit()
        else:
            print("Successfully created the directory %s " % folder_dest_datasets)

        dataset_names = ['promise-reclass']
        datasets = [pd.read_csv(folder_source_datasets + dataset_name + '.csv', engine='python') for dataset_name in
                    dataset_names]

        # the features to use to enrich the datasets
        possible_dependencies_feature_sets = ['FinalSel_vlist']

        # creat all enriched datasets
        for i in range(0, len(datasets)):
            print('Dataset: ' + dataset_names[i])
            for t in possible_dependencies_feature_sets:
                print(t)
                createEnrichedDataset(datasets[i], 'enrich_results.csv', t)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="enrich_results.csv"'
        f = open("enrich_results.csv", "r")
        return_value=[]
        writer = csv.writer(response)
        for line in f:
            writer.writerow([line])

        os.remove("enrich_results.csv")
        return response