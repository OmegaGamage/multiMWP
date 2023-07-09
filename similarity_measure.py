#!/usr/bin/env python3

import os
import time
import scipy
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')

import numpy as np
from sentence_transformers import SentenceTransformer
from laserembeddings import Laser

#Setup laser
laser = Laser()

#Setup LABSE
labse_embedder = SentenceTransformer('sentence-transformers/LaBSE')


#Setup XLM-R
xlmr_para_embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1') #XLMR
xlmr_stsb_embedder = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

startTime=time.time()


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds/norms

# Method for creating embeddings from each encoder
def create_embeddings(encoder,sentences,lang):

    embeddings=[]
    if encoder == "laser":
        embeddings=laser.embed_sentences(sentences,lang=lang)
    elif encoder == "xlmr-para":
        embeddings=xlmr_para_embedder.encode(sentences)
    elif encoder == "xlmr-stsb":
        embeddings=xlmr_stsb_embedder.encode(sentences)
    elif encoder == "labse":
        embeddings = labse_embedder.encode(sentences)

    sum_embeddings=[]
    for emb in embeddings:
        if len(sum_embeddings) ==0:
            sum_embeddings=emb
        else:
            sum_embeddings+=emb
    return sum_embeddings/len(embeddings)

#calculate cosine similarity
def calculate_cosine_similarity(embed1, embed2):
    return 1-scipy.spatial.distance.cdist([embed1], [embed2], 'cosine')

work_dir='data'
out_dir = 'outputs'
# lang_pairs=['si-en', 'ta-en', 'si-ta']
contexts = ['Simple', 'Algebraic']
lang_pairs=['si-en', 'ta-en', 'as-en', 'or-en','hi-en', 'si-ta']

embeddings=['labse', 'xlmr-para', "xlmr-stsb"]

lang_dict = {'si': 'Sinhala', 'ta': 'Tamil', 'en': 'English',
             'as': 'Assamese', 'hi': 'Hindi', 'or': 'Odia'}

for context in contexts:
    for lang_pair in lang_pairs:
        src_lang = lang_pair[:2]
        tgt_lang = lang_pair[-2:]


        src_lines=[question.strip() for question in open('{}/{}-{}.txt'.format(work_dir, context, lang_dict[src_lang]), 'r', encoding='utf8')]
        tgt_lines=[question.strip() for question in open('{}/{}-{}.txt'.format(work_dir, context, lang_dict[tgt_lang]), 'r', encoding='utf8')]

        #output file
        file_out=open('{}/sim_scores_{}.{}-{}.txt'.format(out_dir, context, src_lang, tgt_lang), 'w', encoding='utf8')

        print("Processing language pair: {}...".format(lang_pair))
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            if len(src_line.split()) !=0 and len(tgt_line.split()) !=0:

                #sentence tokenizing for each line
                src_sents=sent_tokenize(src_line)
                tgt_sents=sent_tokenize(tgt_line)

                sim_scores=[]

                for embedding in embeddings:

                    src_embed=create_embeddings(embedding,src_sents,src_lang)
                    tgt_embed=create_embeddings(embedding,tgt_sents,tgt_lang)


                    cos_similarity=calculate_cosine_similarity(src_embed, tgt_embed)
                    sim_scores.append(cos_similarity)

                #update file
                file_out.write('{}\t{:0.6f}\t{:0.6f}\t{:0.6f}\n'.format(src_lines.index(src_line),
                                                     sim_scores[0][0][0],sim_scores[1][0][0], sim_scores[-1][0][0]))
