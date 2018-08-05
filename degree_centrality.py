import os
import sys
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import numpy
import math
from rouge import Rouge
import collections


sentences = ""
distinct_paragraph = []
distinct_sentences = []
filtered_sentences = []
sentence_frequency = {}
degree_mapping = {}
num_sentence = 0
numerator_sum = 0
denom_sum1 = 0
denom_sum2 = 0
threshold = float(sys.argv[1])
cosine_similarity  = numpy.zeros( shape = (10,10))
adjacency_matrix   = numpy.zeros( shape = (10,10))
high_similarity_threshold = 0.5
words_in_summary = 0
temp = []
generated_summary = ""
gold_summary = ""



def extract_sentences():

	global sentences
	global distinct_paragraph
	global distinct_sentences

	folder_name = "Topic4/"
	print folder_name

	for filename in os.listdir(folder_name):

		fp  = open(folder_name + filename,"r")

		lines = fp.readlines()
		sentences = ""

		for i in range(len(lines)):
			lines[i] = lines[i].strip('\n')
			sentences += lines[i]

		splitted = sentences.split('</P>')


		for line in splitted:
			index =  line.find('<P>')

			if ( index > -1 ):
				distinct_paragraph.append(line[index+3:len(line)].lstrip())


		for i in range(len(distinct_paragraph)):
			distinct_sentences.extend(sent_tokenize(distinct_paragraph[i]))

		distinct_paragraph[:] = []







def remove_stopwords():

	global distinct_sentences,filtered_sentences

	stop_words        =  list(stopwords.words('english'))
	punctuation_list  =  [".",";","{","}","[","]","(",")","!","@","#","-","_","--",",","''"]
	stop_words.extend(punctuation_list)


	for sentence in distinct_sentences:

		filtered_string = ""
		words = word_tokenize(sentence)

		for word in words :
			if word not in stop_words:
				filtered_string += word + " "


		temp = filtered_string.rstrip()

		if ( len(temp) > 0 ):
			filtered_sentences.append(temp)




def calculate_sentence_frequency():

	global sentence_frequency,filtered_sentences

	for sentence in filtered_sentences:
		words = list(set(sentence.split(" ")))

		for w in words:
			try:
				sentence_frequency[w] += 1
			except:
				sentence_frequency[w] = 1




def term_frequency( term , sentence ):

	list_of_words = sentence.split(" ")
	term_frequency = list_of_words.count(term)

	return term_frequency




def calculate_similarity( num_sentence ):

	global denom_sum1,denom_sum2,numerator_sum,adjacency_matrix,cosine_similarity,sentence_frequency_w,threshold
	cosine_similarity = numpy.zeros( shape = (num_sentence,num_sentence))


	for i in range(num_sentence):
		for j in range(num_sentence):

			if ( i != j ):

				numerator_sum = denom_sum1 = denom_sum2 = 0


				list_str1 = filtered_sentences[i].split(" ")
				list_str2 = filtered_sentences[j].split(" ")
				list_common = list(set(list_str1) & set(list_str2))


				for w in list_common:

					tf_i = term_frequency(w,filtered_sentences[i])
					tf_j = term_frequency(w,filtered_sentences[j])
					sentence_frequency_w = sentence_frequency[w] 
					idf  = math.log((num_sentence/sentence_frequency_w),2)
					numerator = tf_i * tf_j * (idf**2)
					numerator_sum += numerator


				for x in list_str1:

					tf_str1 = term_frequency(x,filtered_sentences[i])
					sentence_frequency_x = sentence_frequency[x]
					temp1 = (( tf_str1 * sentence_frequency_x ) ** 2 )
					denom_sum1 += temp1

				denom_sum1 = math.sqrt(denom_sum1)


				for y in list_str2:

					tf_str2 = term_frequency(y,filtered_sentences[j])
					sentence_frequency_y = sentence_frequency[y]
					temp2 = (( tf_str2 * sentence_frequency_y ) ** 2 )
					denom_sum2 += temp2

				denom_sum2 = math.sqrt(denom_sum2)


				final_value = numerator_sum/( denom_sum1 * denom_sum2 )
				cosine_similarity[i][j] = final_value



				if ( cosine_similarity[i][j] >= threshold ):
					adjacency_matrix[i][j] = 1




def calculate_degree():

	global adjacency_matrix,degree_mapping

	for i in range(num_sentence):

		degree = 0

		for j in range(num_sentence):
			degree += adjacency_matrix[i][j]

		degree_mapping[i] = degree


	od = collections.OrderedDict(sorted(degree_mapping.items(),key = lambda x: x[1] , reverse = True ))
	return od




def extract_summary( deg_map ):

	visited = numpy.zeros(shape = (num_sentence))				# Keeps trace of the sentences that have been printed

	global words_in_summary,generated_summary

	for item in deg_map:

		if ( visited[item] == 0 ):

			generated_summary += distinct_sentences[item]
			visited[item] = 1

			words_in_summary += len(word_tokenize(distinct_sentences[item]))

			if ( words_in_summary > 250 ):
				break

			for j in cosine_similarity[item]:

				if ( j > high_similarity_threshold ):

					temp  =  cosine_similarity[item].tolist()
					index =  temp.index(j)

					if ( visited[index] == 0 ):

						generated_summary += distinct_sentences[index]
						visited[index] = 1

						words_in_summary += len(word_tokenize(distinct_sentences[index]))

						if ( words_in_summary > 250 ):
							break


			if ( words_in_summary > 250 ):
				break


	print generated_summary





def compare_summary():

	global gold_summary,generated_summary 

	file_name = "GroundTruth/Topic4.1"
	fp = open(file_name,"r")


	for line in fp:
		gold_summary += line.strip('\n')


	r = Rouge()        
	score = r.get_scores(gold_summary,generated_summary)
	print score








extract_sentences()
remove_stopwords()
calculate_sentence_frequency()
num_sentence = len(filtered_sentences)									# Number of sentences in the corpora in filtered form
adjacency_matrix  = numpy.zeros( shape = (num_sentence,num_sentence))
calculate_similarity(num_sentence)
od = calculate_degree()
extract_summary(od)
compare_summary()













