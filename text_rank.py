import os
import sys
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import numpy
import math
import collections
from rouge import Rouge


sentences = ""
distinct_paragraph = []
distinct_sentences = []
filtered_sentences = []
sentence_frequency = {}
degree_mapping = {}
eigen_vector_mapping = {}
num_sentence = 0
numerator_sum = 0
denom_sum1 = 0
denom_sum2 = 0
threshold = float(sys.argv[1])
cosine_similarity  = numpy.zeros( shape = (10,10))
eigen_vector	   = numpy.zeros( shape = 10 )
words_in_summary = 0
temp = []
generated_summary = ""
gold_summary = ""



def extract_sentences():

	print 'Extract Sentences'

	global sentences
	global distinct_paragraph
	global distinct_sentences

	folder_name = "Topic5/"

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

	print 'Remove Stopwords'

	global filtered_sentences,distinct_sentences

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

	print 'In Sentence Frequency'

	global sentence_frequency,filtered_sentences

	for sentence in filtered_sentences:
		words = list(set(sentence.split(" ")))

		for w in words:

			if sentence_frequency.get(w) is None:
				sentence_frequency[w] = 1
			else:
				sentence_frequency[w] += 1




def term_frequency( term , sentence ):

	list_of_words = sentence.split(" ")
	term_frequency = list_of_words.count(term)

	return term_frequency




def calculate_similarity( num_sentence ):

	print 'In Calculate Similarity'

	global denom_sum1,denom_sum2,numerator_sum,cosine_similarity,sentence_frequency_w,threshold
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
					cosine_similarity[i][j] = 1
				else:
					cosine_similarity[i][j] = 0





def calculate_degree():

	print 'In Calculate Degree'

	global cosine_similarity,degree_mapping

	for i in range(num_sentence):

		degree = 0

		for j in range(num_sentence):
			degree += cosine_similarity[i][j]


		if ( degree == 0 ):
			degree = 1

		degree_mapping[i] = degree


	for i in range(num_sentence):
		for j in range(num_sentence):
			cosine_similarity[i][j] = float(cosine_similarity[i][j])/degree_mapping[i]




def power_method():

	print 'In Power Method'

	global eigen_vector,cosine_similarity
	eigen_vector = numpy.zeros(shape = num_sentence)

	
	for i in range(len(eigen_vector)):
		eigen_vector[i] = float(1)/num_sentence


	for i in range(1000):
		eigen_vector = numpy.dot(numpy.transpose(cosine_similarity),eigen_vector)


	for i  in range(len(eigen_vector)):
		eigen_vector_mapping[i] = eigen_vector[i]


	od = collections.OrderedDict(sorted(eigen_vector_mapping.items(),key = lambda x: x[1] , reverse = True ))
	return od




def extract_summary( eigen_map ):

	print 'Extracting Summary'

	global words_in_summary,generated_summary

	for item in eigen_map:

		generated_summary += distinct_sentences[item]	
		words_in_summary  += len(word_tokenize(distinct_sentences[item]))

		if ( words_in_summary > 250 ):
			break


	print generated_summary





def compare_summary():

	print 'Comparing Summary'

	global gold_summary,generated_summary 

	file_name = "GroundTruth/Topic5.1"
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
calculate_similarity(num_sentence)
calculate_degree()
od = power_method()
extract_summary(od)
compare_summary()
