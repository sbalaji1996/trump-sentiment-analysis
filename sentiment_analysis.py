import math
import string
import sys
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer 
from nltk.classify import NaiveBayesClassifier

def main():
	reload(sys)  
	sys.setdefaultencoding('utf8')

	statuses_dirty = pd.read_csv('trump.csv')
	statuses = statuses_dirty[:3098]

	#================================================================================================================================================================
	#sentiment extraction, which is performed for each status in the cleaned array, which is then read into a file for storage
	from nltk.sentiment import SentimentIntensityAnalyzer
	analyzer = SentimentIntensityAnalyzer()
	def get_sentiment_intensity(status):
		return analyzer.polarity_scores(status)

	print('Extracting sentiment ... this may take a few seconds!')
	'''
	orig_stdout = sys.stdout
	f = file('polarities.txt', 'w')
	sys.stdout = f

	for status in statuses["status_message"]:
		print(get_sentiment_intensity(str(status).lower())['compound'])

	sys.stdout = orig_stdout
	f.close()
	'''
	print('Extracted sentiment! See polarities.txt for a list, but the python script should read the sentiments in automatically.')

	lines = []
	with open('polarities.txt') as f:
	    lines = f.read().splitlines()

	#splits statuses into words
	def status_feats(status):
		status = str(status).translate(None, string.punctuation)
		feats = status.split(' ')
		return dict([(word, True) for word in feats])

	#translates sentiments from integers to strings, for easier analysis
	def translate(values):
		translated = []
		for val in values:
			if (float(val) < 0.0): translated.append('neg')
			if (float(val) > 0.0): translated.append('pos')
			if (float(val) == 0.0): translated.append('neut')
		return translated

	res = translate(lines)

	#set up the list of dictionaries, containing ([(list of words in status, True)], sentiment of status) so that the classifier can be trained
	trump_feats = []
	for i, status in enumerate(statuses["status_message"]):
		trump_feats.append((status_feats(str(status).lower()), res[i]))

	classifier = NaiveBayesClassifier.train(trump_feats)
	classifier.show_most_informative_features()

	#================================================================================================================================================================
	#logistic regression performed on statuses that do not have neutral sentiments (0.0)

	#remove neutral sentiments and split into training and testing sets
	def remove_neutrals(values):
		translated = []
		for index, val in enumerate(values):
			if (float(val) < 0.0): translated.append(1)
			if (float(val) > 0.0): translated.append(3)
		three_fourths = len(translated) * 3 / 4
		return [translated[0:three_fourths], translated[three_fourths:]]

	#tests for the presence of specific keywords across the statuses
	def words_presence_without_neuts():
		heavy_feats = ['great', 'our', 'america', 'thank', 'make', 'hillary', 'again', 'clinton', 
						'trump', 'terrorism', 'email', 'crime', 'terror', 'solve', 
						'people', 'lied', 'safe', 'president', 'jobs', 'innocent']

		rtn = []
		for i in range(len(heavy_feats)):
			rtn.append([])

		for index, status in enumerate(statuses["status_message"]):
			if (res[index] != 'neut'):
				status = str(status).translate(None, string.punctuation)
				feats = status.lower().split(' ') 
				for idx, feat in enumerate(heavy_feats):
					if feat in feats:
						rtn[idx].append(2)
					else:
						rtn[idx].append(1)

		training = []
		testing = []
		three_fourths = len(rtn[0]) * 3 / 4

		for index, feat in enumerate(rtn):
			training.append(feat[0:three_fourths])
			testing.append(feat[three_fourths:])

		return [training, testing]

	polarities_without_neuts = remove_neutrals(lines)
	feats_present = words_presence_without_neuts()

	training_pols = polarities_without_neuts[0]
	testing_pols = polarities_without_neuts[1]
	training_word_presence = feats_present[0]
	testing_word_presence = feats_present[1]

	#creates coefficients for each feature
	def log_reg(feats_presence, polarities):
		x = feats_presence
		y = polarities

		stack = np.column_stack(x)
		coefficients = np.linalg.lstsq(stack, y)[0]
		return coefficients

	def sigmoid(x):
		return 1/(1 + math.exp(-x))

	#test done by multiplying each feature by its corresponding coefficient, applying sigmoid, filtering into sections, and checking with the actual sentiment
	def test_without_neuts(feats_presence, coeffs, polarities):
		total = []
		for i in range(len(feats_presence[0])):
			total.append(coeffs[0]*feats_presence[0][i] 
						+ coeffs[1]*feats_presence[1][i] 
						+ coeffs[2]*feats_presence[2][i]
						+ coeffs[3]*feats_presence[3][i]
						+ coeffs[4]*feats_presence[4][i]
						+ coeffs[5]*feats_presence[5][i]
						+ coeffs[6]*feats_presence[6][i]
						+ coeffs[7]*feats_presence[7][i]
						+ coeffs[8]*feats_presence[8][i]
						+ coeffs[9]*feats_presence[9][i]
						+ coeffs[10]*feats_presence[10][i]
						+ coeffs[11]*feats_presence[11][i]
						+ coeffs[12]*feats_presence[12][i]
						+ coeffs[13]*feats_presence[13][i]
						+ coeffs[14]*feats_presence[14][i]
						+ coeffs[15]*feats_presence[15][i]
						+ coeffs[16]*feats_presence[16][i]
						+ coeffs[17]*feats_presence[17][i]
						+ coeffs[18]*feats_presence[18][i]
						+ coeffs[19]*feats_presence[19][i])

		results = []
		for i, item in enumerate(total):
			#print(sigmoid(item))
			if (sigmoid(item) > 0.893):
				results.append(3)
			else:
				results.append(1)

		num_diffs = 0
		for j in range(len(polarities)):
			if (results[j] != polarities[j]):
				num_diffs += 1

		accuracy = (len(results) - num_diffs) * 1.0/(len(results)) * 100

		print 'The accuracy of this model, testing without neutral sentiments, is %f percent.' % (accuracy)

	coeffs = log_reg(training_word_presence, training_pols)
	test_without_neuts(testing_word_presence, coeffs, testing_pols)

	#================================================================================================================================================================
	#logistic regression performed on all statuses

	#split into training and testing polarities and divide into training and testing sets
	def translate_with_neuts(values):
		translated = []
		for index, val in enumerate(values):
			if (float(val) < 0.0): translated.append(1)
			if (float(val) > 0.0): translated.append(3)
			if (float(val) == 0.0): translated.append(2)
		three_fourths = len(translated) * 3 / 4
		return [translated[0:three_fourths], translated[three_fourths:]]

	#test for the presence of specific keywords across the statuses and divide into training and testing sets
	def words_present_with_neuts():
		heavy_feats = ['great', 'judgement', 'america', 'thank', 'make', 'hillary', 'again', 'clinton', 
						'trump', 'terrorism', 'email', 'crime', 'terror', 'solve', 
						'people', 'lied', 'safe', 'president', 'jobs', 'innocent']

		rtn = []
		for i in range(len(heavy_feats)):
			rtn.append([])

		for index, status in enumerate(statuses["status_message"]):
			status = str(status).translate(None, string.punctuation)
			feats = status.lower().split(' ') 
			for idx, feat in enumerate(heavy_feats):
				if feat in feats:
					rtn[idx].append(2)
				else:
					rtn[idx].append(1)

		training = []
		testing = []
		three_fourths = len(rtn[0]) * 3 / 4
		
		for index, feat in enumerate(rtn):
			training.append(feat[0:three_fourths])
			testing.append(feat[three_fourths:])

		return [training, testing]

	polarities_with_neuts = translate_with_neuts(lines)
	feats_present = words_present_with_neuts()
	training_pols = polarities_with_neuts[0]
	testing_pols = polarities_with_neuts[1]
	training_word_presence = feats_present[0]
	testing_word_presence = feats_present[1]

	#test done by multiplying each feature by its corresponding coefficient, applying sigmoid, filtering into sections, and checking with the actual sentiment
	def test_with_neuts(feats_presence, coeffs, polarities):
		total = []
		for i in range(len(feats_presence[0])):
			total.append(coeffs[0]*feats_presence[0][i] 
						+ coeffs[1]*feats_presence[1][i] 
						+ coeffs[2]*feats_presence[2][i]
						+ coeffs[3]*feats_presence[3][i]
						+ coeffs[4]*feats_presence[4][i]
						+ coeffs[5]*feats_presence[5][i]
						+ coeffs[6]*feats_presence[6][i]
						+ coeffs[7]*feats_presence[7][i]
						+ coeffs[8]*feats_presence[8][i]
						+ coeffs[9]*feats_presence[9][i]
						+ coeffs[10]*feats_presence[10][i]
						+ coeffs[11]*feats_presence[11][i]
						+ coeffs[12]*feats_presence[12][i]
						+ coeffs[13]*feats_presence[13][i]
						+ coeffs[14]*feats_presence[14][i]
						+ coeffs[15]*feats_presence[15][i]
						+ coeffs[16]*feats_presence[16][i]
						+ coeffs[17]*feats_presence[17][i]
						+ coeffs[18]*feats_presence[18][i]
						+ coeffs[19]*feats_presence[19][i])

		results = []
		for i, item in enumerate(total):
			if (sigmoid(item) > 0.90):
				results.append(3)
			elif (sigmoid(item) < 0.88):
				results.append(1)
			else:
				results.append(2)


		num_diffs = 0
		for j in range(len(polarities)):
			if (results[j] != polarities[j]):
				num_diffs += 1

		accuracy = (len(results) - num_diffs) * 1.0/(len(results)) * 100

		print 'The accuracy of this model, testing with neutral sentiments, is %f percent.' % (accuracy)

	coeffs = log_reg(training_word_presence, training_pols)
	test_with_neuts(testing_word_presence, coeffs, testing_pols)

if __name__ == "__main__":
	main()
