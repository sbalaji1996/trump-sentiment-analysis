import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import re
import gc
from nltk.sentiment import SentimentIntensityAnalyzer

#==========================DATA============================

pd.options.mode.chained_assignment = None  # default='warn'

def status_cleaner(filename, lim=None):
	file_csv = filename+".csv"
	dirty = pd.read_csv(file_csv)
	
	# Take statuses from 3/18/2015 forward or until a limit is reached

	k = 0
	for time in dirty['status_published']:
		t = time.split()
		t2 = t[0].split("/")
		if len(t2) == 1:
			tmp = t[0].split("-")
			t2 = [tmp[1], tmp[2], tmp[0]]
		for i in range(3):
			if t2[i][0] == '0':
				n = t2[i][1:]
				t2[i] = n
		if t2[2] == '2016':
			if t2[1] == '23':
				if t2[0] == '2':
					break
		k += 1
		if k == lim:
			break
			
	statuses = dirty[:k]

	# Check reactions sums

	for i in range(len(statuses)):
		num_reacts = statuses['num_reactions'][i]
		real_sum = statuses['num_likes'][i]+statuses['num_loves'][i]+statuses['num_hahas'][i]+statuses['num_wows'][i]+statuses['num_angrys'][i]+statuses['num_sads'][i]
		if ((num_reacts-real_sum) != 0):
			statuses['num_reactions'][i] = real_sum

	return statuses



def most_used_words(statuses):
	text = ""
	for msg in statuses["status_message"]:
		text += " "
		text += str(msg)

	with open("content.txt", "w") as text_file:
	    text_file.write("{0}".format(text))

	content = open("content.txt", "r")
	content = content.read()
	for c in string.punctuation:
		content_split = content.replace(c, "")

	content_split = content_split.lower()
	words = re.compile(r'\w+').findall(content_split)
	counts = collections.Counter(words)
	removed_words = ['the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'would', 'there', 'what', 'so', 'about', 'who', 'get', 'which', 'me', 'when', 'is', 'are', 'has', 'out', 'am', 'nan', 'if', 'was', 'than', 'no', 'been', 'want', 'com', 'these', 'like', 'their', 'them', 'us', 'djt', 'american', 'americans', 'makeamericagreatagain', 'trump2016']

	best_words = []
	for item in counts.most_common():
		if (len(item[0]) > 1) and (item[0] not in removed_words):
			best_words.append(item)

	return best_words



analyzer = SentimentIntensityAnalyzer()
def get_sentiment_intensity(status):
	return analyzer.polarity_scores(status)

#==========================VIZ============================

def post_times(statuses, dataname):
	times = []
	for time in statuses['status_published']:
		t = time.split()
		if (len(t[1]) == 7):
			times.append(int(t[1][0:1]))
		else:
			times.append(int(t[1][0:2]))


	number = [0, 0, 0, 0, 0, 0, 0, 0]
	rxns = [0, 0, 0, 0, 0, 0, 0, 0]
	i = 0
	for time in times:
		if (time <=3):
			number[0] += 1
			rxns[0] += statuses['num_reactions'][i]
		elif (time <= 6):
			number[1] += 1
			rxns[1] += statuses['num_reactions'][i]
		elif (time <= 9):
			number[2] += 1
			rxns[2] += statuses['num_reactions'][i]
		elif (time <= 12):
			number[3] += 1
			rxns[3] += statuses['num_reactions'][i]
		elif (time <= 15):
			number[4] += 1
			rxns[4] += statuses['num_reactions'][i]
		elif (time <= 18):
			number[5] += 1
			rxns[5] += statuses['num_reactions'][i]
		elif (time <= 21):
			number[6] += 1
			rxns[6] += statuses['num_reactions'][i]
		else:
			number[7] += 1
			rxns[7] += statuses['num_reactions'][i]
		i += 1

	avg_rxns = []
	j = 0
	while j < len(number):
		avg_rxns.append(rxns[j]/number[j])
		j += 1

	labels = ['12am-3am', '3am-6am', '6am-9am', '9am-12pm', '12pm-3pm', '3pm-6pm', '6pm-9pm', '9pm-12pm']
	labels24 = [0, 1, 2, 3, 4, 5, 6, 7]
	width = 0.2

	ax = plt.subplot(111)
	ax.bar(labels24, avg_rxns, color='g', align='center')
	ax.set_xticks(labels24)
	ax.set_xticklabels(labels)
	ax.set_xlabel('hours posted (24 hour, central standard time)')
	ax.set_ylabel('average #reactions per post')
	plt.xticks(rotation=45)

	# filename = "times/"+dataname		# deposits graphs in a folder named "times"
	# plt.savefig(filename)
	# plt.close()
	plt.show()



def post_types(statuses, dataname):
	link = []
	photo = []
	video = []
	status = []

	for i in range(len(statuses)):
		post = statuses['status_type'][i]	
		if post == "link":
			link.append(statuses['num_reactions'][i])
		elif post == "photo":
			photo.append(statuses['num_reactions'][i])
		elif post == "video":
			video.append(statuses['num_reactions'][i])
		elif post == "status":
			status.append(statuses['num_reactions'][i])

	avgs = []
	for kind in [status, video, photo, link]:
		avg = sum(kind)/len(kind)
		avgs.append(avg)

	y = avgs
	x = range(len(avgs))
	labels = ["Status", "Video", "Photo", "*Link"]
	width = 1/1.5
	plt.bar(x, y, width, color="red", tick_label=labels, align="center")
	plt.xlabel("Post Type")
	plt.ylabel("Average Total Reactions") 

	# filename = "types/"+dataname		# deposits graphs in a folder named "types"
	# plt.savefig(filename)
	# plt.close()
	plt.show()



def reaction_bars(statuses, best_words, dataname, show_likes=False):
	reacts_dict = {}
	for buzz in best_words:
		reacts_dict[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}

	for i in range(len(statuses)):
		for buzz in best_words:
			if buzz in str(statuses["status_message"][i]).lower():
				buzz_dict = reacts_dict[buzz]
				buzz_dict["likes"] += statuses['num_likes'][i]
				buzz_dict["loves"] += statuses['num_loves'][i]
				buzz_dict["wows"] += statuses['num_wows'][i]
				buzz_dict["hahas"] += statuses['num_hahas'][i]
				buzz_dict["sads"] += statuses['num_sads'][i]
				buzz_dict["angrys"] += statuses['num_angrys'][i]
				buzz_dict["total"] += statuses['num_reactions'][i]
				buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

	likes = np.array([])
	loves = np.array([])
	wows = np.array([])
	hahas = np.array([])
	sads = np.array([])
	angrys = np.array([])
	loves_adjusted = np.array([])
	wows_adjusted = np.array([])
	hahas_adjusted = np.array([])
	sads_adjusted = np.array([])
	angrys_adjusted = np.array([])
	labels = []
	for buzz in best_words:
		buzz_dict = reacts_dict[buzz]
		if buzz_dict["total"] == 0:
			buzz_dict["total"] = 1
			buzz_dict["total no likes"] = 1
		
		likes = np.append(likes, buzz_dict["likes"]/buzz_dict["total"])
		loves = np.append(loves, buzz_dict["loves"]/buzz_dict["total"])
		wows = np.append(wows, buzz_dict["wows"]/buzz_dict["total"])
		hahas = np.append(hahas, buzz_dict["hahas"]/buzz_dict["total"])
		sads = np.append(sads, buzz_dict["sads"]/buzz_dict["total"])
		angrys = np.append(angrys, buzz_dict["angrys"]/buzz_dict["total"])
		
		loves_adjusted = np.append(loves_adjusted, buzz_dict["loves"]/buzz_dict["total no likes"])
		wows_adjusted = np.append(wows_adjusted, buzz_dict["wows"]/buzz_dict["total no likes"])
		hahas_adjusted = np.append(hahas_adjusted, buzz_dict["hahas"]/buzz_dict["total no likes"])
		sads_adjusted = np.append(sads_adjusted, buzz_dict["sads"]/buzz_dict["total no likes"])
		angrys_adjusted = np.append(angrys_adjusted, buzz_dict["angrys"]/buzz_dict["total no likes"])
		
		labels.append(buzz)

	x = range(len(best_words))
	width = 0.35       # the width of the bars: can also be len(x) sequence
	axes = plt.gca()
	axes.set_ylim([0,1])

	if (show_likes):
		p1 = plt.bar(x, angrys, width, color='r', tick_label=labels)
		p2 = plt.bar(x, sads, width, color='b',
		             bottom=angrys)
		p3 = plt.bar(x, hahas, width, color='y',
		             bottom=(angrys+sads))
		p4 = plt.bar(x, wows, width, color='g',
		             bottom=(angrys+sads+hahas))
		p5 = plt.bar(x, loves, width, color='m',
		             bottom=(angrys+sads+hahas+wows))
		p6 = plt.bar(x, likes, width, color='c',
		             bottom=(angrys+sads+hahas+loves))
		plt.ylabel('Reaction Percentages')
		plt.legend((p6[0], p5[0], p4[0], p3[0], p2[0], p1[0]), ('Likes', 'Loves', 'Wows', 'Hahas', 'Sads', 'Angrys'))
	else:
		print(angrys_adjusted)
		p1 = plt.bar(x, angrys_adjusted, width, color='r', tick_label=labels)
		p2 = plt.bar(x, sads_adjusted, width, color='b',
		             bottom=angrys_adjusted)
		p3 = plt.bar(x, hahas_adjusted, width, color='y',
		             bottom=(sads_adjusted+angrys_adjusted))
		p4 = plt.bar(x, wows_adjusted, width, color='g',
		             bottom=(sads_adjusted+angrys_adjusted+hahas_adjusted))
		p5 = plt.bar(x, loves_adjusted, width, color='m',
		             bottom=(sads_adjusted+angrys_adjusted+hahas_adjusted+wows_adjusted))
		plt.ylabel('Reaction Percentages')
		plt.legend((p5[0], p4[0], p3[0], p2[0], p1[0]), ('Loves', 'Wows', 'Hahas', 'Sads', 'Angrys'))

	# filename = "rb/"+best_words[0]+"_to_"+best_words[len(best_words)-1]+"_"+dataname		# deposits graphs in a folder named "rb"
	# plt.savefig(filename)
	# plt.close()
	plt.show()



def color_comp_to_whole(statuses, best_words, dataname, show_likes=False, sents=None):
	total_reacts = sum(statuses['num_reactions'])
	total_reacts_no_likes = sum(statuses['num_reactions'])-sum(statuses['num_likes'])
	likes_percent = sum(statuses['num_likes'])/total_reacts
	loves_percent = sum(statuses['num_loves'])/total_reacts
	loves_percent_adjusted = sum(statuses['num_loves'])/total_reacts_no_likes
	wows_percent = sum(statuses['num_wows'])/total_reacts
	wows_percent_adjusted = sum(statuses['num_wows'])/total_reacts_no_likes
	hahas_percent = sum(statuses['num_hahas'])/total_reacts
	hahas_percent_adjusted = sum(statuses['num_hahas'])/total_reacts_no_likes
	sads_percent = sum(statuses['num_sads'])/total_reacts
	sads_percent_adjusted = sum(statuses['num_sads'])/total_reacts_no_likes
	angrys_percent = sum(statuses['num_angrys'])/total_reacts
	angrys_percent_adjusted = sum(statuses['num_angrys'])/total_reacts_no_likes
	total_reacts_percentages = np.array([likes_percent, loves_percent, wows_percent, hahas_percent, sads_percent, angrys_percent])
	total_no_likes_percentages = np.array([loves_percent_adjusted, wows_percent_adjusted, hahas_percent_adjusted, sads_percent_adjusted, angrys_percent_adjusted])

	reacts_dict = {}
	for buzz in best_words:
		reacts_dict[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}

	for i in range(len(statuses)):
		for buzz in best_words:
			if buzz in str(statuses["status_message"][i]).lower():
				buzz_dict = reacts_dict[buzz]
				buzz_dict["likes"] += statuses['num_likes'][i]
				buzz_dict["loves"] += statuses['num_loves'][i]
				buzz_dict["wows"] += statuses['num_wows'][i]
				buzz_dict["hahas"] += statuses['num_hahas'][i]
				buzz_dict["sads"] += statuses['num_sads'][i]
				buzz_dict["angrys"] += statuses['num_angrys'][i]
				buzz_dict["total"] += statuses['num_reactions'][i]
				buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

	likes = np.array([])
	loves = np.array([])
	wows = np.array([])
	hahas = np.array([])
	sads = np.array([])
	angrys = np.array([])
	loves_adjusted = np.array([])
	wows_adjusted = np.array([])
	hahas_adjusted = np.array([])
	sads_adjusted = np.array([])
	angrys_adjusted = np.array([])
	for buzz in best_words:
		buzz_dict = reacts_dict[buzz]
		if buzz_dict["total"] == 0:
			buzz_dict["total"] = 1
			buzz_dict["total no likes"] = 1
		
		likes = np.append(likes, buzz_dict["likes"]/buzz_dict["total"])
		loves = np.append(loves, buzz_dict["loves"]/buzz_dict["total"])
		wows = np.append(wows, buzz_dict["wows"]/buzz_dict["total"])
		hahas = np.append(hahas, buzz_dict["hahas"]/buzz_dict["total"])
		sads = np.append(sads, buzz_dict["sads"]/buzz_dict["total"])
		angrys = np.append(angrys, buzz_dict["angrys"]/buzz_dict["total"])
		
		loves_adjusted = np.append(loves_adjusted, buzz_dict["loves"]/buzz_dict["total no likes"])
		wows_adjusted = np.append(wows_adjusted, buzz_dict["wows"]/buzz_dict["total no likes"])
		hahas_adjusted = np.append(hahas_adjusted, buzz_dict["hahas"]/buzz_dict["total no likes"])
		sads_adjusted = np.append(sads_adjusted, buzz_dict["sads"]/buzz_dict["total no likes"])
		angrys_adjusted = np.append(angrys_adjusted, buzz_dict["angrys"]/buzz_dict["total no likes"])
	
	rgb_arrs = []
	diffs_arrs = []
	for i in range(len(best_words)):
		post_reacts_percentages = np.array([likes[i], loves[i], wows[i], hahas[i], sads[i], angrys[i]])
		comp_percentages = total_reacts_percentages
		if not show_likes:
			post_reacts_percentages = np.array([loves_adjusted[i], wows_adjusted[i], hahas_adjusted[i], sads_adjusted[i], angrys_adjusted[i]])
			comp_percentages = total_no_likes_percentages
		diff_arr = post_reacts_percentages-comp_percentages
		rgbs = []
		for diff in diff_arr:
			p = diff
			rgb = (1, 1, 1)
			if diff > 0:
				scale = 1-p
				rgb = (1, scale, scale)
			elif diff < 0:
				scale = 1+p
				rgb = (scale, scale, 1)
			rgbs.append(rgb)
		rgb_arrs.append(rgbs)
		diff_perc = np.round(diff_arr, 7)*100
		diffs_arrs.append(diff_perc)

	for i in range(len(best_words)):
		slices = [(100/6), (100/6), (100/6), (100/6), (100/6), (100/6)]
		react_names = ['likes', 'loves', 'wows', 'hahas', 'sads', 'angrys']
		k = 6
		if not show_likes:
			slices = [(100/5), (100/5), (100/5), (100/5), (100/5)]
			react_names = react_names[1:]
			k = 5
		slice_texts = []
		for j in range(k):
			diff = diffs_arrs[i][j]
			txt = react_names[j]+"\n"+str(diff)
			slice_texts.append(txt)
		plt.pie(slices, labels=slice_texts, colors=rgb_arrs[i], startangle=90)
		# Set aspect ratio to be equal so that pie is drawn as a circle.
		plt.axis('equal')
		ttl = best_words[i]
		if sents != None:
			sentiment = str(sents[i])[:6]
			if eval(sentiment) > 0:
				sentiment = sentiment[:5]
			ttl = ttl+"\nSentiment: "+sentiment
		plt.title(ttl)

		# filename = "cc2w/"+best_words[i]+"_"+dataname		# deposits graphs in a folder named "cc2w"
		# plt.savefig(filename)
		# plt.close()
		plt.show()



def color_comp_to_other(statuses, others, best_words, dataname, show_likes=False):
	reacts_dict_statuses = {}
	reacts_dict_others = {}
	for buzz in best_words:
		reacts_dict_statuses[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}
		reacts_dict_others[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}

	for i in range(len(statuses)):
		for buzz in best_words:
			if buzz in str(statuses["status_message"][i]).lower():
				buzz_dict = reacts_dict_statuses[buzz]
				buzz_dict["likes"] += statuses['num_likes'][i]
				buzz_dict["loves"] += statuses['num_loves'][i]
				buzz_dict["wows"] += statuses['num_wows'][i]
				buzz_dict["hahas"] += statuses['num_hahas'][i]
				buzz_dict["sads"] += statuses['num_sads'][i]
				buzz_dict["angrys"] += statuses['num_angrys'][i]
				buzz_dict["total"] += statuses['num_reactions'][i]
				buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

	for i in range(len(others)):
		for buzz in best_words:
			if buzz in str(others["status_message"][i]).lower():
				buzz_dict = reacts_dict_others[buzz]
				buzz_dict["likes"] += others['num_likes'][i]
				buzz_dict["loves"] += others['num_loves'][i]
				buzz_dict["wows"] += others['num_wows'][i]
				buzz_dict["hahas"] += others['num_hahas'][i]
				buzz_dict["sads"] += others['num_sads'][i]
				buzz_dict["angrys"] += others['num_angrys'][i]
				buzz_dict["total"] += others['num_reactions'][i]
				buzz_dict["total no likes"] += (others['num_reactions'][i] - others['num_likes'][i])

	likes_statuses = np.array([])
	likes_others = np.array([])
	loves_statuses = np.array([])
	loves_others = np.array([])
	wows_statuses = np.array([])
	wows_others = np.array([])
	hahas_statuses = np.array([])
	hahas_others = np.array([])
	sads_statuses = np.array([])
	sads_others = np.array([])
	angrys_statuses = np.array([])
	angrys_others = np.array([])
	loves_adjusted_statuses = np.array([])
	loves_adjusted_others = np.array([])
	wows_adjusted_statuses = np.array([])
	wows_adjusted_others = np.array([])
	hahas_adjusted_statuses = np.array([])
	hahas_adjusted_others = np.array([])
	sads_adjusted_statuses = np.array([])
	sads_adjusted_others = np.array([])
	angrys_adjusted_statuses = np.array([])
	angrys_adjusted_others = np.array([])
	for buzz in best_words:
		buzz_dict_statuses = reacts_dict_statuses[buzz]
		buzz_dict_others = reacts_dict_others[buzz]
		if buzz_dict_statuses["total"] == 0:
			buzz_dict_statuses["total"] = 1
			buzz_dict_statuses["total no likes"] = 1
		if buzz_dict_others["total"] == 0:
			buzz_dict_others["total"] = 1
			buzz_dict_others["total no likes"] = 1
		
		likes_statuses = np.append(likes_statuses, buzz_dict_statuses["likes"]/buzz_dict_statuses["total"])
		likes_others = np.append(likes_others, buzz_dict_others["likes"]/buzz_dict_others["total"])
		loves_statuses = np.append(loves_statuses, buzz_dict_statuses["loves"]/buzz_dict_statuses["total"])
		loves_others = np.append(loves_others, buzz_dict_others["loves"]/buzz_dict_others["total"])
		wows_statuses = np.append(wows_statuses, buzz_dict_statuses["wows"]/buzz_dict_statuses["total"])
		wows_others = np.append(wows_others, buzz_dict_others["wows"]/buzz_dict_others["total"])
		hahas_statuses = np.append(hahas_statuses, buzz_dict_statuses["hahas"]/buzz_dict_statuses["total"])
		hahas_others = np.append(hahas_others, buzz_dict_others["hahas"]/buzz_dict_others["total"])
		sads_statuses = np.append(sads_statuses, buzz_dict_statuses["sads"]/buzz_dict_statuses["total"])
		sads_others = np.append(sads_others, buzz_dict_others["sads"]/buzz_dict_others["total"])
		angrys_statuses = np.append(angrys_statuses, buzz_dict_statuses["angrys"]/buzz_dict_statuses["total"])
		angrys_others = np.append(angrys_others, buzz_dict_others["angrys"]/buzz_dict_others["total"])
		
		loves_adjusted_statuses = np.append(loves_adjusted_statuses, buzz_dict_statuses["loves"]/buzz_dict_statuses["total no likes"])
		loves_adjusted_others = np.append(loves_adjusted_others, buzz_dict_others["loves"]/buzz_dict_others["total no likes"])
		wows_adjusted_statuses = np.append(wows_adjusted_statuses, buzz_dict_statuses["wows"]/buzz_dict_statuses["total no likes"])
		wows_adjusted_others = np.append(wows_adjusted_others, buzz_dict_others["wows"]/buzz_dict_others["total no likes"])
		hahas_adjusted_statuses = np.append(hahas_adjusted_statuses, buzz_dict_statuses["hahas"]/buzz_dict_statuses["total no likes"])
		hahas_adjusted_others = np.append(hahas_adjusted_others, buzz_dict_others["hahas"]/buzz_dict_others["total no likes"])
		sads_adjusted_statuses = np.append(sads_adjusted_statuses, buzz_dict_statuses["sads"]/buzz_dict_statuses["total no likes"])
		sads_adjusted_others = np.append(sads_adjusted_others, buzz_dict_others["sads"]/buzz_dict_others["total no likes"])
		angrys_adjusted_statuses = np.append(angrys_adjusted_statuses, buzz_dict_statuses["angrys"]/buzz_dict_statuses["total no likes"])
		angrys_adjusted_others = np.append(angrys_adjusted_others, buzz_dict_others["angrys"]/buzz_dict_others["total no likes"])
	
	rgb_arrs = []
	diffs_arrs = []
	for i in range(len(best_words)):
		statuses_reacts_percentages = np.array([likes_statuses[i], loves_statuses[i], wows_statuses[i], hahas_statuses[i], sads_statuses[i], angrys_statuses[i]])
		others_reacts_percentages = np.array([likes_others[i], loves_others[i], wows_others[i], hahas_others[i], sads_others[i], angrys_others[i]])
		if not show_likes:
			statuses_reacts_percentages = np.array([loves_adjusted_statuses[i], wows_adjusted_statuses[i], hahas_adjusted_statuses[i], sads_adjusted_statuses[i], angrys_adjusted_statuses[i]])
			others_reacts_percentages = np.array([loves_adjusted_others[i], wows_adjusted_others[i], hahas_adjusted_others[i], sads_adjusted_others[i], angrys_adjusted_others[i]])
		diff_arr = statuses_reacts_percentages-others_reacts_percentages
		rgbs = []
		for diff in diff_arr:
			p = diff
			rgb = (1, 1, 1)
			if diff > 0:
				scale = 1-p
				rgb = (1, scale, scale)
			elif diff < 0:
				scale = 1+p
				rgb = (scale, scale, 1)
			rgbs.append(rgb)
		rgb_arrs.append(rgbs)
		diff_perc = np.round(diff_arr, 7)*100
		diffs_arrs.append(diff_perc)

	for i in range(len(best_words)):
		slices = [(100/6), (100/6), (100/6), (100/6), (100/6), (100/6)]
		react_names = ['likes', 'loves', 'wows', 'hahas', 'sads', 'angrys']
		k = 6
		if not show_likes:
			slices = [(100/5), (100/5), (100/5), (100/5), (100/5)]
			react_names = react_names[1:]
			k = 5
		slice_texts = []
		for j in range(k):
			diff = diffs_arrs[i][j]
			txt = react_names[j]+"\n"+str(diff)
			slice_texts.append(txt)
		plt.pie(slices, labels=slice_texts, colors=rgb_arrs[i], startangle=90)
		# Set aspect ratio to be equal so that pie is drawn as a circle.
		plt.axis('equal')
		plt.title(best_words[i])

		# filename = "cc2o/"+best_words[i]+"_"+dataname[0]+"_vs_"+dataname[1]		# deposits graphs in a folder named "cc2o"
		# plt.savefig(filename)
		# plt.close()
		plt.show()


special_punc = ["–", "--", "—-", "-–", "- ", "..", "??", "!!"]
def color_comp_w_punc(statuses, best_words, dataname, show_likes=False, punc_arr=special_punc):
	reacts_dict = {}
	reacts_dict_w_punc = {}
	for buzz in best_words:
		reacts_dict[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}
		reacts_dict_w_punc[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}

	for i in range(len(statuses)):
		for buzz in best_words:
			if buzz in str(statuses["status_message"][i]).lower():
				buzz_dict = reacts_dict[buzz]
				buzz_dict["likes"] += statuses['num_likes'][i]
				buzz_dict["loves"] += statuses['num_loves'][i]
				buzz_dict["wows"] += statuses['num_wows'][i]
				buzz_dict["hahas"] += statuses['num_hahas'][i]
				buzz_dict["sads"] += statuses['num_sads'][i]
				buzz_dict["angrys"] += statuses['num_angrys'][i]
				buzz_dict["total"] += statuses['num_reactions'][i]
				buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

				for punc in punc_arr:
					if punc in str(statuses["status_message"][i]):
						buzz_dict = reacts_dict_w_punc[buzz]
						buzz_dict["likes"] += statuses['num_likes'][i]
						buzz_dict["loves"] += statuses['num_loves'][i]
						buzz_dict["wows"] += statuses['num_wows'][i]
						buzz_dict["hahas"] += statuses['num_hahas'][i]
						buzz_dict["sads"] += statuses['num_sads'][i]
						buzz_dict["angrys"] += statuses['num_angrys'][i]
						buzz_dict["total"] += statuses['num_reactions'][i]
						buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

	likes_statuses = np.array([])
	likes_others = np.array([])
	loves_statuses = np.array([])
	loves_others = np.array([])
	wows_statuses = np.array([])
	wows_others = np.array([])
	hahas_statuses = np.array([])
	hahas_others = np.array([])
	sads_statuses = np.array([])
	sads_others = np.array([])
	angrys_statuses = np.array([])
	angrys_others = np.array([])
	loves_adjusted_statuses = np.array([])
	loves_adjusted_others = np.array([])
	wows_adjusted_statuses = np.array([])
	wows_adjusted_others = np.array([])
	hahas_adjusted_statuses = np.array([])
	hahas_adjusted_others = np.array([])
	sads_adjusted_statuses = np.array([])
	sads_adjusted_others = np.array([])
	angrys_adjusted_statuses = np.array([])
	angrys_adjusted_others = np.array([])
	for buzz in best_words:
		buzz_dict_statuses = reacts_dict[buzz]
		buzz_dict_others = reacts_dict_w_punc[buzz]
		if buzz_dict_statuses["total"] == 0:
			buzz_dict_statuses["total"] = 1
			buzz_dict_statuses["total no likes"] = 1
		if buzz_dict_others["total"] == 0:
			buzz_dict_others["total"] = 1
			buzz_dict_others["total no likes"] = 1
		
		likes_statuses = np.append(likes_statuses, buzz_dict_statuses["likes"]/buzz_dict_statuses["total"])
		likes_others = np.append(likes_others, buzz_dict_others["likes"]/buzz_dict_others["total"])
		loves_statuses = np.append(loves_statuses, buzz_dict_statuses["loves"]/buzz_dict_statuses["total"])
		loves_others = np.append(loves_others, buzz_dict_others["loves"]/buzz_dict_others["total"])
		wows_statuses = np.append(wows_statuses, buzz_dict_statuses["wows"]/buzz_dict_statuses["total"])
		wows_others = np.append(wows_others, buzz_dict_others["wows"]/buzz_dict_others["total"])
		hahas_statuses = np.append(hahas_statuses, buzz_dict_statuses["hahas"]/buzz_dict_statuses["total"])
		hahas_others = np.append(hahas_others, buzz_dict_others["hahas"]/buzz_dict_others["total"])
		sads_statuses = np.append(sads_statuses, buzz_dict_statuses["sads"]/buzz_dict_statuses["total"])
		sads_others = np.append(sads_others, buzz_dict_others["sads"]/buzz_dict_others["total"])
		angrys_statuses = np.append(angrys_statuses, buzz_dict_statuses["angrys"]/buzz_dict_statuses["total"])
		angrys_others = np.append(angrys_others, buzz_dict_others["angrys"]/buzz_dict_others["total"])
		
		loves_adjusted_statuses = np.append(loves_adjusted_statuses, buzz_dict_statuses["loves"]/buzz_dict_statuses["total no likes"])
		loves_adjusted_others = np.append(loves_adjusted_others, buzz_dict_others["loves"]/buzz_dict_others["total no likes"])
		wows_adjusted_statuses = np.append(wows_adjusted_statuses, buzz_dict_statuses["wows"]/buzz_dict_statuses["total no likes"])
		wows_adjusted_others = np.append(wows_adjusted_others, buzz_dict_others["wows"]/buzz_dict_others["total no likes"])
		hahas_adjusted_statuses = np.append(hahas_adjusted_statuses, buzz_dict_statuses["hahas"]/buzz_dict_statuses["total no likes"])
		hahas_adjusted_others = np.append(hahas_adjusted_others, buzz_dict_others["hahas"]/buzz_dict_others["total no likes"])
		sads_adjusted_statuses = np.append(sads_adjusted_statuses, buzz_dict_statuses["sads"]/buzz_dict_statuses["total no likes"])
		sads_adjusted_others = np.append(sads_adjusted_others, buzz_dict_others["sads"]/buzz_dict_others["total no likes"])
		angrys_adjusted_statuses = np.append(angrys_adjusted_statuses, buzz_dict_statuses["angrys"]/buzz_dict_statuses["total no likes"])
		angrys_adjusted_others = np.append(angrys_adjusted_others, buzz_dict_others["angrys"]/buzz_dict_others["total no likes"])
	
	rgb_arrs = []
	diffs_arrs = []
	for i in range(len(best_words)):
		post_reacts_percentages = np.array([likes_statuses[i], loves_statuses[i], wows_statuses[i], hahas_statuses[i], sads_statuses[i], angrys_statuses[i]])
		punc_reacts_percentages = np.array([likes_others[i], loves_others[i], wows_others[i], hahas_others[i], sads_others[i], angrys_others[i]])
		if not show_likes:
			post_reacts_percentages = np.array([loves_adjusted_statuses[i], wows_adjusted_statuses[i], hahas_adjusted_statuses[i], sads_adjusted_statuses[i], angrys_adjusted_statuses[i]])
			punc_reacts_percentages = np.array([loves_adjusted_others[i], wows_adjusted_others[i], hahas_adjusted_others[i], sads_adjusted_others[i], angrys_adjusted_others[i]])
		diff_arr = punc_reacts_percentages-post_reacts_percentages
		rgbs = []
		for diff in diff_arr:
			p = diff
			rgb = (1, 1, 1)
			if diff > 0:
				scale = 1-p
				rgb = (1, scale, scale)
			elif diff < 0:
				scale = 1+p
				rgb = (scale, scale, 1)
			rgbs.append(rgb)
		rgb_arrs.append(rgbs)
		diff_perc = np.round(diff_arr, 7)*100
		diffs_arrs.append(diff_perc)

	for i in range(len(best_words)):
		slices = [(100/6), (100/6), (100/6), (100/6), (100/6), (100/6)]
		react_names = ['likes', 'loves', 'wows', 'hahas', 'sads', 'angrys']
		k = 6
		if not show_likes:
			slices = [(100/5), (100/5), (100/5), (100/5), (100/5)]
			react_names = react_names[1:]
			k = 5
		slice_texts = []
		for j in range(k):
			diff = diffs_arrs[i][j]
			txt = react_names[j]+"\n"+str(diff)
			slice_texts.append(txt)
		plt.pie(slices, labels=slice_texts, colors=rgb_arrs[i], startangle=90)
		# Set aspect ratio to be equal so that pie is drawn as a circle.
		plt.axis('equal')
		plt.title(best_words[i])

		# filename = "ccwp/"+best_words[i]+"_"+dataname		# deposits graphs in a folder named "ccwp"
		# plt.savefig(filename)
		# plt.close()
		plt.show()



def color_comp_w_caps(statuses, best_words, dataname, show_likes=False):
	reacts_dict = {}
	reacts_dict_w_caps = {}
	for buzz in best_words:
		reacts_dict[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}
		reacts_dict_w_caps[buzz] = {"likes":0, "loves":0, "wows":0, "hahas":0, "sads":0, "angrys":0, "total":0, "total no likes":0}

	for i in range(len(statuses)):
		for buzz in best_words:
			if buzz in str(statuses["status_message"][i]).lower():
				buzz_dict = reacts_dict[buzz]
				buzz_dict["likes"] += statuses['num_likes'][i]
				buzz_dict["loves"] += statuses['num_loves'][i]
				buzz_dict["wows"] += statuses['num_wows'][i]
				buzz_dict["hahas"] += statuses['num_hahas'][i]
				buzz_dict["sads"] += statuses['num_sads'][i]
				buzz_dict["angrys"] += statuses['num_angrys'][i]
				buzz_dict["total"] += statuses['num_reactions'][i]
				buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

				capped = buzz.upper()
				if capped in str(statuses["status_message"][i]):
						buzz_dict = reacts_dict_w_caps[buzz]
						buzz_dict["likes"] += statuses['num_likes'][i]
						buzz_dict["loves"] += statuses['num_loves'][i]
						buzz_dict["wows"] += statuses['num_wows'][i]
						buzz_dict["hahas"] += statuses['num_hahas'][i]
						buzz_dict["sads"] += statuses['num_sads'][i]
						buzz_dict["angrys"] += statuses['num_angrys'][i]
						buzz_dict["total"] += statuses['num_reactions'][i]
						buzz_dict["total no likes"] += (statuses['num_reactions'][i] - statuses['num_likes'][i])

	likes_statuses = np.array([])
	likes_others = np.array([])
	loves_statuses = np.array([])
	loves_others = np.array([])
	wows_statuses = np.array([])
	wows_others = np.array([])
	hahas_statuses = np.array([])
	hahas_others = np.array([])
	sads_statuses = np.array([])
	sads_others = np.array([])
	angrys_statuses = np.array([])
	angrys_others = np.array([])
	loves_adjusted_statuses = np.array([])
	loves_adjusted_others = np.array([])
	wows_adjusted_statuses = np.array([])
	wows_adjusted_others = np.array([])
	hahas_adjusted_statuses = np.array([])
	hahas_adjusted_others = np.array([])
	sads_adjusted_statuses = np.array([])
	sads_adjusted_others = np.array([])
	angrys_adjusted_statuses = np.array([])
	angrys_adjusted_others = np.array([])
	for buzz in best_words:
		buzz_dict_statuses = reacts_dict[buzz]
		buzz_dict_others = reacts_dict_w_caps[buzz]
		if buzz_dict_statuses["total"] == 0:
			buzz_dict_statuses["total"] = 1
			buzz_dict_statuses["total no likes"] = 1
		if buzz_dict_others["total"] == 0:
			buzz_dict_others["total"] = 1
			buzz_dict_others["total no likes"] = 1
		
		likes_statuses = np.append(likes_statuses, buzz_dict_statuses["likes"]/buzz_dict_statuses["total"])
		likes_others = np.append(likes_others, buzz_dict_others["likes"]/buzz_dict_others["total"])
		loves_statuses = np.append(loves_statuses, buzz_dict_statuses["loves"]/buzz_dict_statuses["total"])
		loves_others = np.append(loves_others, buzz_dict_others["loves"]/buzz_dict_others["total"])
		wows_statuses = np.append(wows_statuses, buzz_dict_statuses["wows"]/buzz_dict_statuses["total"])
		wows_others = np.append(wows_others, buzz_dict_others["wows"]/buzz_dict_others["total"])
		hahas_statuses = np.append(hahas_statuses, buzz_dict_statuses["hahas"]/buzz_dict_statuses["total"])
		hahas_others = np.append(hahas_others, buzz_dict_others["hahas"]/buzz_dict_others["total"])
		sads_statuses = np.append(sads_statuses, buzz_dict_statuses["sads"]/buzz_dict_statuses["total"])
		sads_others = np.append(sads_others, buzz_dict_others["sads"]/buzz_dict_others["total"])
		angrys_statuses = np.append(angrys_statuses, buzz_dict_statuses["angrys"]/buzz_dict_statuses["total"])
		angrys_others = np.append(angrys_others, buzz_dict_others["angrys"]/buzz_dict_others["total"])
		
		loves_adjusted_statuses = np.append(loves_adjusted_statuses, buzz_dict_statuses["loves"]/buzz_dict_statuses["total no likes"])
		loves_adjusted_others = np.append(loves_adjusted_others, buzz_dict_others["loves"]/buzz_dict_others["total no likes"])
		wows_adjusted_statuses = np.append(wows_adjusted_statuses, buzz_dict_statuses["wows"]/buzz_dict_statuses["total no likes"])
		wows_adjusted_others = np.append(wows_adjusted_others, buzz_dict_others["wows"]/buzz_dict_others["total no likes"])
		hahas_adjusted_statuses = np.append(hahas_adjusted_statuses, buzz_dict_statuses["hahas"]/buzz_dict_statuses["total no likes"])
		hahas_adjusted_others = np.append(hahas_adjusted_others, buzz_dict_others["hahas"]/buzz_dict_others["total no likes"])
		sads_adjusted_statuses = np.append(sads_adjusted_statuses, buzz_dict_statuses["sads"]/buzz_dict_statuses["total no likes"])
		sads_adjusted_others = np.append(sads_adjusted_others, buzz_dict_others["sads"]/buzz_dict_others["total no likes"])
		angrys_adjusted_statuses = np.append(angrys_adjusted_statuses, buzz_dict_statuses["angrys"]/buzz_dict_statuses["total no likes"])
		angrys_adjusted_others = np.append(angrys_adjusted_others, buzz_dict_others["angrys"]/buzz_dict_others["total no likes"])
	
	rgb_arrs = []
	diffs_arrs = []
	for i in range(len(best_words)):
		post_reacts_percentages = np.array([likes_statuses[i], loves_statuses[i], wows_statuses[i], hahas_statuses[i], sads_statuses[i], angrys_statuses[i]])
		cap_reacts_percentages = np.array([likes_others[i], loves_others[i], wows_others[i], hahas_others[i], sads_others[i], angrys_others[i]])
		if not show_likes:
			post_reacts_percentages = np.array([loves_adjusted_statuses[i], wows_adjusted_statuses[i], hahas_adjusted_statuses[i], sads_adjusted_statuses[i], angrys_adjusted_statuses[i]])
			cap_reacts_percentages = np.array([loves_adjusted_others[i], wows_adjusted_others[i], hahas_adjusted_others[i], sads_adjusted_others[i], angrys_adjusted_others[i]])
		diff_arr = cap_reacts_percentages-post_reacts_percentages
		rgbs = []
		for diff in diff_arr:
			p = diff
			if show_likes:
				p = diff*10
			rgb = (1, 1, 1)
			if diff > 0:
				scale = 1-p
				rgb = (1, scale, scale)
			elif diff < 0:
				scale = 1+p
				rgb = (scale, scale, 1)
			rgbs.append(rgb)
		rgb_arrs.append(rgbs)
		diff_perc = np.round(diff_arr, 7)*100
		diffs_arrs.append(diff_perc)

	for i in range(len(best_words)):
		slices = [(100/6), (100/6), (100/6), (100/6), (100/6), (100/6)]
		react_names = ['likes', 'loves', 'wows', 'hahas', 'sads', 'angrys']
		k = 6
		if not show_likes:
			slices = [(100/5), (100/5), (100/5), (100/5), (100/5)]
			react_names = react_names[1:]
			k = 5
		slice_texts = []
		for j in range(k):
			diff = diffs_arrs[i][j]
			txt = react_names[j]+"\n"+str(diff)
			slice_texts.append(txt)
		plt.pie(slices, labels=slice_texts, colors=rgb_arrs[i], startangle=90)
		# Set aspect ratio to be equal so that pie is drawn as a circle.
		plt.axis('equal')
		plt.title(best_words[i])

		# filename = "ccwc/"+best_words[i]+"_"+dataname		# deposits graphs in a folder named "ccwc"
		# plt.savefig(filename)
		# plt.close()
		plt.show()



def sent_reacts_ptest(statuses, dataname):
	sents = []
	z = 0
	for msg in statuses["status_message"]:
		z += 1
		sentiment = get_sentiment_intensity(str(msg))
		sents.append(sentiment["compound"])

	likes = []
	loves = []
	wows = []
	hahas = []
	sads = []
	angrys = []

	for i in range(len(statuses)):
		total = statuses["num_reactions"][i]
		likes.append(statuses["num_likes"][i]/total)
		loves.append(statuses["num_loves"][i]/total)
		wows.append(statuses["num_wows"][i]/total)
		hahas.append(statuses["num_hahas"][i]/total)
		sads.append(statuses["num_sads"][i]/total)
		angrys.append(statuses["num_angrys"][i]/total)

	indexsort = np.argsort(sents)
	sentsort = np.array(sents)[indexsort]
	mid_sent = 0
	for i in range(len(statuses)):
		mid_sent = i
		if sentsort[mid_sent] >= 0:
			break

	likesort = np.array(likes)[indexsort]
	lovesort = np.array(loves)[indexsort]
	wowsort = np.array(wows)[indexsort]
	hahasort = np.array(hahas)[indexsort]
	sadsort = np.array(sads)[indexsort]
	angrysort = np.array(angrys)[indexsort]

	pvals = []
	reactnames = ["like", "love", "haha", "wow", "sad", "angry"]
	reacts = [likesort, lovesort, hahasort, wowsort, sadsort, angrysort]
	for x in range(len(reacts)):
		reactsort = reacts[x]
		d = np.mean(reactsort[mid_sent:]) - np.mean(reactsort[0:mid_sent])
		sigdiffs = 0
		num_perms = 100000
		for j in range(num_perms):
		  perm = np.random.permutation(reactsort)
		  diff = np.mean(perm[mid_sent:]) - np.mean(perm[0:mid_sent])
		  if abs(diff) >= abs(d):
		    sigdiffs += 1
		pvalue = sigdiffs/num_perms
		pvals.append((reactnames[x], pvalue, d*100))

	pstring = str(np.array(pvals))
	# filename = "pvals/"+dataname+".txt"		# deposits results in a folder named "pvals"
	filename = dataname+".txt"
	with open(filename, "w") as text_file:
	    text_file.write("{0}".format(pstring))

#==========================MAIN============================

trump = status_cleaner("trump")
cnn = status_cleaner("cnn")
nyt = status_cleaner("nyt")
bw = ['great', 'our', 'america', 'thank', 'make', 'hillary', 'again', 'clinton', 'trump', 'terrorism', 'email', 'crime', 'terror', 'solve', 'people', 'lied', 'safe', 'president', 'jobs', 'innocent']

sent_reacts_ptest(trump, "trump")
sent_reacts_ptest(cnn, "cnn")
sent_reacts_ptest(nyt, "nyt")

post_times(trump, "trump")
post_times(cnn, "cnn")
post_times(nyt, "nyt")
post_types(trump, "trump")
post_types(cnn, "cnn")
post_types(nyt, "nyt")

reaction_bars(trump, bw[:10], "trump")
reaction_bars(trump, bw[10:20], "trump")
reaction_bars(trump, bw[20:30], "trump")
reaction_bars(trump, bw[30:40], "trump")
reaction_bars(trump, bw[40:50], "trump")
reaction_bars(cnn, bw[:10], "cnn")
reaction_bars(cnn, bw[10:20], "cnn")
reaction_bars(cnn, bw[20:30], "cnn")
reaction_bars(cnn, bw[30:40], "cnn")
reaction_bars(cnn, bw[40:50], "cnn")
reaction_bars(nyt, bw[:10], "nyt")
reaction_bars(nyt, bw[10:20], "nyt")
reaction_bars(nyt, bw[20:30], "nyt")
reaction_bars(nyt, bw[30:40], "nyt")
reaction_bars(nyt, bw[40:50], "nyt")

avgs = [0.79393068592057858, 0.30607414248021098, 0.73387016129032279, 0.78924821683309565, 0.75910629921259876, -0.016473449131513624, 0.77177003311258363, 0.0053227272727273019, 0.43017738927738908, -0.63219574468085105, -0.53196086956521738, -0.39987333333333336, -0.55101499999999992, -0.42077777777777775, 0.23233085937500003, -0.58294117647058818, 0.74478604651162783, 0.1969589285714286, 0.24334526315789481, -0.76023333333333332]
color_comp_to_whole(trump, bw, "trump")
color_comp_to_whole(cnn, bw, "cnn")
color_comp_to_whole(nyt, bw, "nyt")
color_comp_to_whole(trump, bw, "trump", sents=avgs)

color_comp_to_other(trump, cnn, bw, ["trump", "cnn"])
color_comp_to_other(trump, nyt, bw, ["trump", "nyt"])
color_comp_to_other(cnn, nyt, bw, ["cnn", "nyt"])
color_comp_to_other(trump, cnn, bw, ["trump", "cnn"], True)
color_comp_to_other(trump, nyt, bw, ["trump", "nyt"], True)
color_comp_to_other(cnn, nyt, bw, ["cnn", "nyt"], True)

color_comp_w_punc(trump, bw, "trump")
color_comp_w_punc(cnn, bw, "cnn")
color_comp_w_punc(nyt, bw, "nyt")

color_comp_w_caps(trump, bw, "trump")
color_comp_w_caps(cnn, bw, "cnn")
color_comp_w_caps(nyt, bw, "nyt")




