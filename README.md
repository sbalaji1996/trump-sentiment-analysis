# About

In this study, we investigate the rhetorical effects that Donald Trump’s 
Facebook statuses have on his social media followers. We begin with sentiment 
analysis of posts published over the course of his 2016 presidential campaign and 
train a classifier to determine natural language sentiment within close proximity of 
human performance. We then create a logistic regression model to prove that the 
presence certain keywords in a status can predict the overall valence of that post. In 
order to confirm our prediction that the emotional rhetoric of a status corresponds 
with users’ reactions to that post, we conduct permutation tests to ascertain the 
correlation between sentiment and particular reactions. Then, using the highvalence
indicative words yielded by our classifier, we produce visualizations that 
relate those keywords to users’ reactions on posts made by the Facebook pages for 
Donald Trump, CNN, and the New York Times. We then analyze that visual data to 
draw conclusions about Trump’s rhetoric and the effects it has on the response of 
his followers. Ultimately we find that, similar to their associations with sentiment, 
certain keywords can predict the reaction distribution of a post. We also find that 
irregular formatting, such as special punctuation and entire-word capitalization, can 
have a magnifying effect on certain reactions.

# Results

Our findings support our 3 hypotheses and allow us make strong assertions 
about the relationship between rhetoric and response in social media. We can 
confidently say that the sentiment of one’s posts will have a predictable effect on the 
reactions to those posts, thus confirming our framework hypothesis. We were able 
to select features from the text in such a manner that our logistic regression model 
achieved an accuracy rating of 78.66%, thereby confirming our statistical hypothesis.
Furthermore, we can predict that posts with special formatting will have a 
modulating response on the reactions to those posts, thus supporting our cognitiveaffective
hypothesis. Finally, our findings give us insight into the effect that Trump’s 
rhetoric has on his followers as well as the nature of their support. The most 
significant finding is that Trump supporters are intensely fond of him, displaying 
their overwhelming “love” on his Facebook posts. Another interesting finding is that 
objectively unassuming language, such as “our” and “make”, is made exceptional by 
Trump’s simple yet absolutist rhetoric, which elicits a powerful response from his 
followers. Lastly, we can say that Trump’s rhetoric on social media is deeply 
entrenched in emotional sentiment, and thus a potent tool for galvanizing his 
conservative base and rallying millions of supporters.

# Acknowledgements

Professor Yang Xu, U.C Berkeley

Max Woolf: https://github.com/minimaxir/facebook-page-post-scraper

# Packages Required

To run these scripts, you need to have the following packages installed: 

1) nltk: www.nltk.org

2) pandas: pandas.pydata.org

3) numpy: www.numpy.org

4) matplotlib: matplotlib.org
