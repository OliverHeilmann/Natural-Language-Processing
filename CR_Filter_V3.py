"""
Created 10/10/2019 by Oliver Heilmann

The intent is to create a simpler and more sophisticated version over the previous. More general functions shall be used
in order to simplify the code and keep clarity throughout.
"""

# import relevant modules
from itertools import product
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import csv, pdb, copy


# Extract Info of .CSV (can select all or specific column)
def extract_csv(path='ExampleCR.csv', column=False):
    data = []
    # encoding="ISO-8859-1"
    with open(path, encoding="utf8") as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            if not column:
                filtered = [i for i in row if i != '']
                data.append(filtered)   
            else:
                data.append(row[column])
    return data


# Determine Part of Speech and Feed to Lemmatizer
def speech_to_lem(text_set):
    # Setup Lemmatizer (which will stem words)
    lem = WordNetLemmatizer()

    # Determine Part of Speech- PoS (eg Noun/Adjective/Verb)
    pos = [(wo, wn.synsets(wo)[0].pos()) for wo in text_set if len(wn.synsets(wo)) > 0 and isinstance(wo, str)]

    # Stem words in PoS with their appropriate parts of speech. Remaining
    # two letter words are deleted here too as they do not carry relevant
    # detail to the text.
    stemmed = [lem.lemmatize(wo[0], pos=wo[1]) for wo in pos if len(wo[0]) > 2]  # LOOK AT THE POS=W[1] PART... WRONG?
    return stemmed


# Filter words from a note- these are also stemmed. One may pass either a string (note) or a list of words (keywords).
# This is detected by the script automatically...
def filter_stem_words(input_text, delete_words):
    # empty list to add all passed words from sentences
    passed_text = []
    if type(input_text) is str:
        # break string into word list: make lowercase, delete stop-words and check if this is a string
        process_list = [wo.lower() for wo in word_tokenize(input_text)
                        if wo not in delete_words and type(wo) is str]
    else:
        # already list: make lowercase, delete stop-words and check if this is a string
        process_list = [wo.lower() for wo in input_text
                        if wo not in delete_words and type(wo) is str]

    # 'process' is a list of words- this line runs through list and deletes numbers from each individual word.
    process_list = [''.join([char for char in word if not char.isdigit()]) for word in process_list]

    # function finds part of speech and uses it to convert to root word- returns roots
    root_words = speech_to_lem(process_list)

    # append all words that pass filter etc. to a global list
    for passed in root_words:
        passed_text.append(passed)
    return passed_text


# Assign weighting to each word. If a word has the same count as another, they are given the same weighting (see
# OneNote page for further explanation of this).
def assign_list_weighting(weighted_list, list_type, flip=False):
    num = float(1.0)
    for j in range(0, len(weighted_list)):
        # if the count is unique/ non unique, change value of 'num'
        if j == 0 or weighted_list[j][-1] == weighted_list[j-1][-1]:
            weighted_list[j].insert(0, round(num, 1))
        else:
            num += 0.1
            weighted_list[j].insert(0, round(num, 1))
    if flip:
        # Used for largest keywords first
        weighted_list = sorted(weighted_list, key=lambda m: m[-1], reverse=True)
    # Calculate weighting using relevant calculation (according to whether it is a note_word list or key_word list)
    # List structure as [weighting, word]
    if list_type == 'note_words':
        weight = [[round(weighted_list[jj][-1]-weighted_list[jj][0], 1), weighted_list[jj][1]]
                  for jj in range(0, len(weighted_list))]
    elif list_type == 'key_words':
        weight = [[round(weighted_list[jj][-1]*20*weighted_list[jj][0], 1), weighted_list[jj][1]]
                  for jj in range(0, len(weighted_list))]
    else:
        print('Not Acceptable Argument')
    return weight


# Count the number of repeated words in a given text. This works for keywords list and notes (as long as it has been
# broken into individual words- see other functions for this capability).
def count_note_repeats(word_list):
    my_dict = {}
    for word in word_list:
        if word not in my_dict.keys():
            my_dict[word] = int(1)
        else:
            count = my_dict[word]
            my_dict[word] = count + int(1)
    # take dictionary and append results to an ordered list.
    word_list = [[key, my_dict.get(key)] for key in my_dict.keys()]
    # order words into new list
    word_list_ordered = sorted(word_list, key=lambda m: m[1], reverse=True)
    # assign weighting to each word (list structure as [weighting, word])
    weighted_list = assign_list_weighting(word_list_ordered, list_type='note_words')
    return weighted_list


# Count the keywords
def count_keywords(note_words, keywords):
    word_dict = []
    for word in keywords:
        # check to see if word in keyword list
        if word in note_words:
            word_dict.append([word, note_words.count(word)])
    # order keywords that appeared in note
    keywords_ordered = sorted(word_dict, key=lambda m: m[1])
    # assign weighting to each word (list structure as [weighting, word])
    weighted_list = assign_list_weighting(keywords_ordered, list_type='key_words', flip=True)
    return weighted_list


# Takes individual weightings of the word lists and combines for an overall weighting (see OneNote Document)
def weight_calculator(note_weights_orig, key_weights):
    # Create a copy of note_weights to add values to
    note_weights = copy.deepcopy(note_weights_orig)
    # Extract note words to separate list (easier to filter through than entering each sub list with another for loop)
    note_words = [ii[1] for ii in note_weights]
    for group in key_weights:
        # If word from keywords is in note_words, the weightings should be added together. This function replaces the
        # weighting of the note_weights list with the new weighting rather than appending to a new list as this method
        # is computationally faster.
        if group[1] in note_words:
            pos = note_weights[note_words.index(group[1])][0]
            pos += group[0]
            note_weights[note_words.index(group[1])][0] = pos
    # Sort list again for finalised weighted list (only top 10)
    weights = sorted(note_weights, key=lambda m: m[0], reverse=True)[:10]
    # Determine how many repeated top weightings there are
    top_repeats = len([j[0] for j in weights if weights[0][0]-j[0]==float(0)])
    return weights, top_repeats


# Take two lists of words and find the pair which are most similar. Note: if 
# there are duplicate matches (i.e. both have same similarity), this code does
# not account for it. It is assumed that the input lists are constructed to
# have distinctly different words (hence called catagories of CRs).
def categorise(weighted_words, catwords):
    if len(weighted_words) > 0 and len(catwords) > 0:
        # Find all synyonyms for all the words in each list
        allsyns1 = [ss for word in weighted_words for ss in wn.synsets(word)]
        allsyns2 = [ss for word in catwords for ss in wn.synsets(word)]

        # Find the best matched synoyms in the two lists (product rule is used to 
        # make every comparison)
        best_syn_match = max([(wn.wup_similarity(s1, s2) or 0, s1, s2) 
                    for s1, s2 in product(allsyns1, allsyns2)])
        
        # Function to relate the synonym back to parent word
        def parent(word_list, loc):
            # Create a list with the number of syns per parent word. Once the 
            # index is found, take value and return parent word (see function 
            # input parameters in following lines)
            syn_size = [len(wn.synsets(word)) for word in word_list]
            i = 0; j = 0
            while i <= loc:
                i += syn_size[j]
                j += 1
            return word_list[j-1]
        
        # Call above function, inputs are lists and index for best match synonym
        p1 = parent(weighted_words, allsyns1.index(best_syn_match[1]))
        p2 = parent(catwords, allsyns2.index(best_syn_match[2]))
    
        return [p1,p2]
    else:
        return [None, None]


'''

THE MAIN SCRIPT STARTS BELOW THIS LINE. ALL FUNCTIONS ABOVE ARE DEFINED IN THE ABOVE SECTIONS.

'''


# MAIN LOOP BELOW
if __name__ == "__main__":
    # Define Stop Words to be Excluded
    stop_words = set(stopwords.words('english'))

    # Define your keywords below. Keywords are passed through the filter in order to ensure all are in fact roots (the
    # same will be conducted for the notes).
    keyWords = ['standard', 'tolerance', 'clarity', 'requirement', 'reference',
                'machine', 'manufacturing', 'forge', 'etch', 'customer', 'contract',
                'forge', 'NCR', 'paint']
    keyWords = filter_stem_words(keyWords, delete_words=stop_words)  # stemming word list

    # Create category list to sort CRs into
    #catWords = ['standard', 'tolerance', 'clarity', 'need', 'refer', 'paint']
    #catWords = filter_stem_words(catWords, delete_words=stop_words)  # stemming word list
    
    # Create category list to sort CRs into
    cats = extract_csv('Cats_Subcats.csv')
    catWords = []
    for i in cats:
        for j in i:
            if j != i[0] and j not in catWords:
                catWords.append(j)
    catWords = filter_stem_words(catWords, stop_words)

    # Make a dictionary for counting reasons for raising CRs
    CR_cat_counts = {}
    for i in catWords:
        CR_cat_counts[i] = 0

    # Extract Notes info from Row[x] in .CSV
    notes = extract_csv(column=int(6))

    # This is the main loop which iterates through all notes. All functions above are used within this main loop
    for text in notes[1:]:
        # extract text from note and filter to stemmed & descriptive words only
        filtered_list = filter_stem_words(text, delete_words=stop_words)  # stemming one string

        # take filtered list and count repeated words within
        n = count_note_repeats(filtered_list)

        # take filtered list and find keywords within
        k = count_keywords(filtered_list, catWords + keyWords)

        # calculate the weightings of words from both lists created above
        w, tr = weight_calculator(n, k)

        # display results from above functions
        print('WEIGHTED NOTE REPEATS (after subtraction):\n{}\n'.format(n))
        print('WEIGHTED KEYWORDS (after multiplication):\n{}\n'.format(k))
        print('WEIGHTED LIST (combinations of above):\n{}\n'.format(w))
        
        # if there is more than one top word, sort both in to categories
        for i in range(0,tr):
            c = categorise([w[i][1]], catWords)
            #pdb.set_trace()
            CR_cat_counts[c[1]] += 1
            if i > 0:
                print('count higher than 1 for CR\n')
            print('WEIGHTED WORD SORTED TO ({}):\n{}\n'.format(i+1,c))
        print('\n\n\n')
        

        '''
        NOTE!!!
        Keywords are words that are known to be reasons for raising a CR. These
        words will affect the weighting of the CR note, however, are not the 
        categories that the CRs will be filtered into. It is important to choose
        good keywords as well as good categories for a successful analysis.
        '''
    
    # Print the final organised CR types (in dictionary format)
    print('CR TYPES GROUPED TOGETHER:\n{}\n\n\n'.format(CR_cat_counts))
        
        
        
        
        
        
        
        
        
        
        