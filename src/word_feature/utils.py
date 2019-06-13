import sys
sys.path.append('../')
import string
import config as cf

def read_caption_raw_file(filename):
    """
        Read the description file
        and create a dictionary to save all caption
    """
    file = open(filename, 'r')
    descriptions = dict()
    for line in file:
        # make id and value
        tokens = line.split()
        # print('Token: ', tokens)
        if any(tokens):
            #print('Token: ', tokens)
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            image_desc = ' '.join(image_desc)
            # append into dict
            if image_id not in [*descriptions]:
                descriptions[image_id] = list()
            descriptions[image_id].append(clean_text(image_desc))
    return descriptions

def read_caption_clean_file(filename):
    """
        Read the description file
        and create a dictionary to save all caption
    """
    file = open(filename, 'r')
    descriptions = dict()
    for line in file:
        # make id and value
        tokens = line.split()
        # print('Token: ', tokens)
        if any(tokens):
            #print('Token: ', tokens)
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            image_desc = ' '.join(image_desc)
            # append into dict
            if image_id not in [*descriptions]:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
    return descriptions

def clean_text(text):
    table = str.maketrans('', '', string.punctuation)
    words = text.split()
    # lower case
    words = [word.lower() for word in words]
    # remove punctuation
    words = [word.translate(table) for word in words]
    # remove standalone characters
    words = [word for word in words if len(word)>1]
    # remove word with number
    words = [word for word in words if word.isalpha()]
    #add startseq and endseq
    words.insert(0, 'startseq')
    words.append('endseq')
    return ' '.join(words)

def get_unique_word(str_list):
    unique = dict()
    for text_list in str_list:
        for text in text_list:
            words = text.split()
            for word in words:
                if word not in unique:
                    unique[word] = 1
                else:
                    unique[word] += 1
    # save only common word
    unique_list = list()
    # print('Dict: ', unique)
    unique_list = [k for k in [*unique] if unique[k]>=cf.word_threshold]
    return unique_list

def map_w2id(str_list):
    """
        Give you 2 dict of mapping word to idx | idx to word | vocab size
    """
    unique = get_unique_word(str_list)
    idxtoword = dict() # ID to words
    wordtoidx = dict() # Words to ID
    idx = 1
    for word in unique:
        idxtoword[idx] = word
        wordtoidx[word] = idx
        idx += 1
    return idxtoword, wordtoidx, len(unique)+1

def analyze_captions(descriptions):
    """
        We eleminate uncommon words
    """
    unique = get_unique_word(descriptions.values())
    #print('Allowed: ', unique)
    for k,v in descriptions.items():
        for i in range(len(v)):
            words = v[i].split()
            words = [word for word in words if word in unique]
            words = ' '.join(words)
            descriptions[k][i] = words

def caculate_caption_max_len(str_list):
    max_len = -1
    for text_list in str_list:
        for text in text_list:
            words = text.split()
            if max_len < len(words): max_len=len(words)
    return max_len

def save_captions(descriptions, filename):
    file = open(filename, 'w')
    for k, v in descriptions.items():
        img = k + '.jpg'
        for caption in v:
            file.write(img)
            file.write('\t')
            file.write(caption)
            file.write('\n')

if __name__ == '__main__':
    desc = read_caption_raw_file('../Flickr8k_text/Flickr8k.token.txt')
    analyze_captions(desc)
    print(caculate_caption_max_len(desc.values()))
    save_captions(desc, '../Flickr8k_text/Flickr8k.cleaned.token.txt')
