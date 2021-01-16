# %%
# create a string

import re
amount = u"₹50"
print('Default string: ', amount, '\n', 'Type of string', type(amount), '\n')

# encode to UTF-8 byte format
amount_encoded = amount.encode('utf-8')
print('Encoded to UTF-8: ', amount_encoded, '\n', 'Type of string', type(amount_encoded), '\n')


# sometime later in another computer...
# decode from UTF-8 byte format
amount_decoded = amount_encoded.decode('utf-8')
print('Decoded from UTF-8: ', amount_decoded, '\n', 'Type of string', type(amount_decoded), '\n')

# %%

# %%
# import the regular expression module

string = 'I saw three trees.'
string = 'I just turned 33.'
# string = 'I saw three tree.'
# regex pattern
pattern = "tree?"  # write regex here

# check whether pattern is present in string or not
result = re.search(pattern, string)  # pass the arguments to the re.search function

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
if result != None:
    print(True)
else:
    print(False)

# %%
string = 'tyz'

# regex pattern
pattern = 'xy?z?'  # write regex pattern here

# check whether pattern is present in string or not
for string in [
    "xyz", "xy",
    "xz",
    "x",
    "Xyyz",
    "Xyzz",
    "Xyy",
    "Xzz",
        "Yz"]:

    result = re.search(pattern, string)

    # evaluate result - don't change the following piece of code, it is used to evaluate your regex
    if result != None:
        print(True)
    else:
        print(False)
# %%

for string in [
    "110",
    "11111110",
    "10",
    "11",
    "00",
    "1",
    "0"
]:

    # regex pattern
    pattern = '1?(0|1)*0?'  # write regex pattern here

    # check whether pattern is present in string or not
    result = re.search(pattern, string)

    # evaluate result - don't change the following piece of code, it is used to evaluate your regex
    if result != None:
        print(True)
    else:
        print(False)
# %%
 # regex pattern
pattern = r'ab{0,3}\b'  # write your regex here

for string in [
    "a",
    "ab",
    "abb",
    "abbb",
    "abbbb",
    "abbbbbbbb"
]:

    # check whether pattern is present in string or not
    result = re.search(pattern, string)

    # evaluate result - don't change the following piece of code, it is used to evaluate your regex
    if result != None:
        print(True)
    else:
        print(False)
# %%
string = "Building careers of tomorrow"
pattern = "^.{1}"
replacement = "$"
re.sub(pattern, replacement, string)

# %%
string = "Building careers of tomorrow"
pattern = r"\b\w+\b"
re.findall(pattern, string)
# %%
string = "Playing outdoor games when its raining outside is always fun!"
# pattern = r"\b(\w+ing)\b"
pattern = r"\w+ing"
print(re.findall(pattern, string))
# result = re.findall(pattern, string)
# print(len(result))
# %%

string = "Today’s date is 18-05-2018."

# regex pattern
pattern = r'(\d{1,2})-(\d{1,2})-(\d{4})'  # write regex to extract date in DD-MM-YYYY format

# store result
result = re.search(pattern, string)  # pass the parameters to the re.search() function

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
if result != None:
    print(result.group(0))  # result.group(0) will output the entire match
else:
    print(False)
# %%
string = "user_name_123@gmail.com"

# regex pattern
pattern = r'@+(.)*\.com$'  # write regex to extract email and use groups to extract domain name ofthe mail
pattern = "\w+@([A-z]+\.com)"  # write regex to extract email and use groups to extract domain name ofthe mail

# store result
result = re.search(pattern, string)
print(f'result: {result}')
# extract domain using group command
if result != None:
    # domain = result.group(1)  # use group to extract the domain from result
    domain = result.group(0)[1:]  # use group to extract the domain from result
else:
    domain = "NA"

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
print(domain)
# %%
# items contains all the files and folders of current directory
items = ['photos', 'documents', 'videos', 'image001.jpg', 'image002.jpg', 'image005.jpg', 'wallpaper.jpg',
         'flower.jpg', 'earth.jpg', 'monkey.jpg', 'image002.png']

# create an empty list to store resultant files
images = []

# regex pattern to extract files that end with '.jpg'
pattern = ".*\.jpg$"

for item in items:
    if re.search(pattern, item):
        images.append(item)

# print result
print(images)
# %%
# items contains all the files and folders of current directory
items = ['photos', 'documents', 'videos', 'image001.jpg', 'image002.jpg', 'image005.jpg', 'wallpaper.jpg',
         'flower.jpg', 'earth.jpg', 'monkey.jpg', 'image002.png']

# create an empty list to store resultant files
images = []

# regex pattern to extract files that start with 'image' and end with '.jpg'
pattern = "image.*\.jpg$"

for item in items:
    if re.search(pattern, item):
        images.append(item)

# print result
print(images)
# %%
