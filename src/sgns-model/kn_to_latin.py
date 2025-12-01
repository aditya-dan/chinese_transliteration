consonants = "ಕಖಗಘಞಚಛಜಝಙಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ"

map = {
       'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ii', 'ಉ': 'u', 'ಊ': 'uu', 'ಋ': 'ru', 'ಎ': 'e', 'ಏ': 'ee', 'ಒ': 'o', 'ಓ': 'oo', 'ಐ': 'ai', 'ಔ': 'au',
       'ಾ': 'aa', 'ಿ': 'i', 'ೀ': 'ii', 'ು': 'u', 'ೂ': 'uu', 'ೃ': 'ru', 'ೆ': 'e', 'ೇ': 'ee', 'ೊ': 'o', 'ೋ': 'oo', 'ೈ': 'ai', 'ೌ': 'au',
       'ಂ': 'an', 'ಃ': 'aha', '್': '',
       'ಕ': 'k', 'ಖ': 'kh', 'ಗ': 'g', 'ಘ': 'gh', 'ಙ': 'n',
       'ಚ': 'ch', 'ಛ': 'chh', 'ಜ': 'j', 'ಝ': 'jh', 'ಙ': 'n',
       'ಟ': 't', 'ಠ': 'th', 'ಡ': 'd', 'ಢ': 'dh', 'ಣ': 'n',
       'ತ': 't', 'ಥ': 'th', 'ದ': 'd', 'ಧ': 'dh', 'ನ': 'n',
       'ಪ': 'p', 'ಫ': 'ph', 'ಬ': 'b', 'ಭ': 'bh', 'ಮ': 'm',
       'ಯ': 'y', 'ರ': 'r', 'ಲ': 'l', 'ವ': 'v',
       'ಶ': 'sh', 'ಷ': 'sh', 'ಸ': 's', 'ಹ': 'h', 'ಳ': 'l'}

def kn_to_latin(input_word):
    output = ""
    for index, character in enumerate(input_word):
        if character not in map.keys():
            output = output + character
        else:
            output = output + map[character]
            if index == len(input_word) - 1 and character in consonants:
                output = output + "a"
            else:
                if character in consonants and input_word[index + 1] in consonants:
                    output = output + "a"

    return output