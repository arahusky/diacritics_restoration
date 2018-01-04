from . import utils


def c2c_per_char_accuracy(sentences, lengths, targets, vocabulary):
    num_chars = 0.0
    num_correct_chars = 0.0

    for sentence, length, target in zip(sentences, lengths, targets):
        for system_char, gold_char in zip(sentence[:length], target[:length]):
            num_chars += 1
            if system_char == gold_char:
                num_correct_chars += 1

    return num_correct_chars / num_chars


def c2c_per_word_accuracy(sentences, lengths, targets, vocabulary, whitespace_to_whitespace=True):
    total_words = 0.0
    words_correct = 0.0

    inverted_vocab = utils.invert_vocabulary(vocabulary)

    debug = 0

    print(len(sentences), len(lengths), len(targets))
    for system, length, gold in zip(sentences, lengths, targets):
        system_sentence, gold_sentence = '', ''
        for system_int, gold_int in zip(system[:length], gold[:length]):
            gold_char = inverted_vocab[gold_int]
            system_char = inverted_vocab[system_int]

            if whitespace_to_whitespace:
                if gold_char.isspace() and not system_char.isspace():
                    system_char = gold_char
                elif system_char.isspace() and not gold_char.isspace():
                    system_char = '@'

            system_sentence += system_char
            gold_sentence += gold_char

        # if debug < 1:
        #     print(sentences[0])
        #     print(targets[0])
        #     print(lengths[0])
        #     print(system_sentence)
        #     print(gold_sentence)
        #     debug += 1
        gold, system = map(lambda s: s.split(" "), [gold_sentence, system_sentence])
        total_words += len(gold)

        if len(gold) != len(system):
            # consider all words being wrong
            continue

        # compute word accuracy
        for i in range(len(gold)):
            if gold[i] == system[i]:
                words_correct += 1

    return words_correct / total_words
