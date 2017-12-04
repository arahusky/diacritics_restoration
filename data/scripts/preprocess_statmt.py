import io

'''
Remove invalid UTF-8, remove all wiki data and addtional information (e.g. web the page was extracted from)
'''


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to statmt extracted file.")
    parser.add_argument("output_file", type=str, help="Path to result (filtered) file.")
    args = parser.parse_args()

    with io.open(args.input_file, encoding='utf8', mode='r') as reader:
        with io.open(args.output_file, 'w', encoding='utf8') as writer:
            skipping = True  # whether current url is wiki

            while True:
                try:
                    line = reader.readline()
                except UnicodeDecodeError as e:
                    print('skipping -- not utf8')
                    continue

                if not line:
                    break

                line = line.strip()

                if line.startswith('df6fa1abb58549287111ba8d776733e9'):
                    if line.startswith('df6fa1abb58549287111ba8d776733e9 uri:https://cs.wikipedia.org/'):
                        # ignore wikipedia texts
                        # writer.write('{}\n'.format(line))
                        skipping = True
                    else:
                        skipping = False
                    # writer.write('\n')
                elif not skipping:
                    writer.write('{}\n\n'.format(line))
                else:
                    writer.write('{}\n\n'.format(line))


if __name__ == "__main__":
    main()
