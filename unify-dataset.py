import os


def save_lines(path_name, sentences):
    with open(path_name, "w", encoding="utf-8") as f_es:
        for line in sentences:
            f_es.write(line)


if __name__ == '__main__':
    data_folder = "europarl"
    data_folder_unified = "europarl_unified"

    sentences_es = []
    sentences_and = []
    sentences_andh = []
    sentences_ands = []
    sentences_andz = []

    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f_opened:
            sentences = f_opened.readlines()

            if filename.endswith("and"):
                sentences_and.extend(sentences)
            elif filename.endswith("andh"):
                sentences_andh.extend(sentences)
            elif filename.endswith("ands"):
                sentences_ands.extend(sentences)
            elif filename.endswith("andz"):
                sentences_andz.extend(sentences)
            else:
                sentences_es.extend(sentences)

    os.makedirs(data_folder_unified, exist_ok=True)

    save_lines(os.path.join(data_folder_unified, "sentences_es"), sentences_es)
    save_lines(os.path.join(data_folder_unified, "sentences_and"), sentences_and)
    save_lines(os.path.join(data_folder_unified, "sentences_andh"), sentences_andh)
    save_lines(os.path.join(data_folder_unified, "sentences_ands"), sentences_ands)
    save_lines(os.path.join(data_folder_unified, "sentences_andz"), sentences_andz)
