
def load_data(data_path):
    data = []
    path = data_path

    with open(path, "r") as f:
        lines = f.readlines()

        for line in lines:
            tmp = line.split("\t")
            qa = tmp[1].split("?")

            if len(qa) == 3:
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()

            data_sample = {
                'question': qa[0] + '?',
                'image_path': tmp[0][:-2],
                'answer': answer
            }

            data.append(data_sample)

    return data
