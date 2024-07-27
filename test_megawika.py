from lmft.data_io.megawika import load_megawika_data


train, dev = load_megawika_data(
    1, 'meta-llama/Meta-Llama-3.1-8B-Instruct', 2048,
    True, True, 100, True
)

for x in train:
    z = 1
